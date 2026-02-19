import time, argparse, os.path as osp, os
import torch, numpy as np
import torch.distributed as dist
import cv2

from PIL import Image
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor

import warnings
warnings.filterwarnings("ignore")

from misc.rerun_vis import (
    init_rerun,
    log_occ,
    log_gaussian,
    log_gaussian_topdown,
)
from misc.open3d_raycast_vis import (
    save_open3d_occ_video,
    ray_cast_occ,
    tensor_to_numpy_occ,
    render_occ_snapshot,
)


def pass_print(*args, **kwargs):
    pass


def _create_video_writer(path, fps, size):
    """Try multiple codecs to improve portability on headless servers."""
    os.makedirs(osp.dirname(path), exist_ok=True)
    candidates = [
        ("mp4v", path),
        ("avc1", path),
        ("MJPG", path.replace(".mp4", ".avi")),
    ]
    for fourcc_str, out_path in candidates:
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*fourcc_str),
            float(fps),
            (int(size[0]), int(size[1])),
        )
        if writer.isOpened():
            return writer, out_path, fourcc_str
    return None, None, None


def main(local_rank, args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir
    os.makedirs(args.work_dir, exist_ok=True)

    # init DDP
    if args.gpus > 1:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "20507")
        hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
        rank = int(os.environ.get("RANK", 0))  # node id
        gpus = torch.cuda.device_count()  # gpus per node
        print(f"tcp://{ip}:{port}")
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}", 
            world_size=hosts * gpus, rank=rank * gpus + local_rank)
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)

        if local_rank != 0:
            import builtins
            builtins.print = pass_print
    else:
        distributed = False
        world_size = 1
    
    writer = None
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger('selfocc', log_file=log_file)
    MMLogger._instance_dict['selfocc'] = logger
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    import model
    from dataset import get_dataloader

    my_model = build_segmentor(cfg.model)
    my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    if distributed:
        if cfg.get('syncBN', True):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            logger.info('converted sync bn.')

        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        raw_model = my_model.module
    else:
        my_model = my_model.cuda()
        raw_model = my_model
    logger.info('done ddp model')

    vis_indices = args.vis_index
    if len(vis_indices) == 0 and args.num_samples > 0:
        vis_indices = list(range(args.vis_start_index, args.vis_start_index + args.num_samples))

    cfg.val_dataset_config.update({
        "vis_indices": vis_indices,
        "num_samples": args.num_samples,
        "vis_scene_index": args.vis_scene_index})

    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=distributed,
        val_only=True)
    
    # resume and load
    cfg.resume_from = ''
    latest_ckpt = osp.join(args.work_dir, 'latest.pth')
    if osp.exists(latest_ckpt):
        cfg.resume_from = latest_ckpt
    if args.resume_from:
        cfg.resume_from = args.resume_from
        if not osp.exists(cfg.resume_from):
            raise FileNotFoundError(
                f"--resume-from file does not exist: {cfg.resume_from}. "
                f"Check the checkpoint path or use {latest_ckpt} if available."
            )
    
    logger.info('resume from: ' + cfg.resume_from)
    logger.info('work dir: ' + args.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        try:
            # raw_model.load_state_dict(ckpt['state_dict'], strict=True)
            raw_model.load_state_dict(ckpt.get('state_dict', ckpt), strict=True)
        except:
            os.system(f"python modify_weight.py --work-dir {args.work_dir} --epoch {args.epoch}")
            cfg.resume_from = os.path.join(args.work_dir, f"epoch_{args.epoch}_mod.pth")
            ckpt = torch.load(cfg.resume_from, map_location=map_location)
            raw_model.load_state_dict(ckpt['state_dict'], strict=True)
        print(f'successfully resumed.')
    elif cfg.load_from:
        if not osp.exists(cfg.load_from):
            raise FileNotFoundError(
                f"Neither a valid resume checkpoint nor load_from exists. "
                f"Missing cfg.load_from: {cfg.load_from}"
            )
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        print(raw_model.load_state_dict(state_dict, strict=False))
        
    print_freq = cfg.print_freq
    from misc.metric_util import MeanIoU
    miou_metric = MeanIoU(
        list(range(1, 17)),
        17, #17,
        ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation'],
         True, 17, filter_minmax=False)
    miou_metric.reset()

    my_model.eval()
    os.environ['eval'] = 'true'
    save_dir = os.path.join(args.work_dir, 'vis')
    if args.vis_tag:
        save_dir = os.path.join(save_dir, args.vis_tag)
    if args.save_ori_images or args.vis_open3d_video or args.vis_raycast:
        os.makedirs(save_dir, exist_ok=True)

    raycast_writer = None
    if local_rank == 0 and args.vis_raycast:
        raycast_video_path = os.path.join(save_dir, "raycast_pred_vs_gt.mp4")
        raycast_writer = cv2.VideoWriter(
            raycast_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(args.raycast_video_fps),
            (1600, 800),
        )
        if not raycast_writer.isOpened():
            raise RuntimeError(f"Failed to open raycast video writer: {raycast_video_path}")

    occ_panel_writer = None
    if local_rank == 0 and args.vis_occ_panel:
        panel_h = int(args.panel_top_height + args.panel_mid_height)
        panel_w = int(args.panel_width)
        panel_video_path = os.path.join(save_dir, "occ_panel_video.mp4")
        occ_panel_writer, panel_video_path, panel_codec = _create_video_writer(
            panel_video_path, args.panel_video_fps, (panel_w, panel_h)
        )
        if occ_panel_writer is None:
            logger.warning(
                "Failed to open panel video writer with available codecs. "
                "Continuing with image-only output in save_dir."
            )
        else:
            logger.info(f'Panel video writer opened: {panel_video_path} (codec={panel_codec})')
    use_rerun = (args.vis_occ or args.vis_gaussian or args.vis_gaussian_topdown) and local_rank == 0
    if use_rerun:
        rerun_info = init_rerun(
            app_id=args.rerun_app_id,
            mode=args.rerun_mode,
            connect_addr=args.rerun_connect_addr,
            save_path=args.rerun_save_path,
            open_browser=args.rerun_open_browser,
            web_port=args.rerun_web_port,
        )
        logger.info(
            f'Rerun initialized: {rerun_info}'
        )
    if args.model_type == "base":
        draw_gaussian_params = dict(
            scalar = 1.5,
            ignore_opa = False,
            filter_zsize = False
        )
    elif args.model_type == "prob":
        draw_gaussian_params = dict(
            scalar = 2.0,
            ignore_opa = True,
            filter_zsize = True
        )

    with torch.no_grad():
        for i_iter_val, data in enumerate(val_dataset_loader):
            
            for k in list(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda()
            input_imgs = data.pop('img')
            ori_imgs = data.pop('ori_img')
            if args.save_ori_images:
                for i in range(ori_imgs.shape[-1]):
                    ori_img = ori_imgs[0, ..., i].cpu().numpy()
                    ori_img = ori_img[..., [2, 1, 0]]
                    ori_img = Image.fromarray(ori_img.astype(np.uint8))
                    ori_img.save(os.path.join(save_dir, f'{i_iter_val}_image_{i}.png'))
            
            # breakpoint()
            result_dict = my_model(imgs=input_imgs, metas=data)
            for idx, pred in enumerate(result_dict['final_occ']):
                pred_occ = pred
                gt_occ = result_dict['sampled_label'][idx]
                occ_shape = [200, 200, 16]
                if args.vis_gaussian_topdown and use_rerun:
                    log_gaussian_topdown(
                        f"samples/{i_iter_val}/gaussian_topdown",
                        result_dict['anchor_init'],
                        result_dict['gaussians'],
                    )
                if args.vis_occ and use_rerun:
                    log_occ(
                        f"samples/{i_iter_val}/occ/pred",
                        pred_occ.reshape(*occ_shape),
                        dataset=args.dataset,
                    )
                    log_occ(
                        f"samples/{i_iter_val}/occ/gt",
                        gt_occ.reshape(*occ_shape),
                        dataset=args.dataset,
                    )
                if args.vis_gaussian and use_rerun:
                    log_gaussian(
                        f"samples/{i_iter_val}/gaussian/final",
                        result_dict['gaussian'],
                        **draw_gaussian_params)

                if local_rank == 0 and args.vis_raycast:
                    pred_occ_np = tensor_to_numpy_occ(pred_occ, occ_shape)
                    gt_occ_np = tensor_to_numpy_occ(gt_occ, occ_shape)
                    pred_bev, _, _ = ray_cast_occ(
                        pred_occ_np,
                        dataset=args.dataset,
                        max_range=args.raycast_max_range,
                        step=args.raycast_step,
                        num_azimuth=args.raycast_num_azimuth,
                    )
                    gt_bev, _, _ = ray_cast_occ(
                        gt_occ_np,
                        dataset=args.dataset,
                        max_range=args.raycast_max_range,
                        step=args.raycast_step,
                        num_azimuth=args.raycast_num_azimuth,
                    )

                    pred_bev = cv2.resize(pred_bev, (800, 800), interpolation=cv2.INTER_NEAREST)
                    gt_bev = cv2.resize(gt_bev, (800, 800), interpolation=cv2.INTER_NEAREST)
                    panel = np.concatenate([pred_bev, gt_bev], axis=1)
                    cv2.putText(panel, "Pred", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(panel, "GT", (820, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
                    frame_stem = f"val_{i_iter_val}_idx_{idx}"
                    cv2.imwrite(os.path.join(save_dir, f"{frame_stem}_raycast.png"), panel[..., ::-1])
                    raycast_writer.write(panel[..., ::-1])

                if local_rank == 0 and args.vis_open3d_video:
                    pred_occ_np = tensor_to_numpy_occ(pred_occ, occ_shape)
                    gt_occ_np = tensor_to_numpy_occ(gt_occ, occ_shape)
                    frame_stem = f"val_{i_iter_val}_idx_{idx}"
                    save_open3d_occ_video(
                        pred_occ_np,
                        save_path=os.path.join(save_dir, f"{frame_stem}_pred_open3d.mp4"),
                        dataset=args.dataset,
                        width=args.open3d_width,
                        height=args.open3d_height,
                        fps=args.open3d_fps,
                        num_frames=args.open3d_num_frames,
                        point_size=args.open3d_point_size,
                    )
                    save_open3d_occ_video(
                        gt_occ_np,
                        save_path=os.path.join(save_dir, f"{frame_stem}_gt_open3d.mp4"),
                        dataset=args.dataset,
                        width=args.open3d_width,
                        height=args.open3d_height,
                        fps=args.open3d_fps,
                        num_frames=args.open3d_num_frames,
                        point_size=args.open3d_point_size,
                    )

                if local_rank == 0 and args.vis_occ_panel:
                    pred_occ_np = tensor_to_numpy_occ(pred_occ, occ_shape)
                    gt_occ_np = tensor_to_numpy_occ(gt_occ, occ_shape)
                    pred_img = render_occ_snapshot(
                        pred_occ_np,
                        dataset=args.dataset,
                        view_name=args.occ_camera_view,
                        width=args.panel_width // 2,
                        height=args.panel_mid_height,
                        point_radius=args.panel_point_radius,
                    )
                    gt_img = render_occ_snapshot(
                        gt_occ_np,
                        dataset=args.dataset,
                        view_name=args.occ_camera_view,
                        width=args.panel_width // 2,
                        height=args.panel_mid_height,
                        point_radius=args.panel_point_radius,
                    )

                    front_idx = max(0, min(args.front_cam_index, ori_imgs.shape[-1] - 1))
                    front = ori_imgs[idx, ..., front_idx].detach().cpu().numpy().astype(np.uint8)
                    # ori_img tensor is BGR in this codebase.
                    top = cv2.resize(front, (args.panel_width, args.panel_top_height), interpolation=cv2.INTER_LINEAR)

                    mid = np.concatenate([pred_img, gt_img], axis=1)
                    panel = np.concatenate([top, mid], axis=0)
                    cv2.putText(panel, "Front RGB", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(panel, "Pred Occupancy", (20, args.panel_top_height + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(panel, "GT Occupancy", (args.panel_width // 2 + 20, args.panel_top_height + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

                    frame_stem = f"frame_{i_iter_val:05d}_{idx:02d}"
                    cv2.imwrite(os.path.join(save_dir, f"{frame_stem}.png"), panel)
                    if occ_panel_writer is not None:
                        occ_panel_writer.write(panel)
                miou_metric._after_step(pred_occ, gt_occ)
            
            if i_iter_val % print_freq == 0 and local_rank == 0:
                logger.info('[EVAL] Iter %5d'%(i_iter_val))
                    
    miou, iou2 = miou_metric._after_epoch()
    logger.info(f'mIoU: {miou}, iou2: {iou2}')
    miou_metric.reset()
    if raycast_writer is not None:
        raycast_writer.release()
        logger.info('Saved raycast video to vis directory.')
    if occ_panel_writer is not None:
        occ_panel_writer.release()
        logger.info('Saved occupancy panel video to vis directory.')

    if use_rerun and args.rerun_mode == 'serve' and args.rerun_keep_alive:
        logger.info(
            f"Rerun server is kept alive at web port {args.rerun_web_port}. "
            "Press Ctrl+C to stop."
        )
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info('Stopping Rerun keep-alive loop.')
    
    if writer is not None:
        writer.close()
        

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vis-occ', action='store_true', default=False)
    parser.add_argument('--vis-gaussian', action='store_true', default=False)
    parser.add_argument('--vis_gaussian_topdown', action='store_true', default=False)
    parser.add_argument('--vis-index', type=int, nargs='+', default=[])
    parser.add_argument('--vis-start-index', type=int, default=0)
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--vis_scene_index', type=int, default=-1)
    parser.add_argument('--vis-scene', action='store_true', default=False)
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='nusc')
    parser.add_argument('--model-type', type=str, default="base", choices=["base", "prob"])
    parser.add_argument('--save-ori-images', action='store_true', default=False)
    parser.add_argument('--rerun-mode', type=str, default='connect', choices=['connect', 'serve', 'save'])
    parser.add_argument('--rerun-connect-addr', type=str, default='127.0.0.1:9876')
    parser.add_argument('--rerun-web-port', type=int, default=9090)
    parser.add_argument('--rerun-open-browser', action='store_true', default=False)
    parser.add_argument('--rerun-save-path', type=str, default=None)
    parser.add_argument('--rerun-app-id', type=str, default='GaussianFormer-Inference')
    parser.add_argument('--rerun-keep-alive', action='store_true', default=False)
    parser.add_argument('--vis-open3d-video', action='store_true', default=False)
    parser.add_argument('--open3d-fps', type=int, default=12)
    parser.add_argument('--open3d-num-frames', type=int, default=120)
    parser.add_argument('--open3d-width', type=int, default=1280)
    parser.add_argument('--open3d-height', type=int, default=720)
    parser.add_argument('--open3d-point-size', type=float, default=3.0)
    parser.add_argument('--vis-raycast', action='store_true', default=False)
    parser.add_argument('--raycast-video-fps', type=int, default=8)
    parser.add_argument('--raycast-max-range', type=float, default=80.0)
    parser.add_argument('--raycast-step', type=float, default=0.5)
    parser.add_argument('--raycast-num-azimuth', type=int, default=360)
    parser.add_argument('--vis-occ-panel', action='store_true', default=False)
    parser.add_argument('--occ-camera-view', type=str, default='bird45', choices=['top', 'front', 'bird45'])
    parser.add_argument('--front-cam-index', type=int, default=0)
    parser.add_argument('--panel-width', type=int, default=1600)
    parser.add_argument('--panel-top-height', type=int, default=450)
    parser.add_argument('--panel-mid-height', type=int, default=800)
    parser.add_argument('--panel-video-fps', type=int, default=8)
    parser.add_argument('--panel-point-radius', type=int, default=2)
    parser.add_argument('--vis-tag', type=str, default='')
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    if (args.vis_occ or args.vis_gaussian or args.vis_gaussian_topdown or args.vis_open3d_video or args.vis_raycast or args.vis_occ_panel) and args.gpus > 1:
        print('Visualization mode uses a single GPU process for stable Rerun streaming.')
        args.gpus = 1
    print(args)

    if args.gpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)
