python eval.py --py-config config/nuscenes_gs25600_solid.py --work-dir out/test/ --resume-from out/test/state_dict.pth

# Rerun 시각화 
CUDA_VISIBLE_DEVICES=0 python visualize.py \
  --py-config config/nuscenes_gs25600_solid.py \
  --work-dir out/nuscenes_gs25600_solid \
  --resume-from out/nuscenes_gs25600_solid/latest.pth \
  --vis-occ --vis-gaussian \
  --num-samples 3 \
  --model-type base \
  --rerun-mode serve \
  --rerun-connect-addr 127.0.0.1:9876 \
  --rerun-web-port 9090 \
  --rerun-keep-alive



# Desktop 에서 실행
rerun --connect rerun+http://127.0.0.1:9876/proxy


# Open3D 비디오 + Ray Casting 시각화 (파일 저장)
CUDA_VISIBLE_DEVICES=0 python visualize.py \
  --py-config config/nuscenes_gs25600_solid.py \
  --work-dir out/nuscenes_gs25600_solid \
  --resume-from out/nuscenes_gs25600_solid/latest.pth \
  --num-samples 10 \
  --model-type base \
  --vis-open3d-video \
  --open3d-num-frames 120 \
  --open3d-fps 12 \
  --vis-raycast \
  --raycast-video-fps 8 \
  --panel-point-radius 3

# 결과 저장 위치:
# out/nuscenes_gs25600_solid/vis_ep0/
#   - val_*_pred_open3d.mp4
#   - val_*_gt_open3d.mp4
#   - val_*_raycast.png
#   - raycast_pred_vs_gt.mp4


# 프레임 패널 시각화 (요구 레이아웃)
#   상단: Front RGB
#   중단 왼쪽: Pred Occupancy
#   중단 오른쪽: GT Occupancy
#   + 모든 프레임을 합쳐 최종 비디오 1개 생성
CUDA_VISIBLE_DEVICES=0 python visualize.py \
  --py-config config/nuscenes_gs25600_solid.py \
  --work-dir out/nuscenes_gs25600_solid \
  --resume-from out/nuscenes_gs25600_solid/latest.pth \
  --num-samples 300 \
  --model-type base \
  --vis-occ-panel \
  --occ-camera-view top \
  --front-cam-index 0 \
  --panel-width 1600 \
  --panel-top-height 450 \
  --panel-mid-height 800 \
  --panel-video-fps 5 \
  --panel-point-radius 3 \
  --vis-tag panel_top_s300

# 카메라 프리셋:
#   --occ-camera-view top
#   --occ-camera-view front
#   --occ-camera-view bird45

# 결과:
#   out/nuscenes_gs25600_solid/vis_ep0/frame_*.png
#   out/nuscenes_gs25600_solid/vis_ep0/occ_panel_video.mp4
