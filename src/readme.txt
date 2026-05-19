conda init bash && source /root/.bashrc

python /root/autodl-tmp/VTON/src/run_tryon_pipeline.py \
  --wonder3d-env wonder3d \
  --vton360-env vton360tmp \
  --collar-module neckline_edge \
  --neckline-edge-ymax-scale 0.60 \
  --neckline-edge-depth-bonus 0.45 \
  --neckline-edge-depth-penalty 0.02 \
  --neckline-edge-slope-strength 0.8 \
  --neckline-edge-slope-power 1.2 \
  --seam-module side_views \
  --seam-band-width 24

python /root/autodl-tmp/VTON/src/ui_server.py \
  --host 127.0.0.1 \
  --port 8766 \
  --wonder3d-env wonder3d \
  --vton360-env vton360tmp

ssh -L 8766:127.0.0.1:8766 -p 21092 root@connect.bjb1.seetacloud.com

python /root/autodl-tmp/VTON/src/mcp_word_server.py --transport stdio