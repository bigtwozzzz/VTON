conda init bash && source /root/.bashrc

python /root/autodl-tmp/VTON/src/run_tryon_pipeline.py \
  --wonder3d-env wonder3d \
  --vton360-env vton360tmp \
  --seam-module side_views \
  --seam-band-width 24

python /root/autodl-tmp/VTON/src/ui_server.py \
  --host 127.0.0.1 \
  --port 8766 \
  --wonder3d-env wonder3d \
  --vton360-env vton360tmp

ssh -L 8766:127.0.0.1:8766 -p 21092 root@connect.bjb1.seetacloud.com

python /root/autodl-tmp/VTON/src/mcp_word_server.py --transport stdio

本毕业设计围绕多视角虚拟换衣的实际应用需求，完成“VTON360 虚拟换衣系统的背面补全研究与实现”。系统以 VTON360 为多视角试穿核心，针对电商场景仅提供单张正面服装图、背面信息缺失的问题，引入 Wonder3D 生成服装多视角条件，并在 Wonder3D 与 VTON360 之间设计三类中间处理模块：前视 alpha 纠错与尺度对齐（降低浅色衣物袖口误抠和边缘白边）、manual_point 交互式领口去除（提升复杂领型稳定性）、接缝一致性修复（feather_stats/side_views，改善多视角旋转时侧缝颜色与纹理连续性）。系统实现了前后端交互、参数配置、进度与日志显示、结果可视化浏览，并可通过配置文件自动将处理后的服装正背面输入写入 VTON360 指定数据目录，完成端到端试穿推理与输出管理。实验表明：所提出的中间模块能在典型伪影（白边、领口遮挡、侧缝断裂）上实现稳定改善，耗时瓶颈主要集中在 VTON360 推理阶段（约占总运行时间的八成）。