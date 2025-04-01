Prompts for CLI:

To run surgikneevision.py use the following prompt in CLI:
python surgikneevision.py --yolo_weight models/YOLOv8n_custom.pt --sam_weight models/sam-vit-base_custom_box.pth --confidence_threshold 0.7 --conf 0.4 --iou 0.9 --show_3dbbox --show_sam --show_yolo

To run yolo_inference_az.py use the following prompt in CLI:
python yolo_inference_az.py --weights models/YOLOv8n_custom.pt --confidence_threshold 0.7 --bbox_color "red" --font_scale 0.5 --font_thickness 1

To run o3d_sam_inference_az.py use the following prompt in CLI:
python o3d_sam_inference_az.py --yolo_weight models/YOLOv8n_custom.pt --sam_weight models/sam-vit-base_custom_box.pth --show_mask --confidence_threshold 0.7 --conf 0.4 --iou 0.9

To run yolosam_inference_az.py use the following prompt in CLI:
python yolosam_inference_az.py  --yolo_weight models/YOLOv8n_custom.pt --sam_weight models/sam-vit-base_custom_box.pth --show_mask --confidence_threshold 0.7 --bbox_color "red" --font_scale 0.5 --font_thickness 1 --conf 0.4 --iou 0.9