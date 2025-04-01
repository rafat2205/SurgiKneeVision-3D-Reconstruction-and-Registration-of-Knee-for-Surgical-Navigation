SurgiKneeVision is a project that uses YOLO as Object Detection model, SAM as Segmentation model and collects the depth map of the segmented part of the knee with Microsoft Azure Kinect for Total Knee Arthroplasty. This tries to register the collected depth map of the knee with conventional 
optimization based and modern deep learning based 3D registration models. I am unable to share the dataset publicly as it was obtained for research purposes under institutional access restrictions. The data was sourced using my institutional credentials and sharing it would violate the terms of use. The trained models are uolpaded [here](https://drive.google.com/drive/folders/1RYUjBZIHGsc9f3skCo8Rk0DL69AanxTJ?usp=sharing). Simply copy the "models" folder from the "Trained Models for SurgiKneeVision" folder inside the "SurgiKneeVision" directory to use the project. 

This work is inspired from [FusionVision](https://github.com/safouaneelg/FusionVision).

Prompts for CLI:

To run surgikneevision.py use the following prompt in CLI:
python surgikneevision.py --yolo_weight models/YOLOv8n_custom.pt --sam_weight models/sam-vit-base_custom_box.pth --confidence_threshold 0.7 --conf 0.4 --iou 0.9 --show_3dbbox --show_sam --show_yolo

To run yolo_inference_az.py use the following prompt in CLI:
python yolo_inference_az.py --weights models/YOLOv8n_custom.pt --confidence_threshold 0.7 --bbox_color "red" --font_scale 0.5 --font_thickness 1

To run o3d_sam_inference_az.py use the following prompt in CLI:
python o3d_sam_inference_az.py --yolo_weight models/YOLOv8n_custom.pt --sam_weight models/sam-vit-base_custom_box.pth --show_mask --confidence_threshold 0.7 --conf 0.4 --iou 0.9

To run yolosam_inference_az.py use the following prompt in CLI:
python yolosam_inference_az.py  --yolo_weight models/YOLOv8n_custom.pt --sam_weight models/sam-vit-base_custom_box.pth --show_mask --confidence_threshold 0.7 --bbox_color "red" --font_scale 0.5 --font_thickness 1 --conf 0.4 --iou 0.9
