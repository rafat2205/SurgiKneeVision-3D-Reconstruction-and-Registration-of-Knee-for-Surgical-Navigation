import cv2
import numpy as np
import torch
import argparse
from FastSAM.fastsam import FastSAM, FastSAMPrompt
from ultralytics import YOLO
import open3d as o3d

from utils import *

from segment_anything import SamPredictor, sam_model_registry
import torch
import numpy as np
import math

def parse_args():
    parser = argparse.ArgumentParser(description='Object Detection with YOLO and SAM')
    parser.add_argument('--yolo_weight', type=str, required=True, help='Path to YOLO weights file')
    parser.add_argument('--sam_weight', type=str, required=True, help='Path to SAM weights file (e.g., sam_vit_h_4b8939.pth)')
    parser.add_argument('--confidence_threshold', type=float, default=0.7, help='Confidence threshold for YOLO detection (default: 0.7)')
    parser.add_argument('--bbox_color', type=str, default="red", help='Bounding box color (default: "red")')
    parser.add_argument('--font_scale', type=float, default=0.5, help='Font scale for displaying text (default: 0.5)')
    parser.add_argument('--font_thickness', type=int, default=1, help='Font thickness for displaying text (default: 1)')
    parser.add_argument('--conf', type=float, default=0.4, help='Confidence threshold for the SAM model (default: 0.4)')
    parser.add_argument('--iou', type=float, default=0.9, help='IoU threshold for non-maximum suppression (default: 0.9)')
    parser.add_argument('--show_mask', action='store_true', help='Show resulting binary mask')
    return parser.parse_args()

def main():
    args = parse_args()

    # Azure Kinect configuration
    config = {
        "color_format": "BGRA32",
        "color_resolution": "720P",
        "depth_mode": "NFOV_UNBINNED",
        "camera_fps": "30",
        "synchronized_images_only": "true",
        "depth_delay_off_color_usec": "0",
        "wired_sync_mode": "Standalone",
        "subordinate_delay_off_master_usec": "0",
        "disable_streaming_indicator": "false"
    }

    # Initialize configuration and sensor
    device = 0  # Device index
    align_depth_to_color = True  # Align depth to color frames

    # Load Azure Kinect sensor configuration
    config = o3d.io.AzureKinectSensorConfig(config)
    sensor = o3d.io.AzureKinectSensor(config)

    # Attempt to connect to the sensor
    if not sensor.connect(device):
        raise RuntimeError("Failed to connect to sensor")

    # Load YOLOv8 model
    yolo_model = YOLO(args.yolo_weight)

    # Load FastSAM model
    sam_model_path = args.sam_weight
    sam = sam_model_registry["vit_b"](checkpoint=sam_model_path).to(device=torch.device('cuda:0'))
    sam_predictor = SamPredictor(sam)

    DEVICE = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(DEVICE)

    cv2.namedWindow('YOLO Inference', cv2.WINDOW_NORMAL)
    cv2.namedWindow('SAM Inference', cv2.WINDOW_NORMAL)

    if args.show_mask:
        cv2.namedWindow('Annotation Mask', cv2.WINDOW_NORMAL)

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            bgra = sensor.capture_frame(align_depth_to_color)
            if bgra is None:
                continue

            color_image = np.asanyarray(bgra.color)
            depth_image = np.asanyarray(bgra.depth)

            grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            grayscale_image = cv2.normalize(grayscale_image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            color_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)

            color_image_resized = cv2.resize(color_image, (640, 480))
            depth_image_resized = cv2.resize(depth_image, (640, 480))


            # Perform YOLO inference using the defined function
            results = yolo_model.predict(source=color_image_resized, conf=0.50)
            predicted_boxes = results[0].boxes.xyxy.cpu().numpy()

            if len(predicted_boxes) > 0:
                
                image_array = np.array(color_image_resized)
                torch_boxes = torch.from_numpy(predicted_boxes)
                transformed_boxes = sam_predictor.transform.apply_boxes_torch(torch_boxes, image_array.shape[:2])

                transformed_boxes = transformed_boxes.to(device='cuda:0')

                # run SAM model on all the boxes
                sam_predictor.set_image(image_array)
                masks, scores, logits = sam_predictor.predict_torch(
                boxes=transformed_boxes,
                multimask_output=False,
                point_coords=None,
                point_labels=None
                )

                final_mask = None
                for mask in masks:
                    
                    #mask_array = mask[0].cpu().numpy()  # Move to CPU and convert to NumPy array
                    mask_array = mask[0].cpu().numpy() * 255
                    if final_mask is None:
                        final_mask = np.bitwise_or(mask_array, 0)
                    else:
                        final_mask = np.bitwise_or(final_mask, mask_array)

                    if final_mask.any():
                        overlay_image = image_array.copy()

                        # Create a color overlay for the mask
                        colored_mask = np.zeros_like(overlay_image)
                        colored_mask[final_mask > 0] = [0, 255, 0]  # Green color for mask

                        # Blend the overlay with the original image
                        alpha = 0.5  # Transparency factor
                        img_with_annotations = cv2.addWeighted(overlay_image, 1 - alpha, colored_mask, alpha, 0)
                        cv2.imshow('SAM Inference', img_with_annotations)

                        # Display SAM results in another window
                        if args.show_mask:
                            cv2.imshow('Annotation Mask', colored_mask)

                    else:
                        # If no SAM results, show the original frame in the SAM window
                        cv2.imshow('SAM Inference', color_image_resized)
            
            confidence_threshold=0.6
            for r in results:
                boxes = r.boxes

                for box in boxes:
                    if box.conf[0] >= confidence_threshold:
                        # Bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        confidence = math.ceil((box.conf[0] * 100)) / 100
                        cls = int(box.cls[0])
                        class_name = yolo_model.names[cls]

                        cv2.rectangle(color_image_resized, (x1, y1), (x2, y2), get_color(args.bbox_color), 3)

                        # Display confidence and class name
                        org = (x1, y1 - 10)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = args.font_scale
                        color = (255, 255, 255)
                        thickness = args.font_thickness

                        cv2.putText(color_image_resized, f"{class_name}: {confidence}", org, font, font_scale, color, thickness)

            cv2.imshow('YOLO Inference', color_image_resized)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
