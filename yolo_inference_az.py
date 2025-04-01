import cv2
import numpy as np
import torch
import argparse
from FastSAM.fastsam import FastSAM, FastSAMPrompt
from ultralytics import YOLO
import open3d as o3d
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Inference with AzureKinect camera.')
    parser.add_argument('--weights', type=str, required=True, help='Path to YOLO weights file.')
    parser.add_argument('--confidence_threshold', type=float, default=0.7, help='Confidence threshold for YOLO inference (default: 0.7)')
    parser.add_argument('--bbox_color', type=str, default="red", help='Bounding box color (default: "red")')
    parser.add_argument('--font_scale', type=float, default=0.5, help='Font scale for displaying text (default: 0.5)')
    parser.add_argument('--font_thickness', type=int, default=1, help='Font thickness for displaying text (default: 1)')
    return parser.parse_args()

def main():

    args = parse_args()
    config = {
        "color_format": "BGRA32",
        "color_resolution": "720P",
        "depth_mode": "WFOV_2X2BINNED",
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
    yolo_model = YOLO(args.weights)

    DEVICE = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(DEVICE)
    cv2.namedWindow('YOLO Inference', cv2.WINDOW_AUTOSIZE)

    # Main loop
    try:
        while True:
            # Capture an bgra frame
            bgra = sensor.capture_frame(align_depth_to_color)
            if bgra is None:
                continue
            
            color_image = np.asanyarray(bgra.color)
            depth_image = np.asanyarray(bgra.depth)

            grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            grayscale_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
            
            color_image_resized = cv2.resize(grayscale_image, (640, 480))
            depth_image_resized = cv2.resize(depth_image, (640, 480))

            detections, _ = perform_yolo_inference(color_image_resized, yolo_model, confidence_threshold=args.confidence_threshold)

            for detection in detections:
                    x1, y1, x2, y2 = detection['bounding_box']
                    confidence = detection['confidence']
                    class_name = detection['class_name']

                    color = get_color(args.bbox_color)
                    cv2.rectangle(color_image_resized, (x1, y1), (x2, y2), color, 3)

                    org = (x1, y1 - 10)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(color_image_resized, f"{class_name}: {confidence}", org, font, args.font_scale, color, args.font_thickness)

            cv2.imshow('YOLO Inference', color_image_resized)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        print("Exiting...")


    finally:
        # Stop streaming
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()