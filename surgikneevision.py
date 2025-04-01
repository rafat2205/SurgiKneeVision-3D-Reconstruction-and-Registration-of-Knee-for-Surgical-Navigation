import cv2
import numpy as np
import argparse
from FastSAM.fastsam import FastSAM, FastSAMPrompt
from ultralytics import YOLO
import open3d as o3d
import csv
from utils import *
from segment_anything import SamPredictor, sam_model_registry
import torch
import numpy as np
import math
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="SurgiKneeVision Object Detection and 3D Reconstruction")
    
    parser.add_argument("--yolo_weight", type=str, required=True,
                        help="Path to the YOLO weights file (e.g yolo_train/runs/detect/train/weights/best.pt)")
    parser.add_argument("--sam_weight", type=str, default='sam_vit_h_4b8939.pth',
                        help="Choose the SAM autodownloadable weight files ('sam_vit_h_4b8939.pth')")
    parser.add_argument("--show_yolo", action='store_true',
                        help="Show cv2 window with YOLO detection (default True)")
    parser.add_argument("--show_sam", action='store_true',
                        help="Show cv2 window with SAM detection (default True)")
    parser.add_argument("--show_mask", action='store_true',
                        help="Show a window with the estimated binary masks (default True)")
    parser.add_argument("--confidence_threshold", type=float, default=0.7,
                        help="Set the confidence threshold for YOLO detection (default: 0.7)")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Set the confidence threshold for the SAM model (default: 0.4)")
    parser.add_argument("--iou", type=float, default=0.9,
                        help="Set the IoU threshold for non-maximum suppression (default: 0.9)")
    parser.add_argument("--show_3dbbox", action='store_true',
                        help="Show in open3D window the 3D bounding box (default: True)")
    
    return parser.parse_args()

def SurgiKneeVision(args):
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

    yolo_model = YOLO(args.yolo_weight)
    
    sam_model_path = args.sam_weight
    sam = sam_model_registry["vit_b"](checkpoint=sam_model_path).to(device=torch.device('cuda:0'))
    sam_predictor = SamPredictor(sam)

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window('Point Cloud Viewer', 640, 480, visible=True)

    pcd = o3d.geometry.PointCloud()

    # Transformation matrix for flipping the point cloud upside down and left to right
    flip_matrix = np.array([[1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, -1, 0],
                            [0, 0, 0, 1]])
    
    if args.show_sam:
        cv2.namedWindow('SAM Inference', cv2.WINDOW_NORMAL)
    if args.show_mask:
        cv2.namedWindow('Annotation Mask', cv2.WINDOW_NORMAL)
    if args.show_yolo:
        cv2.namedWindow('YOLO Inference', cv2.WINDOW_NORMAL)

    try:
        # Main loop
        vis_geometry_added = False
        while True:
            # Capture an RGBD frame
            rgbd = sensor.capture_frame(align_depth_to_color)
            if rgbd is None:
                continue

            color_image = np.asanyarray(rgbd.color)
            depth_image = np.asanyarray(rgbd.depth)

            grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            grayscale_image = cv2.normalize(grayscale_image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            color_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)

            color_image_resized = cv2.resize(color_image, (640, 480))
            depth_image_resized = cv2.resize(depth_image, (640, 480))

            results = yolo_model.predict(source=color_image_resized, conf=0.50)
            predicted_boxes = results[0].boxes.xyxy.cpu().numpy()

            # Create a list to store bounding box lines
            bounding_box_lines = []

            if len(predicted_boxes) > 0:
                
                # Convert the PIL Image to a NumPy array
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
                    mask_array = mask[0].cpu().numpy()  # Move to CPU and convert to NumPy array
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
                        if args.show_sam:
                            cv2.imshow('SAM Inference', img_with_annotations)

                        # Display SAM results in another window
                        if args.show_mask:
                            cv2.imshow('Annotation Mask', colored_mask)

                        ann_mask_uint8 = np.array(final_mask).astype(np.uint8)

                        # Erode the annotation mask (to avoid reconstructing in 3D some background)
                        eroded_ann_mask = cv2.erode(ann_mask_uint8, kernel=np.ones((20, 20), np.uint8), iterations=1)
                        
                        isolated_depth = np.where((eroded_ann_mask > 0) & (depth_image_resized < 1000), depth_image_resized, np.nan)
                        non_nan_points = np.argwhere(~np.isnan(isolated_depth))
                        non_nan_depth_values = isolated_depth[non_nan_points[:, 0], non_nan_points[:, 1]]

                        depth_scale = 1

                        pcd.points = o3d.utility.Vector3dVector(
                            np.column_stack([non_nan_points[:, 1], non_nan_points[:, 0], non_nan_depth_values * depth_scale])
                        )

                        pcd_outlier = pcd.voxel_down_sample(voxel_size=2)

                        denoised_pcd, _ = pcd_outlier.remove_statistical_outlier(nb_neighbors=300,
                                                                                std_ratio=2.0)
                        
                        for box in predicted_boxes:
                            x_min, y_min, x_max, y_max = box

                            print(f"ROI: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")

                            # Extract the region of interest from the denoised point cloud
                            roi_points = np.asarray(denoised_pcd.points)
                            roi_points = roi_points[(roi_points[:, 0] >= x_min) & (roi_points[:, 0] <= x_max) &
                                                    (roi_points[:, 1] >= y_min) & (roi_points[:, 1] <= y_max)]

                            print(f"ROI Points Shape: {roi_points.shape}")
                            
                            ############################################################
                            if roi_points.size == 0:                                   #
                                print(f"Empty ROI points for bounding box: {box}")     #
                                continue                                               #
                            ############################################################

                            # Compute the bounding box dimensions based on the region of interest
                            bbox_lines = o3d.geometry.LineSet()
                            bbox_lines.points = o3d.utility.Vector3dVector([
                                [np.min(roi_points[:, 0]), np.min(roi_points[:, 1]), np.min(roi_points[:, 2])],
                                [np.max(roi_points[:, 0]), np.min(roi_points[:, 1]), np.min(roi_points[:, 2])],
                                [np.max(roi_points[:, 0]), np.min(roi_points[:, 1]), np.min(roi_points[:, 2])],
                                [np.max(roi_points[:, 0]), np.max(roi_points[:, 1]), np.min(roi_points[:, 2])],
                                [np.max(roi_points[:, 0]), np.max(roi_points[:, 1]), np.min(roi_points[:, 2])],
                                [np.min(roi_points[:, 0]), np.max(roi_points[:, 1]), np.min(roi_points[:, 2])],
                                [np.min(roi_points[:, 0]), np.max(roi_points[:, 1]), np.min(roi_points[:, 2])],
                                [np.min(roi_points[:, 0]), np.min(roi_points[:, 1]), np.min(roi_points[:, 2])],
                                [np.min(roi_points[:, 0]), np.min(roi_points[:, 1]), np.max(roi_points[:, 2])],
                                [np.max(roi_points[:, 0]), np.min(roi_points[:, 1]), np.max(roi_points[:, 2])],
                                [np.max(roi_points[:, 0]), np.min(roi_points[:, 1]), np.max(roi_points[:, 2])],
                                [np.max(roi_points[:, 0]), np.max(roi_points[:, 1]), np.max(roi_points[:, 2])],
                                [np.max(roi_points[:, 0]), np.max(roi_points[:, 1]), np.max(roi_points[:, 2])],
                                [np.min(roi_points[:, 0]), np.max(roi_points[:, 1]), np.max(roi_points[:, 2])],
                                [np.min(roi_points[:, 0]), np.max(roi_points[:, 1]), np.max(roi_points[:, 2])],
                                [np.min(roi_points[:, 0]), np.min(roi_points[:, 1]), np.max(roi_points[:, 2])],
                                [np.min(roi_points[:, 0]), np.min(roi_points[:, 1]), np.min(roi_points[:, 2])],
                                [np.min(roi_points[:, 0]), np.min(roi_points[:, 1]), np.max(roi_points[:, 2])],
                                [np.max(roi_points[:, 0]), np.min(roi_points[:, 1]), np.min(roi_points[:, 2])],
                                [np.max(roi_points[:, 0]), np.min(roi_points[:, 1]), np.max(roi_points[:, 2])],
                                [np.max(roi_points[:, 0]), np.max(roi_points[:, 1]), np.min(roi_points[:, 2])],
                                [np.max(roi_points[:, 0]), np.max(roi_points[:, 1]), np.max(roi_points[:, 2])],
                                [np.min(roi_points[:, 0]), np.max(roi_points[:, 1]), np.min(roi_points[:, 2])],
                                [np.min(roi_points[:, 0]), np.max(roi_points[:, 1]), np.max(roi_points[:, 2])],
                            ])
                            bbox_lines.lines = o3d.utility.Vector2iVector([
                                [0, 1], [1, 2], [2, 3], [3, 0],
                                [4, 5], [5, 6], [6, 7], [7, 4],
                                [0, 7], [1, 6], [2, 5], [3, 4],
                                [8, 9], [9, 10], [10, 11], [11, 8],
                                [12, 13], [13, 14], [14, 15], [15, 12],
                                [8, 15], [9, 14], [10, 13], [11, 12],
                                [16, 17], [17, 18], [18, 19], [19, 16],
                                [20, 21], [21, 22], [22, 23], [23, 20],
                                [16, 23], [17, 22], [18, 21], [19, 20]
                            ])
                            bbox_lines.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(bbox_lines.lines))])

                            bounding_box_lines.append(bbox_lines)
                        
                        denoised_pcd.transform(flip_matrix)
                        visualizer.clear_geometries()
                        visualizer.add_geometry(denoised_pcd)
                        visualizer.update_geometry(denoised_pcd)
                        visualizer.poll_events()
                        visualizer.update_renderer()

                        if args.show_3dbbox:
                            for bbox_lines in bounding_box_lines:
                                bbox_lines.transform(flip_matrix)
                                visualizer.add_geometry(bbox_lines)

                                center = np.mean(np.asarray(bbox_lines.points), axis=0)

                                # length of the coordinate axes
                                axis_length = 50

                                # Create coordinate frame mesh
                                coordinate_system = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length, origin=center)

                                visualizer.add_geometry(coordinate_system)

            # if YOLO window = true
            if args.show_yolo:
                for r in results:
                    boxes = r.boxes

                    for box in boxes:
                        if box.conf[0] >= args.conf:
                            # Bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                            confidence = math.ceil((box.conf[0] * 100)) / 100
                            cls = int(box.cls[0])
                            class_name = yolo_model.names[cls]

                            cv2.rectangle(color_image_resized, (x1, y1), (x2, y2), get_color("red"), 3)

                            # Display confidence and class name
                            org = (x1, y1 - 10)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.5
                            color = (255, 255, 255)
                            thickness = 1

                            cv2.putText(color_image_resized, f"{class_name}: {confidence}", org, font, font_scale, color, thickness)

                               
                    cv2.imshow('YOLO Inference', color_image_resized)

            # Check for key press to exit the loop (press 'q' to quit)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Saving point cloud and exiting...")
                o3d.io.write_point_cloud("./scan_knee.ply", denoised_pcd)
                break
                #sys.exit()

            # Add geometry to visualizer if not already added
            if not vis_geometry_added:
                visualizer.add_geometry(rgbd)
                vis_geometry_added = True

            # Update visualizer with new frame
            visualizer.update_geometry(rgbd)
            visualizer.poll_events()
            visualizer.update_renderer()

    finally:
        # Stop streaming
        visualizer.destroy_window()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
    SurgiKneeVision(args)