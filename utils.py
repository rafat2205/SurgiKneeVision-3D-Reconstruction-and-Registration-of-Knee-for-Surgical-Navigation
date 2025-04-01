# utils.py

import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from ultralytics import YOLO
from FastSAM.fastsam import FastSAM, FastSAMPrompt
import torch

should_save_image = False
close_captured_image_window = False

def get_color(color_name):
    color_dict = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255),
        'purple': (128, 0, 128),
        'orange': (0, 165, 255),
        'cyan': (255, 255, 0),
        'magenta': (255, 0, 255),
        'pink': (203, 192, 255),
        'teal': (128, 128, 0),
        'lime': (0, 255, 0),
        'brown': (42, 42, 165),
        'maroon': (0, 0, 128),
        'navy': (128, 0, 0),
        'olive': (0, 128, 128),
        'gray': (128, 128, 128),
        'silver': (192, 192, 192),
        'gold': (0, 215, 255),
        'turquoise': (208, 224, 64),
        'violet': (211, 0, 148),
        'indigo': (130, 0, 75),
        'lavender': (208, 184, 170),
        'peach': (255, 218, 185),
        'salmon': (114, 128, 250),
        'sky_blue': (235, 206, 135),
        'tan': (140, 180, 210),
        'dark_green': (0, 100, 0),
        'dark_red': (0, 0, 139),
        'dark_blue': (139, 0, 0),
    }

    return color_dict.get(color_name, (0, 0, 255))  # Default to red if color_name is not found

def capture_image(image):
    cv2.namedWindow('Captured Image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Captured Image', image)
    cv2.setWindowTitle('RealSense', 'Press \'s\' to save')

    global should_save_image, close_captured_image_window

    while True:
        key = cv2.waitKey(25) & 0xFF
        if key == ord('s'):
            should_save_image = True
            close_captured_image_window = True
        elif key == 27:
            break

        if close_captured_image_window:
            if cv2.getWindowProperty('Captured Image', cv2.WND_PROP_VISIBLE) <= 0:
                close_captured_image_window = False
            else:
                cv2.destroyWindow('Captured Image')
                break

    if should_save_image:
        save_image(image)

def save_image(image):
    global should_save_image
    if should_save_image:
        if not os.path.exists('dataset'):
            os.makedirs('dataset')
        image_name = 'dataset/' + str(int(time.time() * 1000)) + '.png'
        cv2.imwrite(image_name, image)
        print('Image saved at:', image_name)
        should_save_image = False

def detect_and_visualize_yolo(input_data, yolo_model_path=None):
    if isinstance(input_data, str):  
        frame = cv2.imread(input_data)
    else:  
        frame = input_data

    if yolo_model_path is None:
        yolo_model_path = 'yolov8n.pt'
    model = YOLO(yolo_model_path)

    results = model.predict(source=frame, conf=0.50)
    predicted_boxes = results[0].boxes.xyxy.cpu().numpy()

    _, ax = plt.subplots(1)
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    predicted_boxes = results[0].boxes.xyxy.cpu().numpy()  # Convert to NumPy on CPU

    for box in predicted_boxes:
        x, y, w, h = box[:4]
        rect = patches.Rectangle((x, y), w - x, h - y, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.axis('off')
    plt.show()


def perform_yolo_inference(frame, model, confidence_threshold=0.6):
    results = model(frame, stream=True)
    detections = []
    predicted_boxes = []

    for r in results:
        boxes = r.boxes

        for box in boxes:
            if box.conf[0] >= confidence_threshold:
                # Bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                #x, y, w, h = box[:4]
                #rect = patches.Rectangle((x, y), w - x, h - y, linewidth=1, edgecolor='r', facecolor='none')

                # Confidence and class name
                confidence = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = model.names[cls]

                detections.append({
                    'bounding_box': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'class_name': class_name
                })
                predicted_boxes.append([x1, y1, x2, y2])

    return detections, predicted_boxes

