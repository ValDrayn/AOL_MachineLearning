import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import cv2
import os
import shutil
import argparse

from PIL import Image

# ROOT DIRECTORY
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

YOLOV5_VERSIONS = [
    "yolov5n.pt",
    "yolov5s.pt",
    "yolov5m.pt",
    "yolov5l.pt",
    "yolov5x.pt",
    "yolov5n6.pt",
    "yolov5s6.pt",
    "yolov5m6.pt",
    "yolov5l6.pt",
    "yolov5x6.pt"
]

YOLOV8_VERSIONS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
]

YOLOV10_VERSIONS = [
    "yolov10n.pt",
]

YOLOV11_VERSIONS = [
    "yolo11n.pt",
]

YOLOV12_VERSIONS = [
    "yolo12n.pt",
]


# Penyesuaian Annotation
def draw_bounding_boxess(image_path, annotation_path, output_path, **kwargs): # Argumen lama tidak relevan lagi

    image = cv2.imread(image_path)
    
    try:
        with open(annotation_path, 'r') as f:
            annotations = f.readlines()
    except FileNotFoundError:
        annotations = []

    filled_spots = 0
    empty_spots = 0
    
    for annotation in annotations:
        class_id, x_center, y_center, width, height = map(float, annotation.split())
        
        img_h, img_w, _ = image.shape
        left = int((x_center - width / 2) * img_w)
        top = int((y_center - height / 2) * img_h)
        right = int((x_center + width / 2) * img_w)
        bottom = int((y_center + height / 2) * img_h)

        color = (0, 0, 0)
        

        if int(class_id) == 0: # 0: empty
            color = (0, 255, 0)
            empty_spots += 1
        elif int(class_id) == 1: # 1: filled
            color = (0, 0, 255)
            filled_spots += 1
            
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)

    total_spots = empty_spots + filled_spots
    def draw_text_with_background(img, text, position, font, scale, text_color, bg_color, alpha=0.6, thickness=2, padding=10):
        overlay = img.copy()
        text_size, _ = cv2.getTextSize(text, font, scale, thickness)
        text_w, text_h = text_size

        x, y = position
        cv2.rectangle(
            overlay,
            (x - padding, y - text_h - padding),
            (x + text_w + padding, y + padding),
            bg_color,
            -1
        )
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


        cv2.putText(img, text, (x, y), font, scale, text_color, thickness)


    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    text_color = (0, 0, 0)
    bg_color = (200, 200, 200)  
    thickness = 2
    alpha = 0.6


    draw_text_with_background(image, f'Filled Spots: {filled_spots}', (30, 50), font, scale, text_color, bg_color, alpha, thickness)
    draw_text_with_background(image, f'Empty Spots: {empty_spots}', (30, 100), font, scale, text_color, bg_color, alpha, thickness)
    draw_text_with_background(image, f'Total Spots: {total_spots}', (30, 150), font, scale, text_color, bg_color, alpha, thickness)
    

    output_image_path = os.path.join(output_path, os.path.basename(image_path))
    cv2.imwrite(output_image_path, image)

    # Buat DataFrame untuk hasil gambar ini
    results = pd.DataFrame({
        'Image File': [os.path.basename(image_path)],
        'Empty Spots': [empty_spots],
        'Filled Spots': [filled_spots],
        'Total Spots Detected': [total_spots]
    })
        
    return results


def process_labels_baru(data_path, new_labels_folder):
    print(f"Menggunakan label langsung dari folder prediksi: {new_labels_folder}")
    
    if not os.path.exists(new_labels_folder):
        raise FileNotFoundError(f"Folder label prediksi tidak ditemukan di: {new_labels_folder}")
        
    return new_labels_folder


def process_imagess(data_path: str, output_folder: str, model: str = ''): 
    processed_images = 0

    images_folder = os.path.join(data_path, 'images/')

    if model != '':
        # If model is a path
        if "/" in model or "\\" in model:
            new_labels_folder = model
        else: # if model is a name
            new_labels_folder = os.path.join(ROOT, f'results/{model}/labels/')

        try: 
            print(f'Using labels from {new_labels_folder}')

            labels_folder = process_labels_baru(data_path, new_labels_folder)
        except FileNotFoundError as e:
            print(e)
            print(
                f'No labels found for model {model}!\n' +
                'Make sure you wrote the correct model name. Otherwise, train the model first.')
            return pd.DataFrame()
    
    else:
        labels_folder = os.path.join(data_path, 'labels/')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_images_folder = os.path.join(output_folder, 'images/')
    if not os.path.exists(output_images_folder):
        os.makedirs(output_images_folder)

    results_df = pd.DataFrame()

    print('Processing images...')
    for image_file in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image_file)
        
        annotation_file = os.path.splitext(image_file)[0] + '.txt'
        annotation_path = os.path.join(labels_folder, annotation_file)
        
        if os.path.exists(annotation_path):
            results = draw_bounding_boxess(image_path, annotation_path, output_images_folder)
            processed_images += 1
            results_df = pd.concat([results_df, results], ignore_index=True)
        else:
            print(f"No detections for {image_file}, skipping visualization.")

    print(f'Processed {processed_images} images ✅')
    results_df.to_csv(output_folder + 'output.csv', index=False)

    return results_df

def is_custom_model(model: str, yoloversion: str):
    if not model.endswith(".pt"):
        model = model + ".pt"

    versions = YOLOV11_VERSIONS if yoloversion == "11" else YOLOV8_VERSIONS

    if not model in versions:
        if model.__contains__("/") or model.__contains__("\\"):
            model_path = model 
            model = model.split("/")[-1]
            model = model.split("\\")[-1]
        else: 
            model_path = os.path.join(ROOT, f"models/{model}")

    else:
        model_path = model

    return model, model_path


from typing import List

def split_dataset(data_path: str, train_size: float = 0.8):
    # lets put all the train.txt and val.txt info into a list
    full_list = []
    train_list = []
    val_list = []
    
    # Get the names of the files in image folder
    for file in os.listdir(os.path.join(data_path, 'fold_0/images')):
        # Appends the path of the image to the list
        if file.endswith('.jpg'):
            full_list.append('./images/' + file + '\n')

    # Shuffle the list
    np.random.shuffle(full_list)

    # Split the list into train and val lists
    train_size = int(len(full_list) * train_size)
    train_list = full_list[:train_size]
    train_list_rotated = [x.replace('.jpg', '_rotated.jpg') for x in train_list]
    train_list_rotated2 = [x.replace('.jpg', '_rotated2.jpg') for x in train_list]

    val_list = full_list[train_size:]
    val_list_rotated = [x.replace('.jpg', '_rotated.jpg') for x in val_list]
    val_list_rotated2 = [x.replace('.jpg', '_rotated2.jpg') for x in val_list]

    # Write the train.txt file to fold 0
    with open(os.path.join(data_path, 'fold_0/train.txt'), 'w') as f:
        f.writelines(train_list)
        f.writelines(train_list_rotated)
        f.writelines(train_list_rotated2)

    # Write the val.txt file to fold 0
    with open(os.path.join(data_path, 'fold_0/val.txt'), 'w') as f:
        f.writelines(val_list)

    # Write the train.txt file to fold 1
    with open(os.path.join(data_path, 'fold_1/train.txt'), 'w') as f:
        f.writelines(val_list)
        f.writelines(val_list_rotated)
        f.writelines(val_list_rotated2)

    # Write the val.txt file to fold 1
    with open(os.path.join(data_path, 'fold_1/val.txt'), 'w') as f:
        f.writelines(train_list)

    print(f"Dataset split into train and val sets ✅")

def only_car_label(labels_path):
    # Loop over all labels
    for file in os.listdir(labels_path):
        if file.endswith('.txt'):
            with open(os.path.join(labels_path, file), 'r+') as f:
                # I want to delete all lines that do not start with 0
                lines = f.readlines()
                f.seek(0)  # Go back to the beginning of the file
                for line in lines:
                    if line.startswith('1'):
                        f.write(line)

                f.truncate()  # Remove extra lines, if any
                    
    print(f"Only car labels left ✅")


# Rotate images for data augmentation
def rotate_image_and_bboxes(image, bboxes, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate the image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    # Update the coordinates of the bounding boxes
    rotated_bboxes = []
    for bbox in bboxes:
        class_name, cx, cy, bbox_width, bbox_height = bbox

        # Convert to absolute coordinates
        x_min = int((cx - bbox_width / 2) * width)
        y_min = int((cy - bbox_height / 2) * height)
        x_max = int((cx + bbox_width / 2) * width)
        y_max = int((cy + bbox_height / 2) * height)

        # Rotate the coordinates
        rotated_bbox = cv2.transform(np.array([[[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]]]), rotation_matrix)[0]
        x_min_rot, y_min_rot = np.min(rotated_bbox, axis=0)
        x_max_rot, y_max_rot = np.max(rotated_bbox, axis=0)

        # Convert back to relative coordinates
        x_min_rot_rel = x_min_rot / width
        y_min_rot_rel = y_min_rot / height
        x_max_rot_rel = x_max_rot / width
        y_max_rot_rel = y_max_rot / height

        # Calculate the new center
        new_cx = (x_min_rot_rel + x_max_rot_rel) / 2
        new_cy = (y_min_rot_rel + y_max_rot_rel) / 2

        # Calculate the new width and height
        new_width = x_max_rot_rel - x_min_rot_rel
        new_height = y_max_rot_rel - y_min_rot_rel

        rotated_bboxes.append([class_name, new_cx, new_cy, new_width, new_height])

    return rotated_image, rotated_bboxes


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reporoot', type=str, default=ROOT, help='path to repo root')
    opt = parser.parse_args()
    return opt



def mean_df(df: pd.DataFrame):
    columns_to_mean = ['Precision', 'Recall', 'mAP0-50', 'mAP50-95']

    means = []
    for i in range(0, len(df), 2):
        avg = df.iloc[i:i+2][columns_to_mean].mean()
        means.append(avg)

    new_data = {
        'Model': ['YOLOv5n', 'YOLOv5s', 'YOLOv8n', 'YOLOv8s'],
        'Model Size (MB)': [df.iloc[0]['Model Size (MB)'], df.iloc[2]['Model Size (MB)'], df.iloc[4]['Model Size (MB)'], df.iloc[6]['Model Size (MB)']],
        'Parameters': [df.iloc[0]['Parameters'], df.iloc[2]['Parameters'], df.iloc[4]['Parameters'], df.iloc[6]['Parameters']],
        'Precision': [mean['Precision'] for mean in means],
        'Recall': [mean['Recall'] for mean in means],
        'mAP0-50': [mean['mAP0-50'] for mean in means],
        'mAP50-95': [mean['mAP50-95'] for mean in means]
    }

    return pd.DataFrame(new_data)