import os
import ast
import json
import numpy as np
import cv2

def remove_one_quote(word):
    if word.startswith('"'):
        word = word[1:]
    if word.endswith('"'):
        word = word[:-1]
    return word

def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    points = np.array(points, dtype=np.float32)
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.8:
        dst_img = np.rot90(dst_img)
    return dst_img
    
base_folder = input("Enter base folder path(image): ")
output_folder = base_folder + '_crops'

# system_file = input("Enter system_results.txt file path: ")
system_file = 'system_results/' + base_folder + '.txt'
with open(system_file, 'r') as f:
    lines = f.readlines()

base_dir = os.path.join('src_images', base_folder)
output_dir = os.path.join('rec_crops', output_folder)

os.makedirs(output_dir, exist_ok=True)
pred_path = base_folder + '_preds.txt'
txt_file_path = os.path.join('pred_labels', pred_path)
mark_file_path = os.path.join('marker', pred_path)

with open(txt_file_path, 'w') as f:
    with open(mark_file_path, 'w') as w:
        for line in lines:
            img_name, boxes = line.strip().split('\t')
            img_path = os.path.join(base_dir, img_name)
            img = cv2.imread(img_path)
            boxes = json.loads(boxes)
            for i,box in enumerate(boxes):
                points = box['points']
                transcript = box['transcription']
                label = remove_one_quote(transcript)
                crop_img = get_rotate_crop_image(img, points)
                suffix = f"{i}_{img_name}"
                output_path = os.path.join(output_dir, suffix)
                cv2.imwrite(output_path, crop_img)
                f.write(f"{suffix}\t{label}\n")
                w.write(f"{suffix}\t{label}\n")    
    
        
        