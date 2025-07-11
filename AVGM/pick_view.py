import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil
import argparse

# ========== 1. 解析 images.txt 文件 ==========
def parse_images_file(images_file_path, selected_image_names):
    images_data = {}
    with open(images_file_path, 'r') as f:
        for line in f:
            if line[0].isdigit():
                parts = line.split()
                image_name = parts[9]
                if image_name in selected_image_names:
                    image_id = int(parts[0])
                    qw, qx, qy, qz = map(float, parts[1:5])
                    tx, ty, tz = map(float, parts[5:8])
                    images_data[image_name] = {
                        'id': image_id,
                        'quaternion': np.array([qw, qx, qy, qz]),
                        'translation': np.array([tx, ty, tz])
                    }
    return images_data

def calculate_rotation_angle(q1, q2):
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot_product = np.dot(q1, q2)
    angle = 2 * np.arccos(min(1, abs(dot_product)))
    return np.degrees(angle)

def find_best_180_pair(images_data):
    image_names = list(images_data.keys())
    best_pair = (None, None)
    best_diff = float('inf')
    best_angle = 0
    for i in range(len(image_names)):
        for j in range(i+1, len(image_names)):
            q1 = images_data[image_names[i]]['quaternion']
            q2 = images_data[image_names[j]]['quaternion']
            angle = calculate_rotation_angle(q1, q2)
            diff = abs(angle - 180)
            if diff < best_diff:
                best_diff = diff
                best_angle = angle
                best_pair = (image_names[i], image_names[j])
    return best_pair, best_angle

# ========== 2. 解析文件夹中所有 mask 对应的原图名 ==========
def get_all_frame_jpg_names(folder_path):
    modified_filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.png') and filename.startswith('frame') and '_' in filename:
            new_name = filename.split('_')[0] + '.jpg'
            if new_name not in modified_filenames:
                modified_filenames.append(new_name)
    return modified_filenames

# ========== 3. 移动对应的 mask 图像 ==========
def move_masks_by_group(folder_path, group1, group2):
    group1_ids = [img.split('.')[0][5:] for img in group1]
    group2_ids = [img.split('.')[0][5:] for img in group2]
    group1_folder = os.path.join(folder_path, "view1s")
    group2_folder = os.path.join(folder_path, "view2s")
    os.makedirs(group1_folder, exist_ok=True)
    os.makedirs(group2_folder, exist_ok=True)

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".png") and file_name.startswith("frame") and '_' in file_name:
            file_base_name = file_name.split('_')[0]
            file_id = file_base_name[5:]
            source_path = os.path.join(folder_path, file_name)
            if file_id in group1_ids:
                shutil.copy2(source_path, os.path.join(group1_folder, file_name))
            elif file_id in group2_ids:
                shutil.copy2(source_path, os.path.join(group2_folder, file_name))

# ========== 4. 找出每组中最完整的图像 ==========
def calculate_completeness(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    non_black_pixels = np.sum(gray_image > 10)
    return non_black_pixels

def find_most_complete_images(folder_path, output_folder, top_x=1):
    completeness_scores = []
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    for image_file in tqdm(image_files, desc=f"[{os.path.basename(folder_path)}] 完整度计算中"):
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        if image is None:
            continue
        completeness = calculate_completeness(image)
        completeness_scores.append((image_file, completeness))
    completeness_scores.sort(key=lambda x: x[1], reverse=True)
    top_images = completeness_scores[:top_x]
    os.makedirs(output_folder, exist_ok=True)
    for image_file, _ in top_images:
        shutil.copyfile(os.path.join(folder_path, image_file), os.path.join(output_folder, image_file))

# ========== 5. 主程序入口 ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据 mask 分组并选出最完整图像")
    parser.add_argument('--folder-path', type=str, required=True, help="包含 frame_xxxx.png 和 xxxx.jpg 的主路径")
    parser.add_argument('--images-file', type=str, required=True, help="COLMAP 导出的 images.txt 文件路径")
    parser.add_argument('--top-x', type=int, default=1, help="每组选出最完整的前几张图")

    args = parser.parse_args()

    selected_image_names = get_all_frame_jpg_names(args.folder_path)
    images_data = parse_images_file(args.images_file, selected_image_names)

    pair, angle = find_best_180_pair(images_data)
    print("✅ 最佳视角图像对:", pair)
    print("📐 视角夹角（度）:", angle)

    group1, group2 = [pair[0]], [pair[1]]
    move_masks_by_group(args.folder_path, group1, group2)

    find_most_complete_images(os.path.join(args.folder_path, "view1s"), os.path.join(args.folder_path, "FindView1"), top_x=args.top_x)
    find_most_complete_images(os.path.join(args.folder_path, "view2s"), os.path.join(args.folder_path, "FindView2"), top_x=args.top_x)
