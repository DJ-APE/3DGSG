import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import time

def extract_object_from_mask(original_img_path, mask_img_path, save_path, target_size=(1600, 899)):
    original = cv2.imread(original_img_path)
    mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)

    if original is None or mask is None:
        print(f"❌ 图像读取失败: {original_img_path} 或 {mask_img_path}")
        return

    if original.shape[:2] != mask.shape[:2]:
        print(f"⚠️ 尺寸不一致: {original_img_path}")
        return

    _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"⚠️ mask中无目标区域: {mask_img_path}")
        return

    x, y, w, h = cv2.boundingRect(contours[0])
    for cnt in contours[1:]:
        x2, y2, w2, h2 = cv2.boundingRect(cnt)
        x = min(x, x2)
        y = min(y, y2)
        w = max(x + w, x2 + w2) - x
        h = max(y + h, y2 + h2) - y

    padding = 20
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = min(w + 2 * padding, original.shape[1] - x)
    h = min(h + 2 * padding, original.shape[0] - y)

    cropped_img = original[y:y + h, x:x + w]
    cropped_mask = binary_mask[y:y + h, x:x + w]

    b, g, r = cv2.split(cropped_img)
    alpha = cropped_mask
    rgba = cv2.merge((b, g, r, alpha))

    obj_h, obj_w = rgba.shape[:2]
    aspect_ratio = obj_w / obj_h
    target_w, target_h = target_size

    if target_w / target_h > aspect_ratio:
        new_h = target_h
        new_w = int(aspect_ratio * new_h)
    else:
        new_w = target_w
        new_h = int(new_w / aspect_ratio)

    resized_obj = cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    result_img = np.zeros((target_h, target_w, 4), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    result_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_obj

    if not save_path.lower().endswith(".png"):
        save_path = os.path.splitext(save_path)[0] + ".png"
    cv2.imwrite(save_path, result_img)


def watch_and_process(base_dir, target_size=(1600, 899), poll_interval=2):
    processed = set()
    image_extensions = ('.jpg', '.jpeg', '.png')

    # 获取所有 .jpg 文件作为基准数量
    jpg_images = sorted([f for f in os.listdir(base_dir) if f.lower().endswith('.jpg')])
    total_target = len(jpg_images)
    print(f"👀 正在监听: {base_dir}，预计处理 {total_target} 个 frame 文件夹。")

    processed_count = 0

    while True:
        folder_names = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and f.startswith("frame")]
        folder_names.sort()

        for folder in folder_names:
            folder_path = os.path.join(base_dir, folder)
            if folder in processed:
                continue

            original_img_path = os.path.join(base_dir, folder + ".jpg")
            if not os.path.exists(original_img_path):
                continue

            mask_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
            if not mask_files:
                continue

            print(f"📦 发现已生成：{folder}，开始处理...")
            output_dir = os.path.join(folder_path, 'new_resized')
            os.makedirs(output_dir, exist_ok=True)

            for mask_file in mask_files:
                mask_path = os.path.join(folder_path, mask_file)
                save_path = os.path.join(output_dir, mask_file)
                extract_object_from_mask(original_img_path, mask_path, save_path, target_size)

            processed.add(folder)
            processed_count += 1

            # ✅ 判断是否处理完所有 jpg
            if processed_count >= total_target:
                print("✅ 所有图像处理完成，程序即将退出。")
                return

        time.sleep(poll_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="监听render文件夹下frame子文件夹，自动提取目标图像（透明背景）")
    parser.add_argument('--base-dir', type=str, required=True, help="render主目录路径，例如 /path/to/render")
    parser.add_argument('--target-size', type=int, nargs=2, default=[1600, 899], help="目标图像尺寸 (W H)")
    parser.add_argument('--interval', type=int, default=2, help="轮询间隔秒数")

    args = parser.parse_args()
    watch_and_process(args.base_dir, tuple(args.target_size), args.interval)
