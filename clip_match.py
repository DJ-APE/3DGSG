import os
import torch
import clip
import csv
from PIL import Image
from tqdm import tqdm
import argparse

def find_new_folders(root_folder):
    new_folders = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if 'new_resized' in dirnames:
            new_folders.append(os.path.join(dirpath, 'new_resized'))
    return new_folders

def main(args):
    # 加载 CLIP 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    text_inputs = clip.tokenize([args.text]).to(device)

    # 遍历所有 new_resized 文件夹
    new_folders = find_new_folders(args.root_folder)
    print(f"共找到 {len(new_folders)} 个包含 new_resized 的文件夹")

    pairs = []

    for folder in tqdm(new_folders, desc="Processing folders"):
        similarities = []
        image_files = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

        for image_file in image_files:
            image_path = os.path.join(folder, image_file)
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text_inputs)

            similarity = (image_features @ text_features.T).item()
            similarities.append((image_file, similarity))

        top_k = sorted(similarities, key=lambda x: x[1], reverse=True)[:args.topk]

        for match in top_k:
            print(f"[{folder}] 最接近：{match[0]}，相似度 {match[1]:.4f}")
            pairs.append([match[0], match[1], folder])

    # 打印简写名
    abbreviation = ''.join([w[0].upper() for w in args.text.split()])
    print(f"\n📌 Text Abbreviation: {abbreviation}")

    # 保存为 CSV 文件
    csv_path = args.save_csv or f"{abbreviation}_clip_results.csv"
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Value', 'Path'])  # CSV 头
        for row in pairs:
            writer.writerow(row)

    print(f"\n✅ 匹配结果已保存至 CSV: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP Matching for Object Selection")
    parser.add_argument('--text', type=str, required=True, help="Target text description")
    parser.add_argument('--root-folder', type=str, required=True, help="Root folder containing new_resized folders")
    parser.add_argument('--topk', type=int, default=3, help="Number of top matches per folder (default: 3)")
    parser.add_argument('--save-csv', type=str, default=None, help="Optional: path to save the result CSV file")

    args = parser.parse_args()
    main(args)
