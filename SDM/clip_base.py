import os
import torch
import clip
from PIL import Image
from tqdm import tqdm


def clip_base_find(text, output_path, delete_bottom_n=100):
    # 加载CLIP模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    text_inputs = clip.tokenize([text]).to(device)

    root_folder = output_path
    all_similarities = []

    image_files = [os.path.join(root_folder, f)
                   for f in os.listdir(root_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in tqdm(image_files, desc="Comparing images"):
        try:
            image = preprocess(Image.open(image_file)).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text_inputs)

            similarity = (image_features @ text_features.T).item()
            all_similarities.append((image_file, similarity))
        except Exception as e:
            print(f"⚠️ 图像处理失败: {image_file}，错误: {e}")

    # 排序并输出
    sorted_similarities = sorted(all_similarities, key=lambda x: x[1], reverse=True)
    print("\n与描述最接近的图像排序：")
    for image_file, similarity in sorted_similarities:
        print(f"图像: {image_file}，相似度: {similarity:.4f}")

    # 获取最低相似度的图像
    low_similarity_images = sorted(all_similarities, key=lambda x: x[1])[:delete_bottom_n]

    print(f"\n以下是相似度最低的 {delete_bottom_n} 张图像：")
    for image_file, similarity in low_similarity_images:
        print(f"图像: {image_file}，相似度: {similarity:.4f}")

    # 是否删除
    delete_confirmation = input(f"\n是否删除相似度最低的 {delete_bottom_n} 张图像？(y/n): ")
    if delete_confirmation.lower() == 'y':
        for image_file, _ in low_similarity_images:
            try:
                os.remove(image_file)
                print(f"✅ 已删除图像: {image_file}")
            except Exception as e:
                print(f"❌ 删除失败: {image_file}, 错误: {e}")
    else:
        print("未删除任何图像。")
