import os
import torch
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm
import torch.nn.functional as F

# 加载预训练 ResNet 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet50(pretrained=True).to(device)
model.eval()

# 图像预处理流程
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 提取单张图像的特征
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image)
    return features

# 计算两个特征向量之间的余弦相似度
def cosine_similarity(features1, features2):
    return F.cosine_similarity(features1, features2)

# 主函数：从文件夹中找出离群图像并删除
def feature_find(output_path, num_outliers_to_delete=30):
    print(f"📂 分析路径: {output_path}")
    print(f"🧹 预删除离群图像数量: {num_outliers_to_delete}")

    image_files = [
        os.path.join(output_path, f)
        for f in os.listdir(output_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    # 提取特征
    image_features = {}
    for image_file in tqdm(image_files, desc="Extracting features"):
        try:
            image_features[image_file] = extract_features(image_file)
        except Exception as e:
            print(f"⚠️ 图像处理失败: {image_file}, 错误: {e}")

    # 计算平均相似度
    average_similarities = []
    for img1 in tqdm(image_files, desc="Computing similarities"):
        total_similarity = 0
        for img2 in image_files:
            if img1 != img2:
                sim = cosine_similarity(image_features[img1], image_features[img2])
                total_similarity += sim.item()
        average_similarity = total_similarity / (len(image_files) - 1)
        average_similarities.append((img1, average_similarity))

    # 选出最不相似的图像
    sorted_similarities = sorted(average_similarities, key=lambda x: x[1])

    print(f"\n🧾 相似度最低前 {num_outliers_to_delete} 张图像：")
    for i in range(min(num_outliers_to_delete, len(sorted_similarities))):
        img_path, sim_value = sorted_similarities[i]
        print(f"{img_path} - 平均相似度: {sim_value:.4f}")
        try:
            os.remove(img_path)
            print(f"✅ 已删除: {img_path}")
        except Exception as e:
            print(f"❌ 删除失败: {img_path}, 错误: {e}")
