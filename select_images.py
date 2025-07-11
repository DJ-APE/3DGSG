import os
import shutil
import argparse

def select_images(input_folder, output_folder, num_selected):
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有图片文件名并排序
    all_images = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png'))])
    total_images = len(all_images)

    if total_images == 0:
        print("❌ 输入文件夹中没有图片。")
        return

    # 计算抽帧步长
    step = total_images // num_selected if total_images > num_selected else 1
    if total_images < num_selected:
        num_selected = total_images

    selected_indices = [i * step for i in range(num_selected)]
    selected_indices = [i for i in selected_indices if i < total_images]

    # 复制选中图像
    for idx in selected_indices:
        src_path = os.path.join(input_folder, all_images[idx])
        dst_path = os.path.join(output_folder, all_images[idx])
        shutil.copy(src_path, dst_path)

    print(f"✅ 已从 {total_images} 张图中间隔抽取 {len(selected_indices)} 张图，并保存至 {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="选择图像子集并复制到输出文件夹")
    parser.add_argument('--input_folder', type=str, required=True, help='原图像文件夹路径')
    parser.add_argument('--output_folder', type=str, required=True, help='输出图像文件夹路径')
    parser.add_argument('--num_selected', type=int, default=90, help='抽取图像数量')

    args = parser.parse_args()
    select_images(args.input_folder, args.output_folder, args.num_selected)
