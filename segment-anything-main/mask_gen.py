import os
import subprocess
from tqdm import tqdm
import argparse


def execute_command_with_progress(input_folder, output_folder, checkpoint_path, model_type, script_path):
    # 获取所有图片的路径
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # 遍历所有图片并执行命令
    for image_file in tqdm(image_files, desc="Processing images", unit="image"):
        # 去掉扩展名
        file_name_without_ext = os.path.splitext(image_file)[0]

        input_image_path = os.path.join(input_folder, image_file)

        # 构建命令
        command = [
            'python', script_path,
            '--checkpoint', checkpoint_path,
            '--model-type', model_type,
            '--input', input_image_path,
            '--output', output_folder
        ]

        print("Running:", ' '.join(command))
        subprocess.run(command)


def main():
    parser = argparse.ArgumentParser(description="批量运行图像处理脚本")
    parser.add_argument('--input-folder', type=str, required=True, help='输入图片所在文件夹路径')
    parser.add_argument('--output-folder', type=str, required=True, help='输出图像存放路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型 checkpoint 路径')
    parser.add_argument('--model-type', type=str, default='vit_h', help='模型类型（如 vit_h）')
    parser.add_argument('--script', type=str, required=True, help='要运行的 Python 脚本路径')

    args = parser.parse_args()

    execute_command_with_progress(
        args.input_folder,
        args.output_folder,
        args.checkpoint,
        args.model_type,
        args.script
    )


if __name__ == "__main__":
    main()
