import os
import shutil
import re

def extract_frame_and_mask(filename):
    """从 frame000_17.png 提取 (frame000.jpg, 17.png)"""
    match = re.match(r"(frame\d+)_([^.]+)\.png", filename)
    if match:
        frame_name = match.group(1) + ".jpg"
        mask_name = match.group(2) + ".png"
        return frame_name, mask_name
    return None, None

def process_findview(findview_dir, render_dir, output_dir, mapping_log):
    for fname in os.listdir(findview_dir):
        if not fname.endswith(".png") or "_" not in fname:
            continue

        frame_name, mask_name = extract_frame_and_mask(fname)
        if not frame_name or not mask_name:
            print(f"⚠️ 跳过无效文件名: {fname}")
            continue

        frame_id = os.path.splitext(frame_name)[0]
        original_image_path = os.path.join(render_dir, frame_name)
        mask_image_path = os.path.join(render_dir, frame_id, mask_name)

        if not os.path.exists(original_image_path):
            print(f"❌ 原图缺失: {original_image_path}")
            continue
        if not os.path.exists(mask_image_path):
            print(f"❌ mask缺失: {mask_image_path}")
            continue

        # 输出路径
        os.makedirs(output_dir, exist_ok=True)
        out_img = os.path.join(output_dir, frame_name)
        out_mask = os.path.join(output_dir, f"{frame_id}_{mask_name}")

        shutil.copy2(original_image_path, out_img)
        shutil.copy2(mask_image_path, out_mask)

        print(f"{frame_name} <--> {mask_name}")
        mapping_log.append((frame_name, mask_name))

def main(base_dir, render_dir):
    findview1 = os.path.join(base_dir, "FindView1")
    findview2 = os.path.join(base_dir, "FindView2")
    output_dir = os.path.abspath(os.path.join(base_dir, ".."))

    mapping_log = []
    print("📂 处理 FindView1")
    process_findview(findview1, render_dir, output_dir, mapping_log)

    print("📂 处理 FindView2")
    process_findview(findview2, render_dir, output_dir, mapping_log)

    print("\n✅ 完成！配对结果如下：")
    for frame, mask in mapping_log:
        print(f"{frame} <--> {mask}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="从 FindView 提取原图和最佳mask图像")
    parser.add_argument("--base-dir", type=str, required=True, help="包含 FindView1 和 FindView2 的路径")
    parser.add_argument("--render-dir", type=str, required=True, help="render 文件夹完整路径")
    args = parser.parse_args()
    main(args.base_dir, args.render_dir)
