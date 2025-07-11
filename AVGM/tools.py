import os

def parse_results_folder(results_dir):
    jpg_files = {}
    png_matches = []

    for fname in os.listdir(results_dir):
        if fname.endswith(".jpg"):
            base = os.path.splitext(fname)[0]  # e.g. frame000
            jpg_files[base] = fname

    for fname in os.listdir(results_dir):
        if fname.endswith(".png") and "_" in fname:
            parts = fname.split("_")
            base = parts[0]  # e.g. frame000
            if base in jpg_files:
                jpg_name = jpg_files[base]
                full_png_path = os.path.join(results_dir, fname)
                print(f"PNG路径: {full_png_path}")
                print(f"匹配JPG文件名: {jpg_name}")
                print(f"匹配前缀（无扩展）: {base}")
                print("------")
                png_matches.append((full_png_path, jpg_name, base))

    return png_matches


# 示例用法（你可以换成你自己的路径）
if __name__ == "__main__":
    results_folder = "/media/djape/EXData/Grounding/data/results"  # 🔁 修改为你自己的路径
    png_matches = parse_results_folder(results_folder)

    for elmt in png_matches:
        print(elmt)