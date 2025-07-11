import argparse
import os
import pandas as pd

import find_from_mask
import clip_base
import extract_features

def main():
    parser = argparse.ArgumentParser(description="执行一整套从CSV中选择图像、匹配mask、再提取特征的流程")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV文件路径（包含Image, Value, Path）")
    parser.add_argument("--text", type=str, required=True, help="物体的描述文本")
    parser.add_argument("--output_root", type=str, required=True, help="保存目录的根目录（如 /media/.../data）")
    parser.add_argument("--render_path", type=str, required=True, help="render目录路径")
    parser.add_argument("--topn_mask", type=int, default=150, help="用于mask提取的Top-N数量")
    parser.add_argument("--topn_clip", type=int, default=100, help="用于clip过滤的Top-N数量")
    parser.add_argument("--topn_feature", type=int, default=10, help="用于最终特征提取的数量")
    args = parser.parse_args()

    # 检查 CSV 文件是否存在
    if not os.path.exists(args.csv_path):
        print(f"❌ 找不到 CSV 文件: {args.csv_path}")
        return

    # 读取 CSV 文件
    df = pd.read_csv(args.csv_path)
    data = df[["Image", "Value", "Path"]].values.tolist()

    # 调用 pipeline
    org_path = find_from_mask.find_base_mask(data, args.text, args.output_root, args.render_path, args.topn_mask)
    clip_base.clip_base_find(args.text, org_path, args.topn_clip)
    extract_features.feature_find(org_path, args.topn_feature)


if __name__ == "__main__":
    main()
