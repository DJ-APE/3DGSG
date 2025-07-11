import cupy as cp
from tqdm import tqdm
import os
import argparse

def find_intersection_and_save(file1, file2, output_file, batch_size=1000):
    array1 = cp.load(file1)
    array2 = cp.load(file2)

    if array1.shape[1] != array2.shape[1]:
        raise ValueError("输入的两个数组必须具有相同的列数。")

    intersection_list = []
    num_batches = int(cp.ceil(array1.shape[0] / batch_size))

    for i in tqdm(range(num_batches), desc="Finding intersection with CUDA in batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, array1.shape[0])
        batch = array1[start_idx:end_idx]

        matches = cp.all(batch[:, None] == array2, axis=2)
        intersection_mask = cp.any(matches, axis=1)

        if cp.any(intersection_mask):
            intersection_list.append(batch[intersection_mask])

    intersection = cp.concatenate(intersection_list, axis=0) if intersection_list else cp.array([])
    cp.save(output_file, intersection)

    print(f"✅ 交集已保存到: {output_file}")
    return cp.asnumpy(intersection)

def find_npy_files(folder_path):
    return [os.path.join(folder_path, f)
            for f in os.listdir(folder_path) if f.endswith('.npy')]

def main():
    parser = argparse.ArgumentParser(description="使用 CuPy 查找两个 .npy 文件中的交集并保存")
    parser.add_argument('--folder', type=str, required=True, help="包含两个 .npy 文件的文件夹路径")
    parser.add_argument('--output', type=str, default="intersection.npy", help="输出文件名")
    parser.add_argument('--batch-size', type=int, default=2000, help="批处理大小")

    args = parser.parse_args()

    npy_files = find_npy_files(args.folder)
    if len(npy_files) != 2:
        raise ValueError(f"❌ 找到 {len(npy_files)} 个 .npy 文件，必须正好是两个")

    output_path = os.path.join(args.folder, args.output)

    find_intersection_and_save(npy_files[0], npy_files[1], output_path, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
