import cupy as cp
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os

def parse_cameras_txt(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            width = int(parts[2])
            height = int(parts[3])
            fx, fy, cx, cy = map(float, parts[4:8])
            return fx, fy, cx, cy, width, height


def parse_images_txt(file_path):
    images_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            if len(parts) == 10:
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                image_name = parts[9]
                rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()
                translation_vector = np.array([tx, ty, tz])
                images_data[image_name] = {
                    'rotation_matrix': rotation_matrix,
                    'translation_vector': translation_vector
                }
    return images_data


def load_point_cloud(ply_path):
    point_cloud = o3d.io.read_point_cloud(ply_path)
    return np.asarray(point_cloud.points)


def project_point_cloud_to_image(points_3d, fx, fy, cx, cy, rotation_matrix, translation_vector):
    points_camera = (rotation_matrix @ points_3d.T).T + translation_vector
    u = fx * points_camera[:, 0] / points_camera[:, 2] + cx
    v = fy * points_camera[:, 1] / points_camera[:, 2] + cy
    return np.vstack((u, v)).T, points_camera[:, 2]


def find_closest_3d_points_in_image(pixels_2d, points_2d, points_3d, pixel_distance_threshold=100):
    pixels_2d_cp = cp.asarray(pixels_2d)
    points_2d_cp = cp.asarray(points_2d)
    points_3d_cp = cp.asarray(points_3d)
    closest_points_3d = cp.full((len(pixels_2d_cp), 3), cp.nan)

    for i, pixel in enumerate(tqdm(pixels_2d, desc="Processing pixels")):
        u, v = pixel
        pixel_distances = cp.linalg.norm(points_2d_cp - cp.array([u, v]), axis=1)
        closest_idx = cp.argmin(pixel_distances)
        if pixel_distances[closest_idx] < pixel_distance_threshold:
            closest_points_3d[i] = points_3d_cp[closest_idx]

    return cp.asnumpy(closest_points_3d)

from tools import parse_results_folder



def main():
    parser = argparse.ArgumentParser(description="将图像与 mask 投影提取为 3D 点")
    parser.add_argument('--cameras-path', type=str, required=True, help="cameras.txt 路径")
    parser.add_argument('--images-path', type=str, required=True, help="images.txt 路径")
    parser.add_argument('--point-cloud-path', type=str, required=True, help="点云 ply 文件路径")
    parser.add_argument('--results-dir', type=str, required=True, help="包含配对 mask/jpg 的文件夹")
    args = parser.parse_args()

    png_matches = parse_results_folder(args.results_dir)

    for elmt in png_matches:

        # ✅ 静态设置部分（可以手动改）
        mask_path = elmt[0]
        image_name = elmt[1]
        np_save_path = os.path.join(args.results_dir, elmt[2])

        fx, fy, cx, cy, width, height = parse_cameras_txt(args.cameras_path)
        images_data = parse_images_txt(args.images_path)
        points_3d = load_point_cloud(args.point_cloud_path)

        bw_image = Image.open(mask_path).convert("L")
        bw_array = np.array(bw_image)
        white_pixel_coords = np.column_stack(np.where(bw_array == 255))
        pixels_2d = white_pixel_coords[:, ::-1]

        image_info = images_data[image_name]
        R_mat = image_info['rotation_matrix']
        T_vec = image_info['translation_vector']

        points_2d, depths = project_point_cloud_to_image(points_3d, fx, fy, cx, cy, R_mat, T_vec)
        valid_mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < width) & \
                     (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height) & (depths > 0)
        points_2d_valid = points_2d[valid_mask]
        points_3d_valid = points_3d[valid_mask]

        closest_3d_points = find_closest_3d_points_in_image(
            pixels_2d, points_2d_valid, points_3d_valid, pixel_distance_threshold=300)

        np.save(np_save_path, closest_3d_points)
        print(f"✅ 已保存: {np_save_path}.npy")


if __name__ == "__main__":
    main()
