#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr


# 这是第一个版本，用于标定电脑机箱上的东西。
# find6用于根据mask找点
# jiao、bing用来优化点
# 变白用于手动生成mask
# 变白变黑用于重建的影像的mask


from PIL import Image
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
import torchvision
from torchvision import utils

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


import cv2
import numpy as np
def extract_non_masked_region(original_image_path, mask_image_path, output_image_path):
    """
    根据黑底白色mask，提取原图中非白色区域的内容。

    :param original_image_path: 原图路径
    :param mask_image_path: mask图片路径（黑底白色区域）
    :param output_image_path: 输出的结果图片路径
    """
    # 读取原图和mask图
    original_image = cv2.imread(original_image_path)
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)  # 灰度模式读取mask图

    # 检查原图和mask图的尺寸是否匹配
    if original_image.shape[:2] != mask_image.shape[:2]:
        raise ValueError("原图和mask图的尺寸不一致，请确保它们具有相同的宽度和高度")

    # 创建一个反向的mask图，获取黑色区域（即mask中非白色区域）
    _, inverted_mask = cv2.threshold(mask_image, 128, 255, cv2.THRESH_BINARY_INV)

    # 使用反向mask提取原图中的对应部分
    result_image = cv2.bitwise_and(original_image, original_image, mask=inverted_mask)

    # 保存结果图片
    cv2.imwrite(output_image_path, result_image)
    print(f"提取的非白色区域已保存到: {output_image_path}")



def maximize_object_display(org_path, mask_path, output_path, x=1.5):
    """
    根据黑底白色mask，从原图中最大化显示物体，并根据mask边界范围自动选择裁剪区域。
    边界区域根据x参数动态扩展，x越大，裁剪的范围越大。

    :param org_path: 原图路径。
    :param mask_path: 黑底白色的mask图路径。
    :param output_path: 输出的图像路径。
    :param x: 控制从mask边界向外扩展的范围因子，x越大，裁剪范围越大。
    """
    # 读取原图和mask图
    original_image = cv2.imread(org_path)  # 读取原图
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 读取mask图，灰度模式

    # 确保两张图片大小相同
    if original_image.shape[:2] != mask_image.shape[:2]:
        raise ValueError("原图和mask图的尺寸不一致，请确保它们具有相同的宽度和高度")

    # 将mask图转换为二值图，白色区域设为255，黑色区域为0
    _, binary_mask = cv2.threshold(mask_image, 128, 255, cv2.THRESH_BINARY)

    # 找到mask中白色区域的边界框
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 获取包含所有白色区域的边界框
    if len(contours) == 0:
        raise ValueError("未找到白色区域，mask可能是空的")

    x_min, y_min, w, h = cv2.boundingRect(contours[0])

    # 提取所有的白色区域边界
    for cnt in contours:
        x_temp, y_temp, w_temp, h_temp = cv2.boundingRect(cnt)
        x_min = min(x_min, x_temp)
        y_min = min(y_min, y_temp)
        w = max(x_min + w, x_temp + w_temp) - x_min
        h = max(y_min + h, y_temp + h_temp) - y_min

    # 使用x参数扩展裁剪范围，确保裁剪区域向外扩展
    padding_w = int(w * (x - 1))  # 根据x倍扩展宽度
    padding_h = int(h * (x - 1))  # 根据x倍扩展高度

    x_min = max(0, x_min - padding_w)  # 向左扩展
    y_min = max(0, y_min - padding_h)  # 向上扩展
    w = min(w + 2 * padding_w, original_image.shape[1] - x_min)  # 宽度扩展
    h = min(h + 2 * padding_h, original_image.shape[0] - y_min)  # 高度扩展

    # 裁剪出包含白色区域和边界扩展部分的图像
    cropped_object = original_image[y_min:y_min + h, x_min:x_min + w]

    # 获取目标尺寸，自动根据扩展后的宽高动态调整
    target_width = int(w * x)  # 根据裁剪区域的宽度调整目标宽度
    target_height = int(h * x)  # 根据裁剪区域的高度调整目标高度

    # 计算裁剪部分的宽高比
    cropped_height, cropped_width = cropped_object.shape[:2]
    aspect_ratio = cropped_width / cropped_height

    # 根据裁剪部分的宽高比，计算缩放后的新尺寸，保持比例最大化填充目标尺寸
    if (target_width / target_height) > aspect_ratio:
        # 如果目标的宽高比大于裁剪部分，则按高度进行适应
        new_height = target_height
        new_width = int(aspect_ratio * target_height)
    else:
        # 如果目标的宽高比小于裁剪部分，则按宽度进行适应
        new_width = target_width
        new_height = int(target_width / aspect_ratio)

    # 缩放裁剪部分到新的大小
    resized_object = cv2.resize(cropped_object, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # 创建一个黑色背景的目标大小的图像
    result_full_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # 计算在目标图像中的位置，以使物体居中
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # 将缩放后的图像放到目标大小的黑色背景中，使其居中
    result_full_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_object

    # 保存最终的结果
    cv2.imwrite(output_path, result_full_image)
    print(f"最大化显示的图像已保存为: {output_path}")


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, predicted_threshold, input_image_path, output_image_path,
             torch_path, npy_path, save_img_path, img_name):

    if predicted_threshold < 0 :
        print("error < 0")
        # 读取图像
        #input_image_path = '/home/djape/gust/3DGS/gaussian-splatting-main/data3/find/botal/DSCF6054.JPG'
        img = Image.open(input_image_path)
        # 保存图像为新的文件
        #output_image_path = '/home/djape/gust/3DGS/gaussian-splatting-main/data3/find/botal/DSCF6054pre2.jpg'
        img.save(output_image_path)
        print(f"Image saved as {output_image_path}")

    else:
        first_iter = 0
        tb_writer = prepare_output_and_logger(dataset)
        print("dataset", dataset)
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians)
        gaussians.training_setup(opt)

        (model_params, first_iter) = torch.load(torch_path)
        gaussians.restore(model_params, opt)

        # gaussians._features_dc = torch.zeros_like(gaussians._features_dc)

        # 假设你的 PyTorch 张量是 size 为 (9722, 3) 的 3D 点
        points = gaussians._xyz  # 用随机数据举例
        print("高斯点的规模", points.shape)

        # 假设你有一个 (1521, 3) 大小的 numpy 数组，表示 3D 点
        target_points_np = np.load(npy_path)

        # 将 numpy 数组转换为 PyTorch 张量
        target_points = torch.tensor(target_points_np)
        print("初步点的规模", target_points.shape)

        # 选择设备，cuda:0 或 cpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 将张量移动到选定的设备
        points = points.to(device)
        target_points = target_points.to(device)

        # 设置一个阈值，比如 0.1，来定义 "接近" 的含义（根据具体需求调整）
        threshold = predicted_threshold

        # 计算每个 3D 点与目标点的距离，并用 tqdm 显示进度条
        close_points = []
        indices = []

        # 使用 tqdm 进行循环并显示进度
        for i in tqdm(range(points.size(0)), desc="Processing points"):
            point = points[i]

            # 计算该点与 target_points 中所有点的欧氏距离
            distances = torch.norm(target_points - point, dim=1)

            # 找到距离小于阈值的点
            close_mask = distances < threshold

            if close_mask.any():
                # 如果有接近的点，保存这些点和索引
                close_points.append(point)
                indices.append(i)


        # print("close_points", close_points)
        # print("indices", indices)

        # 将结果转换为张量形式
        close_points = torch.stack(close_points)
        indices = torch.tensor(indices)

        # 打印接近的点和索引
        print(f"Close points: {close_points}")
        print(f"Indices of close points: {indices}")
        print(indices.shape)

        # torch.save(indices, '/home/djape/gust/3DGS/gaussian-splatting-main/mytest/clip/names3/finally106/A black keyboard.pt')

        # # 使用 torch.no_grad() 禁用梯度追踪
        # # 选择合适的模式
        # with torch.no_grad():
        #
        #     # 其它全白
        #     for i in tqdm(range(points.size(0)), desc="Points to white"):
        #         if i not in indices:
        #             gaussians._features_dc[i].zero_()  # 将张量的值全部设为 0
        #             # gaussians._opacity[i].zero_()
        #             gaussians._features_dc[i, :, :] = 2
        #
        #     # 自己白色
        #     # for i in indices:
        #     #     gaussians._features_dc[i].zero_()  # 将张量的值全部设为 0
        #     #     #gaussians._opacity[i].zero_()
        #     #     gaussians._features_dc[i, :, :]  = 0.1

        # 获取所有点，并删除
        points = gaussians._xyz
        points = points.to(device)

        # 使用布尔掩码，初始化为 False（表示默认删除所有点）
        mask = torch.zeros(points.size(0), dtype=torch.bool)

        # 将需要保留的索引位置设置为 True
        mask[indices] = True  # 需要保留的索引位置为 True

        # 通过掩码选择保留的点
        gaussians._xyz = gaussians._xyz[mask]
        gaussians._features_dc = gaussians._features_dc[mask]
        gaussians._features_rest = gaussians._features_rest[mask]
        gaussians._scaling = gaussians._scaling[mask]
        gaussians._rotation = gaussians._rotation[mask]
        gaussians._opacity = gaussians._opacity[mask]

        # 生成图片使用
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        viewpoint_stack = scene.getTrainCameras().copy()
        for cam in viewpoint_stack:
            if cam.image_name == img_name:
                render_pkg = render(cam, gaussians, pipe, background)
                print("1", cam.image_name)
                image = render_pkg["render"].detach()
                #custom_path = "/home/djape/gust/3DGS/gaussian-splatting-main/data3/find/botal"
                torchvision.utils.save_image(image, f"{save_img_path}/{cam.image_name}pre.jpg")

                print(image.shape, "image.shape")

                return image


        # iteration = 1062
        #
        # print("\n[ITER {}] Saving Gaussians".format(iteration))
        # scene.save(iteration)
        #



def binary_cross_entropy_loss(pred_image_tensor, true_image_tensor):
    # 使用二值交叉熵来衡量两个图像之间的差异
    return F.binary_cross_entropy(pred_image_tensor, true_image_tensor)



def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()



import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from chamferdist import ChamferDistance
import torch.nn.functional as F
import numpy as np


def compute_non_black_loss(imageA_tensor, imageB_tensor, loss_type='mse'):
    """
    计算两张图片中非黑色区域的损失。

    :param imageA_tensor: 原图像，作为基准图像 (C, H, W)
    :param imageB_tensor: 预测图像，损失根据该图像计算 (C, H, W)
    :param loss_type: 损失类型，'mse' 代表均方误差，'l1' 代表绝对差
    :return: 非黑色区域的损失值
    """
    # 确保两张图片的尺寸一致
    if imageA_tensor.size() != imageB_tensor.size():
        raise ValueError("输入的两张图片大小必须相同")

    # 创建一个非黑色区域的掩码 (黑色是 [0, 0, 0])
    non_black_mask = torch.any(imageA_tensor != 0, dim=0)

    # 获取非黑色区域的像素值
    non_black_pixelsA = imageA_tensor[:, non_black_mask]
    non_black_pixelsB = imageB_tensor[:, non_black_mask]

    # 根据选择的损失类型计算损失
    if loss_type == 'mse':
        # 计算均方误差 (MSE)
        loss = F.mse_loss(non_black_pixelsA, non_black_pixelsB)
    elif loss_type == 'l1':
        # 计算绝对差 (L1 损失)
        loss = F.l1_loss(non_black_pixelsA, non_black_pixelsB)
    else:
        raise ValueError("不支持的损失类型，支持 'mse' 或 'l1'")

    return loss


# PointNet模型，提取特征并预测阈值
class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)  # 输出一个自适应阈值

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(2, 1)  # 输入点云转换为 (batch_size, 3, num_points)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # 直接展平 (batch_size, 1024)
        x = x.max(dim=2)[0]  # 使用 max pooling 取最大值作为全局特征，去掉最后一维

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        threshold = self.fc3(x)  # 阈值预测

        return threshold



# Chamfer距离的损失函数
def chamfer_loss(pointcloud_pred, pointcloud_true):
    chamfer_dist = ChamferDistance()
    loss = chamfer_dist(pointcloud_pred, pointcloud_true)
    return loss


# 图像渲染函数 (简化，假设你已经有方法将点云转换为2D图像)
def render_pointcloud_to_image(pointcloud):
    # 在这里插入你的渲染代码，返回图像张量
    # 假设渲染的结果是 (batch_size, 3, height, width) 的图像张量
    rendered_image = torch.rand(pointcloud.size(0), 3, 224, 224)  # 随机模拟的图像
    return rendered_image


# 计算图像的感知损失 (Perceptual Loss)
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential(*vgg[:4])
        self.slice2 = nn.Sequential(*vgg[4:9])
        self.slice3 = nn.Sequential(*vgg[9:16])
        self.slice4 = nn.Sequential(*vgg[16:23])

        for param in self.parameters():
            param.requires_grad = False  # VGG权重不需要训练

    def forward(self, img1, img2):
        h = self.slice1(img1)
        h2 = self.slice1(img2)
        h_loss = F.mse_loss(h, h2)

        h = self.slice2(h)
        h2 = self.slice2(img2)
        h_loss += F.mse_loss(h, h2)

        h = self.slice3(h)
        h2 = self.slice3(img2)
        h_loss += F.mse_loss(h, h2)

        h = self.slice4(h)
        h2 = self.slice4(img2)
        h_loss += F.mse_loss(h, h2)

        return h_loss





# 从NumPy文件中加载点云数据
def get_sample_data_from_numpy(file_path, threshold):
    # 从NumPy文件中加载点云数据
    pointcloud_data = np.load(file_path)  # 假设文件格式为 [num_points, 3]

    # 假设每次处理批次数据，转为 PyTorch 的 Tensor
    batch_size = 16
    num_points = pointcloud_data.shape[0] // batch_size  # 简化假设，按批次分割

    # 将NumPy数据转换为PyTorch张量
    initial_pointcloud = torch.from_numpy(pointcloud_data[:num_points, :]).float().unsqueeze(0)


    # 模拟真实的阈值标签
    true_threshold = torch.full((batch_size, 1), threshold)  # 设置固定的阈值0.06

    return initial_pointcloud, true_threshold


# 定义图像读取和预处理的转换
transform = transforms.Compose([
    transforms.ToTensor()  # 将图像转换为Tensor，并将像素值归一化到 [0, 1]
])


def load_image_as_tensor(image_path):
    image = Image.open(image_path).convert('RGB')  # 确保图像为RGB格式
    return transform(image)  # 应用转换，将图像转为tensor


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[4000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[4_000, 7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[4000, 7000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--pointcloud_path", type=str, required=True,
                        help="Path to the initial point cloud .npy file")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)



    # 初始点云数据
    initial_pointcloud_path = args.pointcloud_path
    initial_threshold = 0


    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载检查点
    checkpoint_path = os.path.join('ckpts', 'checkpoint.pth')
    checkpoint = torch.load(checkpoint_path)

    # 模型初始化
    model = PointNet().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])


    # 加载NumPy格式的初始点云数据
    initial_pointcloud, true_threshold = get_sample_data_from_numpy(initial_pointcloud_path, initial_threshold)

    # 将输入数据移动到GPU
    initial_pointcloud = initial_pointcloud.to(device)
    true_threshold = true_threshold.to(device)

    # 确保模型处于训练模式
    model.eval()


    predicted_threshold = model(initial_pointcloud)
    print("predicted_threshold", predicted_threshold)
    print("predicted_threshold:", f"{predicted_threshold.item():.4f}")

