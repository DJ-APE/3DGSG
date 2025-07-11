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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, threshold, checkpoint_path, intersection_path):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    print("dataset", dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)


    (model_params, first_iter) = torch.load(checkpoint_path)
    gaussians.restore(model_params, opt)

    #gaussians._features_dc = torch.zeros_like(gaussians._features_dc)

    # 假设你的 PyTorch 张量是 size 为 (9722, 3) 的 3D 点
    points = gaussians._xyz  # 用随机数据举例
    print("高斯点的规模", points.shape)



    # 假设你有一个 (1521, 3) 大小的 numpy 数组，表示 3D 点
    target_points_np = np.load(intersection_path)

    # 将 numpy 数组转换为 PyTorch 张量
    target_points = torch.tensor(target_points_np)
    print("初步点的规模", target_points.shape)


    # 选择设备，cuda:0 或 cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 将张量移动到选定的设备
    points = points.to(device)
    target_points = target_points.to(device)


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

    # 将结果转换为张量形式
    close_points = torch.stack(close_points)
    indices = torch.tensor(indices)

    # 打印接近的点和索引
    print(f"Close points: {close_points}")
    print(f"Indices of close points: {indices}")
    print(indices.shape)

    #torch.save(indices, '/home/djape/gust/3DGS/gaussian-splatting-main/mytest/clip/names3/finally106/A black keyboard.pt')

    # 使用 torch.no_grad() 禁用梯度追踪
    # 选择合适的模式
    with torch.no_grad():

        # 其它全白
        for i in tqdm(range(points.size(0)), desc="Points to white"):
            if i not in indices:
                gaussians._features_dc[i].zero_()  # 将张量的值全部设为 0
                #gaussians._opacity[i].zero_()
                gaussians._features_dc[i, :, :]  = 2

        # 自己白色
        # for i in indices:
        #     gaussians._features_dc[i].zero_()  # 将张量的值全部设为 0
        #     #gaussians._opacity[i].zero_()
        #     gaussians._features_dc[i, :, :]  = 0.1



    # # 生成图片使用
    # bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    # background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    #
    # viewpoint_stack = scene.getTrainCameras().copy()
    # for cam in viewpoint_stack:
    #     if cam.image_name == "frame_0063":
    #         print(1)
    #         render_pkg = render(cam, gaussians, pipe, background)
    #         print("1", cam.image_name)
    #         image = render_pkg["render"].detach()
    #         custom_path = "./mytest/1/"
    #         torchvision.utils.save_image(image, f"{custom_path}/{cam.image_name}.jpg")
    #
    #
    #
    #




    iteration = "GSG"

    print("\n[ITER {}] Saving Gaussians".format(iteration))
    scene.save(iteration)

    print("\n[ITER {}] Saving Checkpoint".format(iteration))
    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")



    # bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    # background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    #
    # iter_start = torch.cuda.Event(enable_timing = True)
    # iter_end = torch.cuda.Event(enable_timing = True)
    #
    # viewpoint_stack = None
    # ema_loss_for_log = 0.0
    # progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    # first_iter += 1
    # for iteration in range(first_iter, opt.iterations + 1):
    #     if network_gui.conn == None:
    #         network_gui.try_connect()
    #     while network_gui.conn != None:
    #         try:
    #             net_image_bytes = None
    #             custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
    #             if custom_cam != None:
    #                 net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
    #                 net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
    #             network_gui.send(net_image_bytes, dataset.source_path)
    #             if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
    #                 break
    #         except Exception as e:
    #             network_gui.conn = None
    #
    #     iter_start.record()
    #
    #     gaussians.update_learning_rate(iteration)
    #
    #     # Every 1000 its we increase the levels of SH up to a maximum degree
    #     if iteration % 1000 == 0:
    #         gaussians.oneupSHdegree()
    #
    #     # Pick a random Camera
    #     if not viewpoint_stack:
    #         viewpoint_stack = scene.getTrainCameras().copy()
    #     viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
    #
    #     # Render
    #     if (iteration - 1) == debug_from:
    #         pipe.debug = True
    #
    #     bg = torch.rand((3), device="cuda") if opt.random_background else background
    #
    #     render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
    #     image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
    #
    #     # Loss
    #     gt_image = viewpoint_cam.original_image.cuda()
    #     Ll1 = l1_loss(image, gt_image)
    #     loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
    #     loss.backward()
    #
    #     iter_end.record()
    #
    #     with torch.no_grad():
    #         # Progress bar
    #         ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
    #         if iteration % 10 == 0:
    #             progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
    #             progress_bar.update(10)
    #         if iteration == opt.iterations:
    #             progress_bar.close()
    #
    #         # Log and save
    #         training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
    #         if (iteration in saving_iterations):
    #             print("\n[ITER {}] Saving Gaussians".format(iteration))
    #             scene.save(iteration)
    #
    #         # # Densification
    #         # if iteration < opt.densify_until_iter:
    #         #     # Keep track of max radii in image-space for pruning
    #         #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
    #         #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
    #         #
    #         #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
    #         #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
    #         #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
    #         #
    #         #     if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
    #
    #         #         gaussians.reset_opacity()
    #
    #         # Optimizer step
    #         if iteration < opt.iterations:
    #             gaussians.optimizer.step()
    #             gaussians.optimizer.zero_grad(set_to_none = True)
    #
    #         if (iteration in checkpoint_iterations):
    #             print("\n[ITER {}] Saving Checkpoint".format(iteration))
    #             torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
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
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--threshold", type=float, default=0.06, help="初始阈值")  # ✅ 新增参数
    parser.add_argument('--checkpoint_path', type=str, required=True, help="模型 checkpoint 路径 (.pth)")
    parser.add_argument('--intersection_path', type=str, required=True, help="交集点云路径 (.npy)")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    #network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,
             args.threshold, args.checkpoint_path, args.intersection_path)

    # All done
    print("\nTraining complete.")
