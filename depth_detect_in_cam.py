import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging, draw_measure_line)
from utils.torch_utils import select_device, load_classifier, time_synchronized
import open3d as o3d
from utils.depth_detect.eval import read_kitti_intrinsics
import os


def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model

    # print(model)
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    # cv2.namedWindow("depth", cv2.WINDOW_NORMAL)
    for path, img, im0s, vid_cap in dataset:
        # file_name = path.split("/")[-1]
        # file_ID = file_name[0:-4]
        # cali_file_path = "/home/tuxiang/DataSets/KITTI/Object3D/object/training/calib"
        # cali_file_path = os.path.join(cali_file_path,file_ID+".txt")
        # intrinsics_matrix = read_kitti_intrinsics(cali_file_path)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()

        pred_ = model(img, augment=opt.augment)
        pred = pred_[0]

        # Apply NMS
        # 目前线索是非极大值抑制函数传进来的pred已经变成了真实的xywh,前4个数字时xywh
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        # pred1_numpy = np.array(pred)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        location = []
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_img or view_img:  # Add bbox to image
                        # label = '%s %.2f' % (names[int(cls)], conf)
                        label = '%s' % (names[int(cls)])

                        # 内参
                        intrinsics_matrix = [960, 540,
                                             775.9, 776.9]  # cx,cy,fx,fy

                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        # 计算并绘制结果
                        xyz_in_camera = draw_measure_line(xyxy, im0, size=2, color=colors[int(cls)], label=cls,
                                                          intrinsics_matrix=intrinsics_matrix)

                        location.append(xyz_in_camera)
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                        if np.asarray(xyz_in_camera).size == 1:
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 8 + '%s' + '\n') % (cls, *xyxy, 0, 0, 0, file_ID))
                        else:
                            x_cam = xyz_in_camera[0]
                            y_cam = xyz_in_camera[1]
                            z_cam = xyz_in_camera[2]
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 8 + '%s' + '\n') % (
                                    cls, *xyxy, x_cam, y_cam, z_cam, file_ID))  # label format
            # else:
            #     with open(txt_path + '.txt', 'a') as f:
            #         pass

            # Print time (inference + NMS)

            print('%sDone. (%.3fs),FPS=%.3f' % (s, t2 - t1, 1 / (t2 - t1)))

            # Stream results
            if view_img:
                cv2.imshow("depth1", im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform.system() == 'Darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

    # 部分非地面目标储存为None
    # print(location)

    if False:

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        from utils.RBG_Depth_to_Clouds import rgb_depth_to_pointcloud
        depth_raw = cv2.imread("/home/tuxiang/DataSets/KITTI/Object3D/object/training/depth/000114.png",
                               -1)  # 参数-1是为了保持uint16格式
        rgb_raw = cv2.imread("/home/tuxiang/DataSets/KITTI/Object3D/object/training/image_2/000114.png")
        rgb_raw = rgb_raw[:, :, ::-1]  # openCV中读取的BGR  其中，[::-1] 表示顺序相反操作 ，如下面操作：
        # rgb_point_mapping, index = rgb_point_mapping(depth_raw,depth_trunc=30000)
        pcd = rgb_depth_to_pointcloud(rgb_raw, depth_raw, intrinsics_matrix=intrinsics_matrix)
        vis.add_geometry(pcd)

        for i in location:
            if np.asarray(i).size == 1:
                continue
            if i[2] > 300:
                continue
            ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
            ball.paint_uniform_color([1, 1, 1])
            ball.translate(i)
            # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            # size=1, origin=i)
            vis.add_geometry(ball)
            # vis.add_geometry(mesh_frame)

        vis.poll_events()
        vis.update_renderer()
        visopt = vis.get_render_option()
        visopt.background_color = np.asarray([0, 0, 0])
        vis.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
