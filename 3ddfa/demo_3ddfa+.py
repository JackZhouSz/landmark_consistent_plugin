#!/usr/bin/env python3
# coding: utf-8

__author__ = 'cleardusk'

### copy this file to 3ddfa root path, and then run this

"""
The pipeline of 3DDFA prediction: given one image, predict the 3d face vertices, 68 landmarks and visualization.

[todo]
1. CPU optimization: https://pmchojnacki.wordpress.com/2018/10/07/slow-pytorch-cpu-performance
"""

import torch
import torchvision.transforms as transforms
import mobilenet_v1
import numpy as np
import cv2
import dlib
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, predict_68pts_pytorch, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors
from utils.cv_plot import plot_pose_box
from utils.estimate_pose import parse_pose
from utils.render import get_depths_image, cget_depths_image, cpncc
from utils.paf import gen_img_paf
import argparse
import torch.backends.cudnn as cudnn

STD_SIZE = 120


def main(args):
    original = False # original dad3d val data. True: non-cropping, False: cropping.
    
    experiment_key = '1'
    epoch = 6
    experiments_dict = {
        '0': {'name':'3ddfa', 'ckpt':'training/snapshot/phase1_wpdc_checkpoint_bw_0.0_epoch_{}.pth.tar'.format(epoch)}, \
        '1': {'name':'3ddfa+', 'ckpt':'training/snapshot/phase1_wpdc_checkpoint_bw_0.05_epoch_{}.pth.tar'.format(epoch)}, \
    }
    # 1. load pre-tained model
    checkpoint_fp = experiments_dict[experiment_key]['ckpt']
    # checkpoint_fp = 'training/snapshot/phase1_wpdc_checkpoint_bw_1.0_epoch_1.pth.tar'
    arch = 'mobilenet_1'

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    if args.mode == 'gpu':
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()

    # 2. load dlib model for face detection and landmark used for face cropping
    if args.dlib_landmark:
        dlib_landmark_model = './shape_predictor/shape_predictor_68_face_landmarks.dat'
        face_regressor = dlib.shape_predictor(dlib_landmark_model)
    if args.dlib_bbox:
        face_detector = dlib.get_frontal_face_detector()

    # 3. forward
    tri = sio.loadmat('visualize/tri.mat')['tri']
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
    ####lbz modified####
    import os, pdb
    input_image_path = 'images'
    images_names_ = sorted(os.listdir(input_image_path))
    images_names = []
    for f in images_names_:
        if not f.endswith('.jpg') and not f.endswith('.jpeg') and not f.endswith('.png'):
            continue
        else:
            images_names.append(f)
    cnt = 0

    ####lbz modified####
    for img_name in images_names:
        img_fp = os.path.join(input_image_path, img_name)
        img_ori = cv2.imread(img_fp)
        if args.dlib_bbox:
            rects = face_detector(img_ori, 1)
        else:
            rects = []

        if len(rects) == 0:
            print('3DDFA detector fails on {}'.format(img_name))
            continue

            # rects = dlib.rectangles()
            # rect_fp = img_fp + '.bbox'
            # lines = open(rect_fp).read().strip().split('\n')[1:]
            # for l in lines:
            #     l, r, t, b = [int(_) for _ in l.split(' ')[1:]]
            #     rect = dlib.rectangle(l, r, t, b)
            #     rects.append(rect)

        pts_res = []
        Ps = []  # Camera matrix collection
        poses = []  # pose collection, [todo: validate it]
        vertices_lst = []  # store multiple face vertices
        ind = 0
        suffix = get_suffix(img_fp)
        for rect in rects:
            # whether use dlib landmark to crop image, if not, use only face bbox to calc roi bbox for cropping
            if args.dlib_landmark:
                # - use landmark for cropping
                pts = face_regressor(img_ori, rect).parts()
                pts = np.array([[pt.x, pt.y] for pt in pts]).T
                roi_box = parse_roi_box_from_landmark(pts)
            else:
                # - use detected face bbox
                bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]
                roi_box = parse_roi_box_from_bbox(bbox)

            img = crop_img(img_ori, roi_box)

            # forward: one step
            img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
            input = transform(img).unsqueeze(0)
            with torch.no_grad():
                if args.mode == 'gpu':
                    input = input.cuda()
                param_pytorch = model(input)
                param = param_pytorch.squeeze().cpu().numpy().flatten().astype(np.float32)

            # 68 pts
            # pts68 = predict_68pts(param, roi_box)
            pts68_pytorch = predict_68pts_pytorch(param_pytorch, roi_box)
            pts68 = pts68_pytorch.squeeze().cpu().numpy()
            
            # two-step for more accurate bbox to crop face
            if args.bbox_init == 'two':
                roi_box = parse_roi_box_from_landmark(pts68)
                img_step2 = crop_img(img_ori, roi_box)
                img_step2 = cv2.resize(img_step2, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
                input = transform(img_step2).unsqueeze(0)
                with torch.no_grad():
                    if args.mode == 'gpu':
                        input = input.cuda()
                    param = model(input)
                    param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

                pts68 = predict_68pts(param, roi_box)

            pts_res.append(pts68)
            print('cnt: {}/{}, image_name:{}'.format(str(cnt).zfill(4), str(len(images_names)).zfill(4), img_name))
            

        radius = max(1, int(min(img_ori.shape[:2]) * 0.005))
        points_pred = pts68.transpose(1, 0)[:, :2]
        for i in range(points_pred.shape[0]): cv2.circle(img_ori, (int(points_pred[i][0]), int(points_pred[i][1])), radius, (0, 0, 255), -1) 
        cv2.imwrite('./outputs/{}_{}.jpg'.format(img_name.split('.')[0], experiments_dict[experiment_key]['name']), img_ori)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    parser.add_argument('-f', '--files', nargs='+',
                        help='image files paths fed into network, single or multiple images')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--show_flg', default='true', type=str2bool, help='whether show the visualization result')
    parser.add_argument('--bbox_init', default='one', type=str,
                        help='one|two: one-step bbox initialization or two-step')
    parser.add_argument('--dump_res', default='true', type=str2bool, help='whether write out the visualization image')
    parser.add_argument('--dump_vertex', default='false', type=str2bool,
                        help='whether write out the dense face vertices to mat')
    parser.add_argument('--dump_ply', default='true', type=str2bool)
    parser.add_argument('--dump_pts', default='true', type=str2bool)
    parser.add_argument('--dump_roi_box', default='false', type=str2bool)
    parser.add_argument('--dump_pose', default='true', type=str2bool)
    parser.add_argument('--dump_depth', default='true', type=str2bool)
    parser.add_argument('--dump_pncc', default='true', type=str2bool)
    parser.add_argument('--dump_paf', default='false', type=str2bool)
    parser.add_argument('--paf_size', default=3, type=int, help='PAF feature kernel size')
    parser.add_argument('--dump_obj', default='true', type=str2bool)
    parser.add_argument('--dlib_bbox', default='true', type=str2bool, help='whether use dlib to predict bbox')
    parser.add_argument('--dlib_landmark', default='true', type=str2bool,
                        help='whether use dlib landmark to crop image')

    args = parser.parse_args()
    main(args)
