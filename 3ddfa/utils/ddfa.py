#!/usr/bin/env python3
# coding: utf-8

import os.path as osp
import os
import random
from pathlib import Path
import numpy as np

import torch
import torch.utils.data as data
import cv2
import pickle
import argparse
from .io import _numpy_to_tensor, _load_cpu, _load_gpu
from .params import *


def _parse_param(param):
    """Work for both numpy and tensor"""
    p_ = param[:12].reshape(3, -1)
    p = p_[:, :3]
    offset = p_[:, -1].reshape(3, 1)
    alpha_shp = param[12:52].reshape(-1, 1)
    alpha_exp = param[52:].reshape(-1, 1)
    return p, offset, alpha_shp, alpha_exp

def _parse_param_batch(param):
    """Work for both numpy and tensor
    param: B, 62
    """
    B, N = param.shape
    p_ = param[:, :12].reshape(B, 3, -1)
    p = p_[:, :, :3]
    offset = p_[:, :, -1].reshape(B, 3, 1)
    alpha_shp = param[:, 12:52].reshape(B, -1, 1)
    alpha_exp = param[:, 52:].reshape(B, -1, 1)
    return p, offset, alpha_shp, alpha_exp


def reconstruct_vertex(param, whitening=True, dense=False, transform=True):
    """Whitening param -> 3d vertex, based on the 3dmm param: u_base, w_shp, w_exp
    dense: if True, return dense vertex, else return 68 sparse landmarks. All dense or sparse vertex is transformed to
    image coordinate space, but without alignment caused by face cropping.
    transform: whether transform to image space
    """
    if len(param) == 12:
        param = np.concatenate((param, [0] * 50))
    if whitening:
        if len(param) == 62:
            param = param * param_std + param_mean
        else:
            
            param = np.concatenate((param[:11], [0], param[11:]))
            param = param * param_std + param_mean

    p, offset, alpha_shp, alpha_exp = _parse_param(param)

    if dense:
        vertex = p @ (u + w_shp @ alpha_shp + w_exp @ alpha_exp).reshape(3, -1, order='F') + offset

        if transform:
            # transform to image coordinate space
            vertex[1, :] = std_size + 1 - vertex[1, :]
    else:
        """For 68 pts"""
        # import pdb; pdb.set_trace()
        vertex = p @ (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp).reshape(3, -1, order='F') + offset

        if transform:
            # transform to image coordinate space
            vertex[1, :] = std_size + 1 - vertex[1, :]

    return vertex



def reconstruct_vertex_pytorch(param, whitening=True, dense=False, transform=True):
    """Whitening param -> 3d vertex, based on the 3dmm param: u_base, w_shp, w_exp
    dense: if True, return dense vertex, else return 68 sparse landmarks. All dense or sparse vertex is transformed to
    image coordinate space, but without alignment caused by face cropping.
    transform: whether transform to image space
    """
    B, N = param.shape

    device = param.device
    param_std_pt = torch.from_numpy(param_std).to(device).repeat(B, 1)
    param_mean_pt = torch.from_numpy(param_mean).to(device).repeat(B, 1)
    u_base_pt = torch.from_numpy(u_base).to(device).repeat(B, 1, 1)
    w_shp_base_pt = torch.from_numpy(w_shp_base).to(device).repeat(B, 1, 1)
    w_exp_base_pt = torch.from_numpy(w_exp_base).to(device).repeat(B, 1, 1)
    
    
    param = param * param_std_pt + param_mean_pt
    p, offset, alpha_shp, alpha_exp = _parse_param_batch(param)
        
    vertex = torch.bmm(p, (u_base_pt + torch.bmm(w_shp_base_pt, alpha_shp) + torch.bmm(w_exp_base_pt, alpha_exp)).reshape(B, -1, 3).permute(0, 2, 1)) + offset

    if transform:
        # transform to image coordinate space
        vertex[:, 1, :] = std_size + 1 - vertex[:, 1, :]

    return vertex


def img_loader(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ToTensorGjz(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NormalizeGjz(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor.sub_(self.mean).div_(self.std)
        return tensor


class DDFADataset(data.Dataset):
    def __init__(self, root, filelists, param_fp, transform=None, **kargs):
        self.root = root
        self.transform = transform
        self.lines = Path(filelists).read_text().strip().split('\n')
        self.params = _numpy_to_tensor(_load_cpu(param_fp))
        self.img_loader = img_loader

    def _target_loader(self, index):
        target = self.params[index]

        return target

    def __getitem__(self, index):
        path = osp.join(self.root, self.lines[index])
        img = self.img_loader(path)

        target = self._target_loader(index)

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.lines)


class DDFATestDataset(data.Dataset):
    def __init__(self, filelists, root='', transform=None):
        self.root = root
        self.transform = transform
        self.lines = Path(filelists).read_text().strip().split('\n')

    def __getitem__(self, index):
        path = osp.join(self.root, self.lines[index])
        img = img_loader(path)

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.lines)



class DDFADatasetBooster(data.Dataset):
    def __init__(self, root, filelists, param_fp, config, transform=None, **kargs):
        self.root = root
        self.transform = transform
        self.lines = Path(filelists).read_text().strip().split('\n')
        self.params = _numpy_to_tensor(_load_cpu(param_fp))
        self.img_loader = img_loader

        self.multiview_dir = config["dataset_multiview_root"]
        self.multiview_lm2d_dir = config["dataset_multiview_lm2d_root"]
        self.scenes = np.load(config["sample_list"]).tolist()[:config['sample_num']]
        self.loader_name = config["loader_name"]
        self.img_size = config["img_size"]

        self.cam_ext = np.load('/nfs/STG/CodecAvatar/lelechen/libingzeng/EG3D_Inversion/eg3d/out/out/outputs_views_ortho_mouth_cam/cam2world_pose.npy')

    def _target_loader(self, index):
        target = self.params[index]

        return target

    def __getitem__(self, index):
        
        scene_name = self.scenes[index]
        scene_multiview_dir = os.path.join(self.multiview_dir, '{}.mp4'.format(scene_name))
        scene_multiviews = []
        cap = cv2.VideoCapture(scene_multiview_dir)
        success,image = cap.read()
        assert success, "reading mp4 of {} fails".format(scene_name)
        while success:
            success,image = cap.read()
            scene_multiviews.append(image)

        scene_multiview_lm2d_dir = os.path.join(self.multiview_lm2d_dir, 'lm2d_68_multiview_2d', '{}_lm2d_68_multiview_2d.npy'.format(scene_name))
        scene_multiview_lm2d = np.load(scene_multiview_lm2d_dir)
        assert scene_multiview_lm2d.shape[0] == 512 and scene_multiview_lm2d.shape[1] == 68 and scene_multiview_lm2d.shape[2] == 2, 'scene_multiview_lm2d : {:}'.format(KRT.shape)

        sample_num = 32
        if self.loader_name == 'val':
            # views_samples_list = random.sample(range(512-1), sample_num)
            # views_samples_list = [0, 125, 250, 375, 390]
            views_samples_list = [int(i * 512 / sample_num) for i in range(sample_num)]
        else:
            views_samples_list = random.sample(range(512-1), sample_num)
            # views_samples_list = [0, 125, 250, 375, 390]
            # views_samples_list = [int(i * 512 / sample_num) for i in range(sample_num)]
        # t2 = time.time()
        # print('t2-t1:others, {}'.format(t2-t1))

        names, views, KRTs = [], [], []
        flame_meshes = [] # all views share the same mesh, just repeatedly save.
        lm2d_points_gt = []
        # keypoint_indices = []
        for v in views_samples_list:
            im_ = scene_multiviews[v]
            im_ = cv2.cvtColor(im_, cv2.COLOR_BGR2RGB, dst=im_)
            im = cv2.resize(im_, (self.img_size, self.img_size), interpolation = cv2.INTER_AREA)
            view_name = 'view_{}'.format(str(v).zfill(3))
            im = self.transform(im)
            K = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]]) # * torch.tensor([[256],[256], [1]])
            RT = torch.inverse(torch.tensor(self.cam_ext[v]))[:3, :]   # Notice: KRT should use world2cam matrix not cam2world matrix
            KRT = torch.mm(K, RT)
                        
            # lm2d_view_dir = os.path.join(lm2d_folder, 'frame_{}_lm2d_68.npy'.format(str(v).zfill(3)))
            lm2d_points = torch.tensor(scene_multiview_lm2d[v]).float() / 512.0
            # lm2d_points = torch.tensor(np.load(lm2d_view_dir)).float() / 512.0

            names.append(scene_name + '_' + view_name)
            # view_names.append(view_name)
            views.append(im)
            KRTs.append(KRT)
            
            lm2d_points_gt.append(lm2d_points)

        # if self.config['single_view_gt']:
        #     im = self.img_loader(os.path.join(self.config['single_view_img_gt_path'], '{}.png'.format(scene_name)))
        #     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB, dst=im)
        #     im = self._transform2(im)
        #     im = torch.tensor(im).permute(2, 0, 1)
        #     K = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]]) # * torch.tensor([[256],[256], [1]])
        #     cam_gt_path = os.path.join(self.config['single_view_cam_gt_path'], '{}.npy'.format(scene_name))
        #     RT = torch.inverse(torch.tensor(np.load(cam_gt_path)[:16].reshape(4, 4))).float()[:3, :]   # Notice: KRT should use world2cam matrix not cam2world matrix
        #     KRT = torch.mm(K, RT)
                        
        #     lm2d_gt_path = os.path.join(self.config['single_view_lm2d_gt_path'], '{}__cropped.npy'.format(scene_name))
        #     lm2d_points = torch.tensor(np.load(lm2d_gt_path)).float() / 512.0

        #     names.append(scene_name + '_gt')
        #     views.append(im)
        #     KRTs.append(KRT)
            
        #     lm2d_points_gt.append(lm2d_points)

        views = torch.stack(views, dim=0) # Vx3xHxW
        KRTs = torch.stack(KRTs, dim=0) # Vx3x4
        lm2d_points_gt = torch.stack(lm2d_points_gt, dim=0) # Vx68x2   

        original_data_samples_num = 512 #3#22
        original_data_samples_list = random.sample(range(len(self.lines)-1), original_data_samples_num)
        original_data_items_img = []
        original_data_items_target = []
        for i in original_data_samples_list:
            path = osp.join(self.root, self.lines[i])
            img = self.img_loader(path)

            target = self._target_loader(i)

            if self.transform is not None:
                img = self.transform(img)

            original_data_items_img.append(img)
            original_data_items_target.append(target)
        
        original_data_items_img = torch.stack(original_data_items_img, dim=0)
        original_data_items_target = torch.stack(original_data_items_target, dim=0)
        original_data_items = {'img':original_data_items_img, 'target':original_data_items_target}
        
        batch = {'names':names, 'views':views, 'KRTs':KRTs, 'lm2d_points_gt':lm2d_points_gt, 'original_data_items':original_data_items}


        return batch

    def __len__(self):
        return len(self.scenes)
