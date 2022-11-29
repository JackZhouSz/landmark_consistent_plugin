#!/usr/bin/env python3
# coding: utf-8

import os.path as osp
from pathlib import Path
import numpy as np
import argparse
import time
import logging

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import mobilenet_v1
import torch.backends.cudnn as cudnn

from utils.ddfa import DDFADatasetBooster, ToTensorGjz, NormalizeGjz
from utils.ddfa import str2bool, AverageMeter
from utils.io import mkdir
from vdc_loss import VDCLoss
from wpdc_loss import WPDCLoss
from utils.ddfa import reconstruct_vertex_pytorch
from utils.inference import draw_landmarks

import pdb

# global args (configuration)
args = None
lr = None
arch_choices = ['mobilenet_2', 'mobilenet_1', 'mobilenet_075', 'mobilenet_05', 'mobilenet_025']


def parse_args():
    parser = argparse.ArgumentParser(description='3DMM Fitting')
    parser.add_argument('-j', '--workers', default=6, type=int)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--start-epoch', default=1, type=int)
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('-vb', '--val-batch-size', default=32, type=int)
    parser.add_argument('--base-lr', '--learning-rate', default=0.001, type=float)
    parser.add_argument('--bw', '--booster-weight', default=0.0, type=float)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float)
    parser.add_argument('--print-freq', '-p', default=20, type=int)
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    parser.add_argument('--devices-id', default='0,1', type=str)
    parser.add_argument('--filelists-train',
                        default='', type=str)
    parser.add_argument('--filelists-val',
                        default='', type=str)
    parser.add_argument('--root', default='')
    parser.add_argument('--snapshot', default='', type=str)
    parser.add_argument('--log-file', default='output.log', type=str)
    parser.add_argument('--log-mode', default='w', type=str)
    parser.add_argument('--size-average', default='true', type=str2bool)
    parser.add_argument('--num-classes', default=62, type=int)
    parser.add_argument('--arch', default='mobilenet_1', type=str,
                        choices=arch_choices)
    parser.add_argument('--frozen', default='false', type=str2bool)
    parser.add_argument('--milestones', default='15,25,30', type=str)
    parser.add_argument('--task', default='all', type=str)
    parser.add_argument('--test_initial', default='false', type=str2bool)
    parser.add_argument('--warmup', default=-1, type=int)
    parser.add_argument('--param-fp-train',
                        default='',
                        type=str)
    parser.add_argument('--param-fp-val',
                        default='')
    parser.add_argument('--opt-style', default='resample', type=str)  # resample
    parser.add_argument('--resample-num', default=132, type=int)
    parser.add_argument('--loss', default='vdc', type=str)

    global args
    args = parser.parse_args()

    # some other operations
    args.devices_id = [int(d) for d in args.devices_id.split(',')]
    args.milestones = [int(m) for m in args.milestones.split(',')]

    snapshot_dir = osp.split(args.snapshot)[0]
    mkdir(snapshot_dir)


def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)


def adjust_learning_rate(optimizer, epoch, milestones=None):
    """Sets the learning rate: milestone is a list/tuple"""

    def to(epoch):
        if epoch <= args.warmup:
            return 1
        elif args.warmup < epoch <= milestones[0]:
            return 0
        for i in range(1, len(milestones)):
            if milestones[i - 1] < epoch <= milestones[i]:
                return i
        return len(milestones)

    n = to(epoch)

    global lr
    lr = args.base_lr * (0.2 ** n)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info(f'Save checkpoint to {filename}')



# many multiview systems, each of them has N cameras with N KRTs and [N * P * 2] points
# KRTs   : M * N * 3 * 4
# points : M * N * P * 2
def TriangulateDLT_BatchCam(KRTs, points):
  assert KRTs.dim() == 4 and points.dim() == 4 and KRTs.size(0) == points.size(0) and KRTs.size(1) == points.size(1), 'KRTs : {:}, points : {:}'.format(KRTs.shape, points.shape)
  assert KRTs.size(2) == 3 and KRTs.size(3) == 4 and points.size(-1) == 2, 'KRTs : {:}, points : {:}'.format(KRTs.shape, points.shape)
  assert points.size(-2) >= 3, 'There should be at least 3 points'.format(points.shape)
  batch_mv, batch_cam = KRTs.size(0), KRTs.size(1)
  KRTs = KRTs.view(batch_mv, batch_cam, 1, 3, 4)      # size = M * N * 1 * 3 * 4
  U = KRTs[...,0,:] - KRTs[...,2,:] * points[...,0:1] # size = M * N * P * 4
  V = KRTs[...,1,:] - KRTs[...,2,:] * points[...,1:2] 
  Dmatrix = torch.cat((U,V), dim=1).transpose(1,2)    # size = M * P * 2N * 4
  A      = Dmatrix[...,:3]                            # size = M * P * 2N * 3
  At     = torch.transpose(A, 2, 3)                   # size = M * P * 3 * 2N
  AtA    = torch.matmul(At, A)                        # size = M * P * 3 * 3
  invAtA = torch.inverse( AtA )
  P3D    = torch.matmul(invAtA, torch.matmul(At, -Dmatrix[...,3:]))
  return P3D.view(batch_mv, -1, 3)


# Batch KRT and Batch PTS3D
# KRT     : .... x 3 x 4
# PTS3D   : .... x N x 3
# projPTS : .... x N x 2
def ProjectKRT_Batch(KRT, PTS3D):
  assert KRT.dim() == PTS3D.dim() and PTS3D.size(-1) == 3 and KRT.size(-2) == 3 and KRT.size(-1) == 4, 'KRT : {:} | PTS3D : {:}'.format(KRT.shape, PTS3D.shape)
  MPs  = torch.matmul(KRT[...,:3], PTS3D.transpose(-1,-2)) + KRT[...,3:]
  NMPs = MPs.transpose(-1,-2)
  projPTS = NMPs[..., :2] / NMPs[..., 2:]
  return projPTS


def train(train_loader, model, criterion, optimizer, epoch, config):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_original = AverageMeter()
    losses_booster = AverageMeter()
    losses = AverageMeter()

    model.train()
    booster_weight = config['booster_weight']

    end = time.time()
    # loader is batch style
    # for i, (input, target) in enumerate(train_loader):
    
    for i, batch in enumerate(train_loader):
        # weight_original = self.weight_original
        # weight_proj = self.weight_proj
        # weight_proj_ref_gt = self.weight_proj_ref_gt
        # weight_proj_srt_gt = self.weight_proj_srt_gt
        # weight_lm2d = self.weight_lm2d
        num_views_srt = 4 # int(len(names)/4)

        names = batch['names']
        images = batch['views'].cuda() # BxVx3xHxW (512)
        krts = batch['KRTs'].cuda() # Vx3x4
        lm_2d_gt = batch['lm2d_points_gt'].cuda()
        original_data_img = batch['original_data_items']['img'][0].cuda() # 512x3x120x120
        original_data_target = batch['original_data_items']['target'][0].cuda() # 512x62
        original_data_target.requires_grad = False
        original_data_target = original_data_target.cuda(non_blocking=True)
        B, V, C, H, W = images.shape
        output_all = model(torch.cat((images.reshape(-1, C, H, W), original_data_img), 0))
        
        data_time.update(time.time() - end)
        
        output_booster = output_all[:B*V]
        output_original = output_all[B*V:]
        loss_original = criterion(output_original, original_data_target)
                
        lm_2d = reconstruct_vertex_pytorch(output_booster)[:, :2, :].permute(0, 2, 1).unsqueeze(0) / H

        names_tgt = names[num_views_srt:]
        images_tgt = images[:, num_views_srt:, :, :, :]
        lm_2d_srt = lm_2d[:, :num_views_srt, :, :]
        krts_srt = krts[:, :num_views_srt, :, :]
        lm_2d_tgt = lm_2d[:, num_views_srt:, :, :]
        krts_tgt = krts[:, num_views_srt:, :, :]
        lm_2d_gt_srt = lm_2d_gt[:, :num_views_srt, :, :]
        lm_2d_gt_tgt = lm_2d_gt[:, num_views_srt:, :, :]
        lm_3d_srt = TriangulateDLT_BatchCam(krts_srt, lm_2d_srt)
        lm_2d_proj = ProjectKRT_Batch(krts, lm_3d_srt.unsqueeze(1))
        lm_2d_proj_srt = lm_2d_proj[:, :num_views_srt, :, :]
        lm_2d_proj_tgt = lm_2d_proj[:, num_views_srt:, :, :]
        lm_2d_loss = torch.nn.L1Loss()(lm_2d, lm_2d_gt)
        lm_2d_proj_loss = torch.nn.L1Loss()(lm_2d_tgt, lm_2d_proj_tgt)
        lm_2d_proj_srt_gt_loss = torch.nn.L1Loss()(lm_2d_gt_srt, lm_2d_proj_srt)
        loss_booster = (lm_2d_proj_loss + lm_2d_proj_srt_gt_loss + lm_2d_loss) * booster_weight
        loss = loss_original + loss_booster


        # import os
        # output_path = '/nfs/home/uss00054/projects/3DDFA/outputs'
        # os.makedirs(output_path, exist_ok=True)
        # pts68_np = pts68.detach().cpu().numpy()
        # for i in range(len(input)):
        #     img_output_dir = os.path.join(output_path, 'lm2d_{}_3ddfa.jpg'.format(str(i).zfill(3)))
        #     img = input[i].permute(1, 2, 0).detach().cpu().numpy() * 128 + 127.5
        #     pdb.set_trace()
        #     # img_ori.shape, (669, 486, 3)
        #     # [pts68_np[i]]
        #     # args.show_flg = True
        #     draw_landmarks(img, [pts68_np[i]], wfp=img_output_dir, show_flg=True)

        data_time.update(time.time() - end)

        losses_original.update(loss_original.item(), 1)
        losses_booster.update(loss_booster.item(), 1)
        losses.update(loss.item(), 1)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # log
        if i % args.print_freq == 0:
            logging.info(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                         f'LR: {lr:8f}\t'
                         f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         # f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         f'Loss_original {losses_original.val:.4f} ({losses_original.avg:.4f})\t'
                         f'Loss_booster {losses_booster.val:.4f} ({losses_booster.avg:.4f})\t'
                         f'Loss {losses.val:.4f} ({losses.avg:.4f})')


def validate(val_loader, model, criterion, epoch):
    model.eval()
    batch_time = AverageMeter()
    losses_original = AverageMeter()
    losses_booster = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    with torch.no_grad():
        losses = []
        for i, batch in enumerate(val_loader):
            num_views_srt = 4 # int(len(names)/4)

            names = batch['names']
            images = batch['views'].cuda() # BxVx3xHxW (512)
            krts = batch['KRTs'].cuda() # Vx3x4
            lm_2d_gt = batch['lm2d_points_gt'].cuda()
            original_data_img = batch['original_data_items']['img'][0].cuda() # 512x3x120x120
            original_data_target = batch['original_data_items']['target'][0].cuda() # 512x62
            original_data_target.requires_grad = False
            original_data_target = original_data_target.cuda(non_blocking=True)
            B, V, C, H, W = images.shape
            output_all = model(torch.cat((images.reshape(-1, C, H, W), original_data_img), 0))
                        
            output_booster = output_all[:B*V]
            output_original = output_all[B*V:]
            loss_original = criterion(output_original, original_data_target)
                    
            lm_2d = reconstruct_vertex_pytorch(output_booster)[:, :2, :].permute(0, 2, 1).unsqueeze(0) / H

            names_tgt = names[num_views_srt:]
            images_tgt = images[:, num_views_srt:, :, :, :]
            lm_2d_srt = lm_2d[:, :num_views_srt, :, :]
            krts_srt = krts[:, :num_views_srt, :, :]
            lm_2d_tgt = lm_2d[:, num_views_srt:, :, :]
            krts_tgt = krts[:, num_views_srt:, :, :]
            lm_2d_gt_srt = lm_2d_gt[:, :num_views_srt, :, :]
            lm_2d_gt_tgt = lm_2d_gt[:, num_views_srt:, :, :]
            lm_3d_srt = TriangulateDLT_BatchCam(krts_srt, lm_2d_srt)
            lm_2d_proj = ProjectKRT_Batch(krts, lm_3d_srt.unsqueeze(1))
            lm_2d_proj_srt = lm_2d_proj[:, :num_views_srt, :, :]
            lm_2d_proj_tgt = lm_2d_proj[:, num_views_srt:, :, :]
            lm_2d_loss = torch.nn.L1Loss()(lm_2d, lm_2d_gt)
            lm_2d_proj_loss = torch.nn.L1Loss()(lm_2d_tgt, lm_2d_proj_tgt)
            lm_2d_proj_srt_gt_loss = torch.nn.L1Loss()(lm_2d_gt_srt, lm_2d_proj_srt)
            loss_booster = lm_2d_proj_loss + lm_2d_proj_srt_gt_loss + lm_2d_loss
            loss = loss_original + loss_booster


            # import os
            # output_path = '/nfs/home/uss00054/projects/3DDFA/outputs'
            # os.makedirs(output_path, exist_ok=True)
            # pts68_np = pts68.detach().cpu().numpy()
            # for i in range(len(input)):
            #     img_output_dir = os.path.join(output_path, 'lm2d_{}_3ddfa.jpg'.format(str(i).zfill(3)))
            #     img = input[i].permute(1, 2, 0).detach().cpu().numpy() * 128 + 127.5
            #     pdb.set_trace()
            #     # img_ori.shape, (669, 486, 3)
            #     # [pts68_np[i]]
            #     # args.show_flg = True
            #     draw_landmarks(img, [pts68_np[i]], wfp=img_output_dir, show_flg=True)


            losses_original.update(loss_original.item(), 1)
            losses_booster.update(loss_booster.item(), 1)
            losses.update(loss.item(), 1)
            # compute gradient and do SGD step

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # log
            if i % args.print_freq == 0:
                logging.info(f'Epoch: [{epoch}][{i}/{len(val_loader)}]\t'
                            f'LR: {lr:8f}\t'
                            f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            # f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                            f'Loss_original {losses_original.val:.4f} ({losses_original.avg:.4f})\t'
                            f'Loss_booster {losses_booster.val:.4f} ({losses_booster.avg:.4f})\t'
                            f'Loss {losses.val:.4f} ({losses.avg:.4f})')

def main():
    parse_args()  # parse global argsl

    # logging setup
    logging.basicConfig(
        format='[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode=args.log_mode),
            logging.StreamHandler()
        ]
    )

    print_args(args)  # print args

    # step1: define the model structure
    model = getattr(mobilenet_v1, args.arch)(num_classes=args.num_classes)

    torch.cuda.set_device(args.devices_id[0])  # fix bug for `ERROR: all tensors must be on devices[0]`

    model = nn.DataParallel(model, device_ids=args.devices_id).cuda()  # -> GPU

    # step2: optimization: loss and optimization method
    # criterion = nn.MSELoss(size_average=args.size_average).cuda()
    if args.loss.lower() == 'wpdc':
        print(args.opt_style)
        criterion = WPDCLoss(opt_style=args.opt_style).cuda()
        logging.info('Use WPDC Loss')
    elif args.loss.lower() == 'vdc':
        criterion = VDCLoss(opt_style=args.opt_style).cuda()
        logging.info('Use VDC Loss')
    elif args.loss.lower() == 'pdc':
        criterion = nn.MSELoss(size_average=args.size_average).cuda()
        logging.info('Use PDC loss')
    else:
        raise Exception(f'Unknown Loss {args.loss}')

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.base_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    # step 2.1 resume
    if args.resume:
        if Path(args.resume).is_file():
            logging.info(f'=> loading checkpoint {args.resume}')
            
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)['state_dict']
            # checkpoint = torch.load(args.resume)['state_dict']
            model.load_state_dict(checkpoint)

        else:
            logging.info(f'=> no checkpoint found at {args.resume}')

    # step3: data
    normalize = NormalizeGjz(mean=127.5, std=128)  # may need optimization

    cfg_train = {'dataset_multiview_root':'/nfs/STG/CodecAvatar/lelechen/DAD-3DHeadsDataset/train/PTI/multiview', \
                 'dataset_multiview_lm2d_root':'/nfs/STG/CodecAvatar/lelechen/libingzeng/DAD-3DHeadsDataset/train/PTI/lm2d_68_multiview_006167/', \
                 'sample_list':'/nfs/STG/CodecAvatar/lelechen/libingzeng/DAD-3DHeadsDataset/train/PTI/lm2d_68_multiview_006167/train_list.npy', \
                 'sample_num': 4200, \
                 'img_size':120, \
                 'loader_name':'train', \
                 }

    cfg_val = {'dataset_multiview_root':'/nfs/STG/CodecAvatar/lelechen/DAD-3DHeadsDataset/train/PTI/multiview', \
                 'dataset_multiview_lm2d_root':'/nfs/STG/CodecAvatar/lelechen/libingzeng/DAD-3DHeadsDataset/train/PTI/lm2d_68_multiview_006167/', \
                 'sample_list':'/nfs/STG/CodecAvatar/lelechen/libingzeng/DAD-3DHeadsDataset/train/PTI/lm2d_68_multiview_006167/val_list.npy', \
                 'sample_num': 500, \
                 'img_size':120, \
                 'loader_name':'val', \
                 }

    train_dataset = DDFADatasetBooster(
        root=args.root,
        filelists=args.filelists_train,
        param_fp=args.param_fp_train,
        config=cfg_train,
        transform=transforms.Compose([ToTensorGjz(), normalize])
    )
    val_dataset = DDFADatasetBooster(
        root=args.root,
        filelists=args.filelists_val,
        param_fp=args.param_fp_val,
        config=cfg_val,
        transform=transforms.Compose([ToTensorGjz(), normalize])
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                              shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=args.workers,
                            shuffle=False, pin_memory=True)

    # step4: run
    cudnn.benchmark = True
    if args.test_initial:
        logging.info('Testing from initial')
        validate(val_loader, model, criterion, args.start_epoch)

    config = {'booster_weight': args.bw}
    for epoch in range(args.start_epoch, args.epochs + 1):
        # adjust learning rate
        adjust_learning_rate(optimizer, epoch, args.milestones)
        

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, config=config)
        booster_weight = config['booster_weight']
        filename = f'{args.snapshot}_checkpoint_bw_{booster_weight}_epoch_{epoch}.pth.tar'
        save_checkpoint(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                # 'optimizer': optimizer.state_dict()
            },
            filename
        )

        # validate(val_loader, model, criterion, epoch)


if __name__ == '__main__':
    main()
