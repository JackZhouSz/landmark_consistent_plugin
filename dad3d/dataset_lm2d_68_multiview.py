from functools import partial
from collections import namedtuple
from json.encoder import py_encode_basestring
import os
import json
import cv2
import pdb
import numpy as np
import torch
import pickle
import time

from fire import Fire
from pytorch_toolbelt.utils import read_rgb_image
from ray_sampler import RaySampler

from predictor import FaceMeshPredictor
from demo_utils import (
    draw_landmarks,
    draw_3d_landmarks,
    draw_mesh,
    draw_pose,
    get_mesh,
    get_flame_params,
    get_output_path,
    MeshSaver,
    ImageSaver,
    JsonSaver,
)

DemoFuncs = namedtuple(
    "DemoFuncs",
    ["processor", "saver"],
)

demo_funcs = {
    "68_landmarks": DemoFuncs(draw_landmarks, ImageSaver),
    "247_landmarks": DemoFuncs(partial(draw_3d_landmarks, subset="247"), ImageSaver),
    "191_landmarks": DemoFuncs(partial(draw_3d_landmarks, subset="191"), ImageSaver),
    "445_landmarks": DemoFuncs(partial(draw_3d_landmarks, subset="445"), ImageSaver),
    "head_mesh": DemoFuncs(partial(draw_mesh, subset="head"), ImageSaver),
    "face_mesh": DemoFuncs(partial(draw_mesh, subset="face"), ImageSaver),
    "pose": DemoFuncs(draw_pose, ImageSaver),
    "3d_mesh": DemoFuncs(get_mesh, MeshSaver),
    "flame_params": DemoFuncs(get_flame_params, JsonSaver)
}

# one KRT and 3D-points
# KRT : 3x4 ; PTS3D : Nx3
def ProjectKRT(KRT, PTS3D):
    assert KRT.dim() == 2 and KRT.size(0) == 3 and KRT.size(1) == 4, 'KRT : {:}'.format(KRT.shape)
    assert PTS3D.dim() == 2 and PTS3D.size(-1) == 3, 'PTS3D : {:}'.format(PTS3D.shape)
    MPs = torch.matmul(KRT[:,:3], PTS3D.transpose(1,0)) + KRT[:, 3:]
    X = MPs[0] / MPs[2]
    Y = MPs[1] / MPs[2]
    return torch.stack((X,Y), dim=1)

# a list of KRTs and a list of points for different cameras
# KRTs   : N * 3 * 4
# points : N * 2 
def TriangulateDLT(KRTs, points):
    assert KRTs.dim() == 3 and points.dim() == 2 and KRTs.size(0) == points.size(0), 'KRTs : {:}, points : {:}'.format(KRTs.shape, points.shape)
    assert KRTs.size(1) == 3 and KRTs.size(2) == 4 and points.size(-1) == 2, 'KRTs : {:}, points : {:}'.format(KRTs.shape, points.shape)
    U = KRTs[:,0,:] - KRTs[:,2,:] * points[:,0].view(-1, 1)
    V = KRTs[:,1,:] - KRTs[:,2,:] * points[:,1].view(-1, 1)
    Dmatrix = torch.cat((U,V))
    A   = Dmatrix[:,:3]
    At  = torch.transpose(A, 0, 1)
    AtA = torch.mm(At, A)
    invAtA = torch.inverse( AtA )
    P3D = torch.mm(invAtA, torch.mm(At, -Dmatrix[:,3:]))
    return P3D.view(-1, 3)


# http://cmp.felk.cvut.cz/cmp/courses/TDV/2012W/lectures/tdv-2012-07-anot.pdf
# a multiview system has N cameras with N KRTs, [N * P * 2] points
# KRTs   : N * 3 * 4
# points : N * P * 2
def TriangulateDLT_BatchPoints(KRTs, points):
  assert KRTs.dim() == 3 and points.dim() == 3 and KRTs.size(0) == points.size(0), 'KRTs : {:}, points : {:}'.format(KRTs.shape, points.shape)
  assert KRTs.size(1) == 3 and KRTs.size(2) == 4 and points.size(-1) == 2, 'KRTs : {:}, points : {:}'.format(KRTs.shape, points.shape)
  assert points.size(1) >= 3, 'There should be at least 3 points'.format(points.shape)
  KRTs = KRTs.view(KRTs.size(0), 1, 3, 4)             # size = N * 1 * 3 * 4
  U = KRTs[:,:,0,:] - KRTs[:,:,2,:] * points[...,0:1] # size = N * P * 4
  V = KRTs[:,:,1,:] - KRTs[:,:,2,:] * points[...,1:2] 
  Dmatrix = torch.cat((U,V), dim=0).transpose(1,0)    # size = P * 2N * 4
  A      = Dmatrix[:,:,:3]
  At     = torch.transpose(A, 2, 1)
  AtA    = torch.matmul(At, A)
  invAtA = torch.inverse( AtA )
  P3D    = torch.matmul(invAtA, torch.matmul(At, -Dmatrix[:,:,3:]))
  return P3D.view(-1, 3)


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


def to_camera(points_world, extrinsics):
    """map points in world space to camera space
    Args:
        points_world (B, 3, H, W): points in world space.
        extrinsics (B, 3, 4): [R, t] of camera.
    Returns:
        points_camera (B, 3, H, W): points in camera space.
    """
    B, p_dim, H, W = points_world.shape
    assert p_dim == 3, "dimension of point {} != 3".format(p_dim)
    
    # map to camera:
    # R'^T * (p - t') where t' of (B, 3, 1), R' of (B, 3, 3) and p of (B, 3, H*W)
    R_tgt = extrinsics[..., :p_dim]
    t_tgt = extrinsics[..., -1:]
    points_cam = torch.bmm(R_tgt.transpose(1, 2), points_world.view(B, p_dim, -1) - t_tgt)

    return points_cam.view(B, p_dim, H, W)


def point_to_image(point_world, intrinsics, extrinsics, resolution):
    """map point in world space to image pixel
    Args:
        point_world (1, 3): point in world space.
        intrinsics (3, 3): camera intrinsics of target camera
        extrinsics (3, 4): camera extrinsics of target camera
        resolution (1): scalar, image resolution.
    Returns:
        pixel_image2 (2): pixel coordinates on target image plane.
    """
    K = intrinsics
    RT = extrinsics
    points_cam2 = to_camera(point_world.unsqueeze(-1).unsqueeze(-1), RT.unsqueeze(0))[:, :, 0, 0]
    points_image2 = ((points_cam2 / points_cam2[:, 2]) * K[0, 0] + K[0, 2]) * resolution
    pixel_image2 = points_image2[0, :2]
    return pixel_image2


def points_to_image(points_world, intrinsics, extrinsics, resolution):
    """map points in world space to image pixel
    Args:
        points_world (B, 3): points in world space.
        intrinsics (B, 3, 3): camera intrinsics of target camera
        extrinsics (B, 3, 4): camera extrinsics of target camera
        resolution (1): scalar, image resolution.
    Returns:
        pixel_image2 (2): pixel coordinates on target image plane.
    """
    K = intrinsics
    RT = extrinsics
    points_cam2 = to_camera(points_world.unsqueeze(-1).unsqueeze(-1), RT.unsqueeze(0).repeat(68, 1, 1))[:, :, 0, 0]
    points_image2 = ((points_cam2 / points_cam2[:, -1:]) * K[0, 0] + K[0, 2]) * resolution
    pixels_image2 = points_image2[:, :2]
    return pixels_image2


def lm2d_eye_center(lm2d):
    lm_eye_left      = lm2d[36 : 42]  # left-clockwise
    lm_eye_right     = lm2d[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm2d[48 : 60]  # left-clockwise
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    return eye_avg


def lm2d_68_multiview_projection_dad3d_dataset():
    '''
    generate pseudo ground truth 2D landmarks for each view
    using 2D landmarks estimated by DAD-3D for anchor frame, #390,
    and EG3D depth map of the anchor frame
    '''
    multiview_path = '/nfs/STG/CodecAvatar/lelechen/DAD-3DHeadsDataset/train/PTI/multiview'
    anchor_depth_path = '/nfs/STG/CodecAvatar/lelechen/DAD-3DHeadsDataset/train/PTI/depth'
    cam_pose_path = '/nfs/STG/CodecAvatar/lelechen/DAD-3DHeadsDataset/train/PTI/cam2world_pose.pt'
    sample_list_path = '/nfs/STG/CodecAvatar/lelechen/libingzeng/DAD-3DHeadsDataset/train/PTI/goodviews_depths_list_006167.npy'
    cam_extrinsics_list = torch.load(cam_pose_path)
    cam_intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]])
    ref_lmark_path = '/nfs/STG/CodecAvatar/lelechen/DAD-3DHeadsDataset/train/lmark'
    ref_cropped_image_path = '/nfs/STG/CodecAvatar/lelechen/DAD-3DHeadsDataset/train/cropped_img'
    
    type_of_lm2d_output: str = "68_landmarks"
    predictor = FaceMeshPredictor.dad_3dnet()

    sample_list = np.load(sample_list_path)
    lm2d_68_multiview_path = '/nfs/STG/CodecAvatar/lelechen/libingzeng/DAD-3DHeadsDataset/train/PTI/lm2d_68_multiview_{}'.format(str(len(sample_list)).zfill(6))
    output_jpg_path = os.path.join(lm2d_68_multiview_path, 'jpg')
    output_anchor_points_2d_path = os.path.join(lm2d_68_multiview_path, 'lm2d_68_multiview_2d')
    output_anchor_points_3d_path = os.path.join(lm2d_68_multiview_path, 'lm2d_68_multiview_3d')
    os.makedirs(lm2d_68_multiview_path, exist_ok=True)
    os.makedirs(output_jpg_path, exist_ok=True)
    os.makedirs(output_anchor_points_2d_path, exist_ok=True)
    os.makedirs(output_anchor_points_3d_path, exist_ok=True)
    sample_cnt = 0
    # for i, im_path in enumerate(sample_list):
    idx_st = 1477
    for i in range(idx_st, len(sample_list)):
        im_path = sample_list[i]
        sample_cnt = i
        st_time = time.time()
        scene_name = im_path.split('.')[0]
        scene_jpg_path = os.path.join(output_jpg_path, scene_name)
        scene_anchor_points_2d_path = os.path.join(output_anchor_points_2d_path, scene_name+'_lm2d_68_multiview_2d.npy')
        scene_anchor_path = os.path.join(output_anchor_points_3d_path, scene_name+'_lm2d_68_multiview_3d.npy')
        scene_multiview_path = os.path.join(multiview_path, scene_name+'.mp4')
        cap = cv2.VideoCapture(scene_multiview_path)
        amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        anchor_frame_num = 390
        cap.set(cv2.CAP_PROP_POS_FRAMES, anchor_frame_num)
        anchor_ret, anchor_frame_ = cap.read()
        lm_3d = None
        if anchor_ret:
            anchor_frame = cv2.cvtColor(anchor_frame_, cv2.COLOR_BGR2RGB)
            predictions = predictor(anchor_frame)
            lm2d = predictions['points']
            # for j in range(lm2d.shape[0]): cv2.circle(anchor_frame, (int(lm2d[j][0]), int(lm2d[j][1])), 3, (0, 255, 0), -1)

            depth_dir = os.path.join(anchor_depth_path, scene_name+'.npy')
            depth = np.transpose(np.load(depth_dir), (1, 2, 0))
            res_scale = anchor_frame.shape[0] / depth.shape[0]
            ray_sampler = RaySampler()
            ray_origins, ray_directions = ray_sampler(cam_extrinsics_list[anchor_frame_num].cpu(), cam_intrinsics.unsqueeze(0).cpu(), depth.shape[0])
            
            origin = ray_origins[:, 0, :]
            directions = ray_directions[0].reshape(depth.shape[0], depth.shape[1], 3)
            lm2d_depth_coords = (lm2d / res_scale).astype(int) # 512 -> 128
            lm2d_depth_coords_x, lm2d_depth_coords_y = lm2d_depth_coords[:, 0], lm2d_depth_coords[:, 1]
            lm2d_depth = depth[lm2d_depth_coords_y, lm2d_depth_coords_x]
            lm_3d = origin + directions[lm2d_depth_coords_y, lm2d_depth_coords_x] * lm2d_depth
            np.save(scene_anchor_path, lm_3d.numpy())
        else:
            continue

        anchor_points_projs, anchor_points_projs_srt, KRTs = [], [], []
        test_frame_list = [0, 125, 250, 375, anchor_frame_num]
        checking_interval = 100
        for f in range(int(amount_of_frames)):
            # anchor_points_proj = points_to_image(lm_3d, cam_intrinsics.cpu(), cam_extrinsics_list[f][0][:3, :].cpu(), anchor_frame.shape[0])[:, [1, 0]] # output (y, x)
            anchor_points_proj = points_to_image(lm_3d, cam_intrinsics.cpu(), cam_extrinsics_list[f][0][:3, :].cpu(), anchor_frame.shape[0]) # output (x, y) which is important and along with KRT
            anchor_points_projs.append(anchor_points_proj)
            
            # check the accuracy of 2d anchor points each checking_interval scenes.
            if sample_cnt % checking_interval == 0:
                os.makedirs(scene_jpg_path, exist_ok=True)
                if f in test_frame_list:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                    ret, frame_ = cap.read()
                    view_img = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
                    predictions = predictor(view_img)
                    for l in range(anchor_points_proj.shape[0]):cv2.circle(view_img, (int(anchor_points_proj[l][0]), int(anchor_points_proj[l][1])), 2, (0, 0, 0), -1)    
                    result = demo_funcs[type_of_lm2d_output].processor(predictions, view_img)
                    saver = demo_funcs[type_of_lm2d_output].saver()  # instantiate the Saver
                    output_path = os.path.join(scene_jpg_path, 'frame_{}_dad3d_lm2d.jpg'.format(str(f).zfill(3)))
                    saver(result, output_path)

                    K = cam_intrinsics.cpu() * torch.tensor([[view_img.shape[0]],[view_img.shape[1]], [1]])
                    RT = torch.inverse(cam_extrinsics_list[f][0])[:3, :].cpu()  # Notice: KRT should use world2cam matrix not cam2world matrix
                    KRT = torch.mm(K, RT)                
                    KRTs.append(KRT)
                    anchor_points_projs_srt.append(anchor_points_proj)

        # save 2d anchor points of 512 views
        anchor_points_projs = torch.stack(anchor_points_projs, dim=0)
        np.save(scene_anchor_points_2d_path, anchor_points_projs)
        
        # check the accuracy of 2d anchor points each checking_interval scenes.
        if sample_cnt % checking_interval == 0:
            # check the accuracy of anchor_point_projs by
            # estimating 3d point using anchor_point_projs
            # and then projecting the 3d point to each view of [0, 125, 250, 375, anchor_frame_num]
            KRTs = torch.stack(KRTs, dim=0)
            anchor_points_projs_srt = torch.stack(anchor_points_projs_srt, dim=0)
            # anchor_points_gt_srt = TriangulateDLT_BatchPoints(KRTs, anchor_points_projs_srt)
            anchor_points_gt_srt = TriangulateDLT_BatchCam(KRTs.unsqueeze(0), anchor_points_projs_srt.unsqueeze(0))
            anchor_points_projs_srt_projs = []

            for i in range(KRTs.shape[0]):
                anchor_points_proj_srt = ProjectKRT_Batch(KRTs[i].unsqueeze(0), anchor_points_gt_srt)[0]
                anchor_points_projs_srt_projs.append(anchor_points_proj_srt)
            
            for i in range(len(test_frame_list)):
                output_lm2d_dir = os.path.join(scene_jpg_path, 'frame_{}_dad3d_lm2d.jpg'.format(str(test_frame_list[i]).zfill(3)))
                view_img = cv2.imread(output_lm2d_dir)
                for lp in range(anchor_points_proj_srt.shape[0]):cv2.circle(view_img, (int(anchor_points_projs_srt_projs[i][lp][0]), int(anchor_points_projs_srt_projs[i][lp][1])), 1, (255, 255, 255), -1)    
                output_lm2d_anchor_srt_proj_dir = output_lm2d_dir.replace('.jpg', '_anchor-srt-proj-white_dad3d-green_3d-proj-black.jpg')
                cv2.imwrite(output_lm2d_anchor_srt_proj_dir, view_img)
        
        # draw reference lm2d on reference image
        ref_image_ = cv2.imread(os.path.join(ref_cropped_image_path, '{}.png'.format(scene_name)))
        ref_image = cv2.cvtColor(ref_image_, cv2.COLOR_BGR2RGB)
        ref_lmark = np.load(os.path.join(ref_lmark_path, '{}__cropped.npy'.format(scene_name)))
        for rl in range(ref_lmark.shape[0]): cv2.circle(ref_image, (int(ref_lmark[rl][0]), int(ref_lmark[rl][1])), 2, (0, 0, 0), -1)    
        ref_output_path = os.path.join(scene_jpg_path, 'ref_image_ref_lmark.jpg')
        cv2.imwrite(ref_output_path, cv2.cvtColor(ref_image, cv2.COLOR_RGB2BGR))
        
        ed_time = time.time()
        sample_cnt += 1
        print('='*10, 'sample_cnt:{}'.format(str(sample_cnt).zfill(6)), '='*10, 'cost time:{}'.format(ed_time - st_time))
        
if __name__ == "__main__":
    Fire(lm2d_68_multiview_projection_dad3d_dataset)