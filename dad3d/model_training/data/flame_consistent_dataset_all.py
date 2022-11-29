from math import floor
import os
import json
import numpy as np
import random
import pdb
import time

import torch
import torch.utils.data as data
from typing import Dict, Any, List, Union, Tuple, Optional
import albumentations as A
import cv2
from pytorch_toolbelt.utils import read_rgb_image
import pytorch_toolbelt.utils as pt_utils
from model_training.data.transforms import get_resize_fn, get_normalize_fn
from model_training.data.utils import ensure_bbox_boundaries, extend_bbox, read_as_rgb, get_68_landmarks
from hydra.utils import instantiate


from collections import namedtuple
from model_training.data.config import (
    IMAGE_FILENAME_KEY,
    SAMPLE_INDEX_KEY,
    INPUT_IMAGE_KEY,
    INPUT_BBOX_KEY,
    INPUT_SIZE_KEY,
    TARGET_PROJECTION_MATRIX,
    TARGET_3D_MODEL_VERTICES,
    TARGET_3D_WORLD_VERTICES,
    TARGET_2D_LANDMARKS,
    TARGET_LANDMARKS_HEATMAP,
    TARGET_2D_FULL_LANDMARKS,
    TARGET_2D_LANDMARKS_PRESENCE,
)
from model_training.utils import load_2d_indices

MeshArrays = namedtuple(
    "MeshArrays",
    ["vertices3d", "vertices3d_world_homo", "projection_matrix"],
)

def collate_skip_none(batch: Any) -> Any:    
    len_batch = len(batch)
    batch = list(filter(lambda x: x is not None, batch))
    if len_batch > len(batch):
        diff = len_batch - len(batch)
        batch = batch + batch[:diff]
    return torch.utils.data.dataloader.default_collate(batch)

class FlameConsistentDatasetAll(data.Dataset):
    
    def __init__(self, config, transform=None):
        self.root_dir = config["dataset_root"]
        self.multiview_dir = config["dataset_multiview_root"]
        self.multiview_lm2d_dir = config["dataset_multiview_lm2d_root"]
        self.scenes = np.load(config["sample_list"]).tolist()[:config['sample_num']]
        self.config = config
        self.transform = transform
        self.img_size = config["img_size"]
        self.filename_key = "img_path"
        self.aug_pipeline = self._get_aug_pipeline(config["transform"])
        self.num_classes = config.get("num_classes")
        self.keypoints_indices = load_2d_indices(config["keypoints"])
        self.tensor_keys = [INPUT_IMAGE_KEY]
        self.coder = instantiate(config["coder"], config, self.num_classes)

        self.cam_ext = np.load('/nfs/STG/CodecAvatar/lelechen/libingzeng/EG3D_Inversion/eg3d/out/out/outputs_views_ortho_mouth_cam/cam2world_pose.npy')

        if self.config['data_loading_once']:
            st_time = time.time()
            self.scenes_multiviews = {}
            for i in range(config['sample_num']):
                scene_name = self.scenes[i]
                scene_multiview_dir = os.path.join(self.multiview_dir, '{}.mp4'.format(scene_name))
                images = []
                cap = cv2.VideoCapture(scene_multiview_dir)
                success,image = cap.read()
                while success:
                    success,image = cap.read()
                    images.append(image)
                self.scenes_multiviews[scene_name] = images
            ed_time = time.time()
            print('loading {} sample videos costs {}'.format(config['sample_num'], ed_time-st_time))

        with open(self.config['dad3d_data_json']) as json_file: 
            self.dad3d_data_list = json.load(json_file)
        self.dad3d_data_list_len = len(self.dad3d_data_list)
        self.dad3d_data_samples_num = self.config['dad3d_data_samples_num']
         
    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        # return cls(root=config["dataset_root"])
        return cls(config=config)

    @staticmethod
    def _project_vertices_onto_image(
            vertices3d_world_homo: np.ndarray,
            projection_matrix: np.ndarray,
            height: int,
            crop_point_x: int,
            crop_point_y: int
    ):
        vertices2d_homo = np.transpose(np.matmul(projection_matrix, np.transpose(vertices3d_world_homo)))
        vertices2d = vertices2d_homo[:, :2] / vertices2d_homo[:, [3]]
        vertices2d = np.stack((vertices2d[:, 0], (height - vertices2d[:, 1])), -1)
        vertices2d -= (crop_point_x, crop_point_y)
        return vertices2d

    @staticmethod
    def _load_mesh(mesh_path: str) -> MeshArrays:
        with open(mesh_path) as json_data:
            data = json.load(json_data)        
        flame_vertices3d = np.array(data["vertices"], dtype=np.float32)
        model_view_matrix = np.array(data["model_view_matrix"], dtype=np.float32)
        flame_vertices3d_homo = np.concatenate((flame_vertices3d, np.ones_like(flame_vertices3d[:, [0]])), -1)
        # rotated and translated (to world coordinates)
        flame_vertices3d_world_homo = np.transpose(np.matmul(model_view_matrix, np.transpose(flame_vertices3d_homo)))
        return MeshArrays(
            vertices3d=flame_vertices3d,
            vertices3d_world_homo=flame_vertices3d_world_homo,  # with pose and translation
            projection_matrix=np.array(data["projection_matrix"], dtype=np.float32),
        )

    def __len__(self):
        return len(self.scenes)

    def _transform2(self, x: np.ndarray) -> np.ndarray:
        aug = A.Compose(
            [
                A.LongestMaxSize(self._img_size, always_apply=True),
                A.PadIfNeeded(self._img_size, self._img_size, border_mode=cv2.BORDER_CONSTANT, always_apply=True),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return aug(image=x)["image"]


    def __getitem__(self, idx):
        scene_name = self.scenes[idx]

        (
            flame_vertices3d,
            flame_vertices3d_world_homo,
            projection_matrix,
        ) = self._load_mesh(os.path.join(self.config["ann_path"], '{}.json'.format(scene_name)))

        if self.config['data_loading_once']:
            scene_multiviews = self.scenes_multiviews[scene_name]
        else:
            scene_multiview_dir = os.path.join(self.multiview_dir, '{}.mp4'.format(scene_name))
            scene_multiviews = []
            cap = cv2.VideoCapture(scene_multiview_dir)
            success,image = cap.read()
            assert success, "reading mp4 of {} fails".format(scene_name)
            while success:
                success,image = cap.read()
                scene_multiviews.append(image)
        
        # scene_multiview_lm2d_dir = os.path.join(self.multiview_lm2d_dir, 'npy', '{}_lm2d_multiviews.npy'.format(scene_name))
        scene_multiview_lm2d_dir = os.path.join(self.multiview_lm2d_dir, 'lm2d_68_multiview_2d', '{}_lm2d_68_multiview_2d.npy'.format(scene_name))
        scene_multiview_lm2d = np.load(scene_multiview_lm2d_dir)
        assert scene_multiview_lm2d.shape[0] == 512 and scene_multiview_lm2d.shape[1] == 68 and scene_multiview_lm2d.shape[2] == 2, 'scene_multiview_lm2d : {:}'.format(KRT.shape)

        sample_num = 32
        if self.config['loader_name'] == 'val':
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
            im = scene_multiviews[v]
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB, dst=im)
            view_name = 'view_{}'.format(str(v).zfill(3))
            self._img_size = self.config['img_size']
            im = self._transform2(im)
            im = torch.tensor(im).permute(2, 0, 1)
            K = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]]) # * torch.tensor([[256],[256], [1]])
            RT = torch.inverse(torch.tensor(self.cam_ext[v]))[:3, :]   # Notice: KRT should use world2cam matrix not cam2world matrix
            KRT = torch.mm(K, RT)
            
            flame_mesh = torch.tensor(flame_vertices3d) # 5023x3
            
            # lm2d_view_dir = os.path.join(lm2d_folder, 'frame_{}_lm2d_68.npy'.format(str(v).zfill(3)))
            lm2d_points = torch.tensor(scene_multiview_lm2d[v]).float() / 512.0
            # lm2d_points = torch.tensor(np.load(lm2d_view_dir)).float() / 512.0

            names.append(scene_name + '_' + view_name)
            # view_names.append(view_name)
            views.append(im)
            KRTs.append(KRT)
            
            flame_meshes.append(flame_mesh)
            lm2d_points_gt.append(lm2d_points)

        if self.config['single_view_gt']:
            im = read_rgb_image(os.path.join(self.config['single_view_img_gt_path'], '{}.png'.format(scene_name)))
            im = self._transform2(im)
            im = torch.tensor(im).permute(2, 0, 1)
            K = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]]) # * torch.tensor([[256],[256], [1]])
            cam_gt_path = os.path.join(self.config['single_view_cam_gt_path'], '{}.npy'.format(scene_name))
            RT = torch.inverse(torch.tensor(np.load(cam_gt_path)[:16].reshape(4, 4))).float()[:3, :]   # Notice: KRT should use world2cam matrix not cam2world matrix
            KRT = torch.mm(K, RT)
            
            flame_mesh = torch.tensor(flame_vertices3d) # 5023x3
            
            lm2d_gt_path = os.path.join(self.config['single_view_lm2d_gt_path'], '{}__cropped.npy'.format(scene_name))
            lm2d_points = torch.tensor(np.load(lm2d_gt_path)).float() / 512.0

            names.append(scene_name + '_gt')
            views.append(im)
            KRTs.append(KRT)
            
            flame_meshes.append(flame_mesh)
            lm2d_points_gt.append(lm2d_points)

        views = torch.stack(views, dim=0) # Vx3xHxW
        KRTs = torch.stack(KRTs, dim=0) # Vx3x4
        flame_meshes = torch.stack(flame_meshes, dim=0) # Vx5023x3
        lm2d_points_gt = torch.stack(lm2d_points_gt, dim=0) # Vx68x2
        
        dad3d_data_samples_num = self.dad3d_data_samples_num #3#22
        dad3d_data_samples_list = random.sample(range(self.dad3d_data_list_len-1), dad3d_data_samples_num)
        dad3d_data_items = []
        for i in dad3d_data_samples_list:
            item_anno = self.dad3d_data_list[i]
            item_data = self._parse_anno(item_anno)
            item_data = self._transform(item_data)
            item_dict = self._form_anno_dict(item_data)
            item_dict = self._add_index(i, item_anno, item_dict)
            item_dict = self._convert_images_to_tensors(item_dict)
            dad3d_data_items.append(item_dict)
        
        

        batch = {'names':names, 'views':views, 'KRTs':KRTs, 'flame_meshes':flame_meshes, 'lm2d_points_gt':lm2d_points_gt, 'dad3d_data_items':dad3d_data_items}

        return batch

    def _add_index(self, idx: int, annotation: Any, item_dict: Dict[str, Any]) -> Dict[str, Any]:
        if item_dict is not None:
            item_dict.update({SAMPLE_INDEX_KEY: idx, IMAGE_FILENAME_KEY: annotation[self.filename_key]})
        return item_dict

    def _get_item_anno(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

    def _convert_images_to_tensors(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        if item_data is not None:
            for key, item in item_data.items():
                if isinstance(item, np.ndarray) and key in self.tensor_keys:
                    item_data[key] = pt_utils.image_to_tensor(item.astype("float32"))
        return item_data

    def _parse_anno(self, item_anno: Dict[str, Any]) -> Dict[str, Any]:
        img = read_as_rgb(os.path.join(self.config["dataset_root"], item_anno["img_path"]))
        bbox = item_anno["bbox"]
        offset = tuple(0.1 * np.random.uniform(size=4) + 0.05)
        x, y, w, h = ensure_bbox_boundaries(extend_bbox(np.array(bbox), offset), img.shape[:2])
        cropped_img = img[y : y + h, x : x + w]
        (
            flame_vertices3d,
            flame_vertices3d_world_homo,
            projection_matrix,
        ) = self._load_mesh(os.path.join(self.config["dataset_root"], item_anno["annotation_path"]))
        return {
            INPUT_IMAGE_KEY: cropped_img,
            INPUT_BBOX_KEY: (x, y, w, h),
            INPUT_SIZE_KEY: img.shape,
            TARGET_3D_MODEL_VERTICES: flame_vertices3d,
            TARGET_3D_WORLD_VERTICES: flame_vertices3d_world_homo,
            TARGET_PROJECTION_MATRIX: projection_matrix
        }

    def _get_2d_landmarks_w_presence(
        self,
        vertices3d_world_homo: np.ndarray,
        projection_matrix: np.ndarray,
        img_shape: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if self.num_classes == 68:
            landmarks_3d_world_subset = get_68_landmarks(
                torch.from_numpy(vertices3d_world_homo[..., :3]).view(-1, 3)
            ).numpy()
            landmarks_3d_world_subset = np.concatenate(
                (landmarks_3d_world_subset, np.ones_like(landmarks_3d_world_subset[:, [0]])), -1
            )
        else:
            landmarks_3d_world_subset = vertices3d_world_homo[self.keypoints_indices]
        x, y, w, h = bbox

        landmarks_2d_subset = self._project_vertices_onto_image(
            landmarks_3d_world_subset, projection_matrix, img_shape[0], x, y
        )
        keypoints_2d = self._project_vertices_onto_image(vertices3d_world_homo, projection_matrix, img_shape[0], x, y)

        presence_subset = np.array([False] * len(landmarks_2d_subset))

        for i in range(len(landmarks_2d_subset)):
            if 0 < landmarks_2d_subset[i, 0] < w and 0 < landmarks_2d_subset[i, 1] < h:
                presence_subset[i] = True

        return landmarks_2d_subset, presence_subset, keypoints_2d

    def _transform(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        vertices_2d_subset, presence_subset, vertices_2d = self._get_2d_landmarks_w_presence(
            item_data[TARGET_3D_WORLD_VERTICES],
            item_data[TARGET_PROJECTION_MATRIX],
            item_data[INPUT_SIZE_KEY],
            item_data[INPUT_BBOX_KEY],
        )

        result = self.aug_pipeline(
            image=item_data[INPUT_IMAGE_KEY], keypoints=np.concatenate((vertices_2d_subset, vertices_2d), 0)
        )

        return {
            INPUT_IMAGE_KEY: result["image"],
            INPUT_BBOX_KEY: item_data[INPUT_BBOX_KEY],
            TARGET_3D_MODEL_VERTICES: item_data[TARGET_3D_MODEL_VERTICES],
            TARGET_2D_LANDMARKS: np.array(result["keypoints"][: self.num_classes], dtype=np.float32),
            TARGET_2D_FULL_LANDMARKS: np.array(result["keypoints"][self.num_classes :], dtype=np.float32),
            TARGET_2D_LANDMARKS_PRESENCE: presence_subset
        }

    def _form_anno_dict(self, item_data: Dict[str, np.ndarray]) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        landmarks = item_data[TARGET_2D_LANDMARKS]
        presence = item_data[TARGET_2D_LANDMARKS_PRESENCE]
        heatmap = self.coder(landmarks, presence)
        item_data[TARGET_2D_LANDMARKS] = landmarks / self.img_size
        item_data[TARGET_LANDMARKS_HEATMAP] = np.uint8(255.0 * heatmap)
        return item_data

    def _get_aug_pipeline(self, aug_config: Dict[str, Any]) -> A.Compose:
        normalize = get_normalize_fn(aug_config.get("normalize", "imagenet"))
        resize = get_resize_fn(self.img_size, mode=aug_config.get("resize_mode", "longest_max_size"))
        return A.Compose(
            [resize, normalize],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False)
        )

    def get_collate_fn(self) -> Any:
        return collate_skip_none
        