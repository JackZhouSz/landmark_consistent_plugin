import json
from typing import Dict, List
import numpy as np
from fire import Fire
import cv2
import os
from demo_utils import draw_points, get_output_path
from model_training.utils import load_indices_from_npy
from smplx.utils import Struct
import pickle
import pdb
from typing import Dict, Any, List, Union, Tuple, Optional
from model_training.data.utils import ensure_bbox_boundaries, extend_bbox, read_as_rgb, get_68_landmarks
import torch

def get_2d_keypoints(data: Dict[str, List], img_height: int) -> np.ndarray:
    flame_vertices3d = np.array(data["vertices"], dtype=np.float32)
    model_view_matrix = np.array(data["model_view_matrix"], dtype=np.float32)
    projection_matrix = np.array(data["projection_matrix"], dtype=np.float32)
    flame_vertices3d_homo = np.concatenate((flame_vertices3d, np.ones_like(flame_vertices3d[:, [0]])), -1)
    flame_vertices3d_world_homo = np.transpose(np.matmul(model_view_matrix, np.transpose(flame_vertices3d_homo)))

    flame_vertices2d_homo = np.transpose(
        np.matmul(projection_matrix, np.transpose(flame_vertices3d_world_homo))
    )
    flame_vertices2d = flame_vertices2d_homo[:, :2] / flame_vertices2d_homo[:, [3]]
    return np.stack((flame_vertices2d[:, 0], (img_height - flame_vertices2d[:, 1])), -1).astype(int)

def get_2d_keypoints_68(data: Dict[str, List], img_height: int) -> np.ndarray:
    flame_vertices3d = np.array(data["vertices"], dtype=np.float32)
    model_view_matrix = np.array(data["model_view_matrix"], dtype=np.float32)
    projection_matrix = np.array(data["projection_matrix"], dtype=np.float32)
    flame_vertices3d_homo = np.concatenate((flame_vertices3d, np.ones_like(flame_vertices3d[:, [0]])), -1)
    flame_vertices3d_world_homo = np.transpose(np.matmul(model_view_matrix, np.transpose(flame_vertices3d_homo)))
    
    flame_vertices3d_subset = get_68_landmarks( # lbz added to extract lm3d
        torch.from_numpy(flame_vertices3d_homo[..., :3]).view(-1, 3)
    ).numpy()

    landmarks_3d_world_subset = get_68_landmarks(
        torch.from_numpy(flame_vertices3d_world_homo[..., :3]).view(-1, 3)
    ).numpy()
    landmarks_3d_world_subset_homo = np.concatenate(
        (landmarks_3d_world_subset, np.ones_like(landmarks_3d_world_subset[:, [0]])), -1
    )

    flame_vertices2d_homo = np.transpose(
        np.matmul(projection_matrix, np.transpose(landmarks_3d_world_subset_homo))
    )
    flame_vertices2d = flame_vertices2d_homo[:, :2] / flame_vertices2d_homo[:, [3]]
    return np.stack((flame_vertices2d[:, 0], (img_height - flame_vertices2d[:, 1])), -1).astype(int), flame_vertices3d_subset, flame_vertices3d
    # return np.stack((flame_vertices2d[:, 0], (img_height - flame_vertices2d[:, 1])), -1).astype(int), landmarks_3d_world_subset

def get_sparse_keypoints(meshpoint, keylist_path):
    indices =[]
    for npyfile in sorted(os.listdir(keylist_path)):
        filename = os.path.join(keylist_path, npyfile)
        if os.path.exists(filename):
            indices += load_indices_from_npy(filename)
        else:
            raise ValueError(f"[{filename.split('.')[0].split('/')[-1]}] class of keypoints doesn't exist")

    sparse_point = meshpoint[indices]
    print(sparse_point.shape)
    return sparse_point

def visualize(
        subset: str,
        id: str,
        base_path: str = 'dataset',
        outputs_folder: str = "outputs_dataset"
) -> None:

    os.makedirs(outputs_folder, exist_ok=True)
    json_path = os.path.join(base_path, 'DAD-3DHeadsDataset', subset, 'annotations', id + '.json')
    img_path = json_path.replace('annotations', 'images').replace('json', 'png')
    print (img_path)
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    with open(json_path) as json_data:
        mesh_data = json.load(json_data)

    keypoints_2d = get_2d_keypoints(mesh_data, img.shape[0])
    print (keypoints_2d.shape)
    img = draw_points(img, keypoints_2d)

    output_filename = get_output_path(img_path, outputs_folder, 'GT_landmarks', '.png')
    cv2.imwrite(output_filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    point = get_sparse_keypoints(keypoints_2d, './model_training/model/static/face_keypoints/keypoints_191')
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = draw_points(img, point)

    output_filename = get_output_path(img_path, outputs_folder, '191', '.png')
    cv2.imwrite(output_filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))



def dataset_lm2d_68_original_extraction(
        subset: str = 'train',
        base_path: str = 'dataset_synthetic/example',
        outputs_folder: str = "dataset_synthetic/example/images_lm2d"
) -> None:

    os.makedirs(outputs_folder, exist_ok=True)
    ids = ['468ab0de-ed66-426d-8a59-4f537efbe8a0']
    for id in ids:
        json_path = os.path.join(base_path, 'annotations', id + '.json')
        img_path = json_path.replace('annotations', 'images').replace('json', 'png')
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        with open(json_path) as json_data:
            mesh_data = json.load(json_data)

        # get ground truth 2d landmarks
        num_classes = 68 # 191, 247, 445
        if num_classes == 68:
            point, point_3d, mesh3d = get_2d_keypoints_68(mesh_data, img.shape[0])
            # np.save(os.path.join('/nfs/STG/CodecAvatar/lelechen/libingzeng/Consistent_Facial_Landmarks/data/dad3d_val/images_lm3d', '{}_lm3d.npy'.format(id)), point_3d)
            # np.save(os.path.join('/nfs/STG/CodecAvatar/lelechen/libingzeng/Consistent_Facial_Landmarks/data/dad3d_val/images_lm3d', '{}_mesh3d.npy'.format(id)), mesh3d)
        else:
            keypoints_2d = get_2d_keypoints(mesh_data, img.shape[0])
            point = get_sparse_keypoints(keypoints_2d, './model_training/model/static/face_keypoints/keypoints_191')

        with open('/home/uss00054/projects/Consistent_Facial_Landmarks/model_training/model/static/flame_static_embedding.pkl', "rb") as f:
            static_embeddings = Struct(**pickle.load(f, encoding="latin1"))

        lmk_faces_idx = static_embeddings.lmk_face_idx.astype(np.int64)

        # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = draw_points(img, point)
        
        lm2d_mask = np.zeros_like(img)
        H, W, _ = lm2d_mask.shape
        for i in range(len(point)):
            lm2d_mask[point[i][1], point[i][0], :] = (i+1)*10
             

        output_filename = get_output_path(img_path, outputs_folder, '{}'.format(num_classes), '.png')
        cv2.imwrite(output_filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # output_lm2d_mask_filename = get_output_path(img_path, outputs_folder, '{}_lm2d_mask'.format(num_classes), '.png')
        # cv2.imwrite(output_lm2d_mask_filename, cv2.cvtColor(lm2d_mask, cv2.COLOR_RGB2BGR))
        
        output_lm2d_points_filename = get_output_path(img_path, outputs_folder, '{}_lm2d_points'.format(num_classes), '.npy')
        with open(output_lm2d_points_filename, 'wb') as f: 
            np.save(f, {'point': point})
        # with open(output_lm2d_points_filename, 'rb') as f: 
        #     points2 = np.load(f, allow_pickle=True).tolist()['point']

        # pdb.set_trace()
        os.system('cp -r {} {}'.format(img_path, outputs_folder))



if __name__ == "__main__":
    Fire(dataset_lm2d_68_original_extraction)
