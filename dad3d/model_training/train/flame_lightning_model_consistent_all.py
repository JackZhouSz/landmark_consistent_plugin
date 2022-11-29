import os
from joblib import cpu_count
import typing
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, ConcatDataset
from torchmetrics import MetricCollection
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.base import DummyLogger

from model_training.data.config import (
    TARGET_2D_LANDMARKS,
    OUTPUT_LANDMARKS_HEATMAP,
    TARGET_LANDMARKS_HEATMAP,
    OUTPUT_3DMM_PARAMS,
    TARGET_3D_MODEL_VERTICES,
    OUTPUT_2D_LANDMARKS,
    TARGET_2D_FULL_LANDMARKS,
    TARGET_2D_LANDMARKS_PRESENCE,
    INPUT_BBOX_KEY,
)
from model_training.model.utils import unravel_index, normalize_to_cube, load_from_lighting
from model_training.head_mesh import HeadMesh
from model_training.metrics.iou import SoftIoUMetric
from model_training.metrics.keypoints import FailureRate, KeypointsNME
from model_training.train.loss_module import LossModule
from model_training.train.mixins import KeypointsDataMixin, KeypointsVisualizationMixin
from model_training.train.optimizers import get_optimizer
from model_training.train.schedulers import get_scheduler
from model_training.train.utils import any2device
from model_training.utils import create_logger
from model_training.train.multiview import triangulate as srt

import pdb
import cv2
import time

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

logger = create_logger(__name__)



def GetKRT(K, R, t, dtype=torch.FloatTensor):
    if isinstance(K, np.ndarray): K = torch.from_numpy(K)
    if isinstance(R, np.ndarray): R = torch.from_numpy(R)
    if isinstance(t, np.ndarray): t = torch.from_numpy(t)
    Rt = torch.cat((R, t), dim=1)
    KRT = torch.mm(K, Rt).type( dtype )
    return KRT


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


# one KRT and 3D-points
# KRT : 3x4 ; PTS3D : Nx3
def ProjectKRT(KRT, PTS3D):
    assert KRT.dim() == 2 and KRT.size(0) == 3 and KRT.size(1) == 4, 'KRT : {:}'.format(KRT.shape)
    assert PTS3D.dim() == 2 and PTS3D.size(-1) == 3, 'PTS3D : {:}'.format(PTS3D.shape)
    MPs = torch.matmul(KRT[:,:3], PTS3D.transpose(1,0)) + KRT[:, 3:]
    X = MPs[0] / MPs[2]
    Y = MPs[1] / MPs[2]
    return torch.stack((X,Y), dim=1)



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


def lm247_to_lm54(lmark):
    '''
    lmark247: BxVx247x2 tensor
    lmark54: BxVx54x2 tensor
    '''
    ear_top = lmark[:, :, [38, 39, 87, 51], :] # 4
    ear_left = lmark[:, :, 58:86, :].mean(dim=-2).unsqueeze(-2)
    ear_right = lmark[:, :, 87:114, :].mean(dim=-2).unsqueeze(-2)
    jaw_left = lmark[:, :, sorted([i for i in range(32, 38, 1)], reverse=True), :] # 6
    jaw_right = lmark[:, :, sorted([i for i in range(53, 58, 1)], reverse=True), :] # 5
    eye_left = lmark[:, :, 114:136, :].mean(dim=-2).unsqueeze(-2) # 1
    eye_right = lmark[:, :, 136:158, :].mean(dim=-2).unsqueeze(-2) # 1
    nose = lmark[:, :, 216:247, :].mean(dim=-2).unsqueeze(-2) # 1
    mouth = lmark[:, :, [176, 204, 206, 235, 208, 210, 214, 185, 183, 182, 181, 179, 165, 193, 194, 195, 188, 171, 170, 169], :] # 20
    mouth_center = lmark[:, :, 165:215, :].mean(dim=-2).unsqueeze(-2) # 1
    brow = lmark[:, :, [8, 10, 12, 14, 15, 31, 30, 28, 26, 24], :] # 10
    brow_left_center = lmark[:, :, 0:16, :].mean(dim=-2).unsqueeze(-2) # 1
    brow_right_center = lmark[:, :, 16:32, :].mean(dim=-2).unsqueeze(-2) # 1
    forehead = lmark[:, :, [i for i in range(40, 51, 1)]+[i for i in range(158, 165, 1)], :].mean(dim=-2).unsqueeze(-2) # 1
    lmark54 = torch.cat([ear_top, ear_left, ear_right, jaw_left, jaw_right, eye_left, eye_right, nose, mouth, mouth_center, brow, brow_left_center, brow_right_center, forehead], dim=-2)
    
    return lmark54

def lm68_to_lm54(lmark):
    '''
    lmark68: BxVx68x2 tensor
    lmark54: BxVx54x2 tensor
    '''
    ear_top = lmark[:, :, [1, 0, 15, 16], :] # 4
    ear_left = lmark[:, :, 2:3, :].mean(dim=-2).unsqueeze(-2)
    ear_right = lmark[:, :, 14:15, :].mean(dim=-2).unsqueeze(-2)
    jaw = lmark[:, :, 3:14, :] # 11
    eye_left = lmark[:, :, 36:42, :].mean(dim=-2).unsqueeze(-2) # 1
    eye_right = lmark[:, :, 42:48, :].mean(dim=-2).unsqueeze(-2) # 1
    nose = lmark[:, :, 27:35, :].mean(dim=-2).unsqueeze(-2)  # 1
    mouth = lmark[:, :, 48:68, :] # 20
    mouth_center = lmark[:, :, 48:68, :].mean(dim=-2).unsqueeze(-2) # 1
    brow = lmark[:, :, 17:27, :] # 10
    brow_left_center = lmark[:, :, 17:22, :].mean(dim=-2).unsqueeze(-2) # 1
    brow_right_center = lmark[:, :, 22:27, :].mean(dim=-2).unsqueeze(-2) # 1
    forehead = lmark[:, :, 21:23, :].mean(dim=-2).unsqueeze(-2) # 1
    lmark54 = torch.cat([ear_top, ear_left, ear_right, jaw, eye_left, eye_right, nose, mouth, mouth_center, brow, brow_left_center, brow_right_center, forehead], dim=-2)

    return lmark54


class FlameLightningModel(pl.LightningModule, KeypointsDataMixin, KeypointsVisualizationMixin):

    _initial_learning_rates: List[List[float]]

    def __init__(self, model: torch.nn.Module, config: Dict[str, Any], train: Dataset, val: Dataset) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.train_dataset = train
        self.val_dataset = val

        self._load_model(self.model)
        self.criterion = self._build_loss(config.get("loss", None))
        self.weight_dad3d = self.config['loss']['weight_dad3d']
        self.our_weight = self.config['loss']['our_weight']
        self.our_proj = self.config['loss']['our_proj']
        self.our_mesh = self.config['loss']['our_mesh']
        self.weight_jaw_pseudo = self.config['loss']['weight_jaw_pseudo']
        self.weight_proj = self.config['loss']['weight_proj'] * self.our_weight * self.our_proj
        self.weight_proj_ref_gt = self.config['loss']['weight_proj_ref_gt'] * self.our_weight * self.our_proj
        self.weight_proj_srt_gt = self.config['loss']['weight_proj_srt_gt'] * self.our_weight * self.our_proj
        self.weight_lm2d = self.config['loss']['weight_lm2d'] * self.our_weight
        self.weight_mesh = self.config['loss']['weight_mesh'] * self.our_weight * self.our_mesh
        self.num_views_srt = self.config['loss']['num_views_srt']
        
        self.use_ddp = self.config.get("accelerator", None) == "ddp"
        self.log_step = config.get("log_step", 1000)
        self.current_step = 0
        self.epoch_num = 0
        self.tensorboard_logger = None
        self.learning_rate: Optional[float] = None

        self.stride = config["train"].get("stride", 2)
        self.images_log_freq = config.get("images_log_freq", 100)
        
        if self.config['single_view_gt']:
            self.out_dir = os.path.join(self.config['output_root'], 'outputs_trains{}_vals{}_single_view_gt{}_lrnew{}_ourweight{}_ourproj{}_ourmesh{}_jawweight{}_dad3d-imgs{}_num-views-srt{}'.format(self.config['train']['sample_num'], self.config['val']['sample_num'], self.config['single_view_gt'], self.config['optimizer']['lr'], self.our_weight, self.our_proj, self.our_mesh, self.weight_jaw_pseudo, self.config['train']['dad3d_data_samples_num'], self.num_views_srt))
        else:
            self.out_dir = os.path.join(self.config['output_root'], 'outputs_trains{}_vals{}'.format(self.config['train']['sample_num'], self.config['val']['sample_num']))
        os.makedirs(self.out_dir, exist_ok=True)

        self.flame_indices = {}
        for key, value in config["train"]["flame_indices"]["files"].items():
            self.flame_indices[key] = np.load(os.path.join(config["train"]["flame_indices"]["folder"], value))
        
        
        self._img_size = self.config["model"]["model_config"]["img_size"]
        self.head_mesh = HeadMesh(flame_config=config["constants"], batch_size=config["batch_size"],
                                  image_size=self._img_size)

        # # region metrics initialization
        # self.iou_metric = SoftIoUMetric(compute_on_step=True)
        # self.metrics_2d = MetricCollection(
        #     {
        #         "fr_2d_005": FailureRate(compute_on_step=True, threshold=0.05, below=True),
        #         "fr_2d_01": FailureRate(compute_on_step=True, threshold=0.1, below=True),
        #         "nme_2d": KeypointsNME(compute_on_step=True),
        #     }
        # )

        self.metrics_reprojection = MetricCollection(
            {
                "reproject_fr_2d_005": FailureRate(compute_on_step=True, threshold=0.05, below=True),
                "reproject_fr_2d_01": FailureRate(compute_on_step=True, threshold=0.1, below=True),
                "reproject_nme_2d": KeypointsNME(compute_on_step=True),
            }
        )

        self.metrics_3d = MetricCollection(
            {
                "fr_3d_005": FailureRate(compute_on_step=True, threshold=0.05, below=True),
                "fr_3d_01": FailureRate(compute_on_step=True, threshold=0.1, below=True),
                "nme_3d": KeypointsNME(compute_on_step=True),
            }
        )
        # endregion

    @property
    def is_master(self) -> bool:
        """
        Returns True if the caller is the master node (Either code is running on 1 GPU or current rank is 0)
        """
        return (self.use_ddp is False) or (torch.distributed.get_rank() == 0)

    def _load_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.config.get("load_weights", False):
            weights_path = self.config["weights_path"]
            if "h5" in weights_path:
                model = torch.load(weights_path)
            else:
                model = load_from_lighting(self.model, weights_path)
        return model

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def compute_loss(
            self, loss: Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Return tuple of loss tensor and dictionary of named losses as second argument (if possible)
        """
        if torch.is_tensor(loss):
            return loss, {}

        elif isinstance(loss, (tuple, list)):
            total_loss = sum(loss)
            return total_loss, dict((f"loss_{i}", l) for i, l in enumerate(loss))

        elif isinstance(loss, dict):
            total_loss = 0
            for k, v in loss.items():
                total_loss = total_loss + v

            return total_loss, loss
        else:
            raise ValueError("Incompatible Loss type")

    def _get_batch_size(self, mode: str = "train") -> int:
        if isinstance(self.config["batch_size"], dict):
            return self.config["batch_size"][mode]
        return self.config["batch_size"]

    def _get_num_workers(self, loader_name: str) -> int:
        if "num_workers" not in self.config:
            return cpu_count()
        if isinstance(self.config["num_workers"], float):
            return int(cpu_count() * self.config["num_workers"])
        if isinstance(self.config["num_workers"], dict):
            return self.config["num_workers"][loader_name]
        return self.config["num_workers"]

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.train_dataset, self.config, "train")

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.val_dataset, self.config, "val")

    def _get_dataloader(self, dataset: Dataset, config: Dict[str, Any], loader_name: str) -> DataLoader:
        """
        Instantiate DataLoader for given dataset w.r.t to config and mode.
        It supports creating a custom sampler.
        Note: For DDP mode, we support custom samplers, but trainer must be called with:
            >>> replace_sampler_ddp=False

        Args:
           dataset: Dataset instance
            config: Dataset config
            loader_name: Loader name (train or val)

        Returns:

        """
        
        collate_fn = get_collate_for_dataset(dataset)

        dataset_config = config[loader_name]
        if "sampler" not in dataset_config or dataset_config["sampler"] == "none":
            sampler = None

        drop_last = loader_name == "train"

        if self.use_ddp:
            world_size = torch.distributed.get_world_size()
            local_rank = torch.distributed.get_rank()
            if sampler is None:
                sampler = DistributedSampler(dataset, world_size, local_rank)
            # else:
            #     sampler = DistributedSamplerWrapper(sampler, world_size, local_rank)

        should_shuffle = (sampler is None) and (loader_name == "train")
        batch_size = self._get_batch_size(loader_name)
        # Number of workers must not exceed batch size
        num_workers = min(batch_size, self._get_num_workers(loader_name))
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last,
            collate_fn=collate_fn,
        )
        return loader

    def training_step(self, batch: Dict[str, Any], batch_nb: int) -> Dict[str, Any]:
        # # pdb.set_trace()
        batch = any2device(batch, self.device)
        return self._step_fn(batch, batch_nb, loader_name="train")

    def validation_step(self, batch: Dict[str, Any], batch_nb: int) -> Dict[str, Any]:
        # # pdb.set_trace()
        batch = any2device(batch, self.device)
        return self._step_fn(batch, batch_nb, loader_name="valid")

    def _get_optim(self, model: torch.nn.Module, optimizer_config: Dict[str, Any]) -> torch.optim.Optimizer:
        """Creates model optimizer from Trainer config
        Args:
            params (list): list of named model parameters to be trained
        Returns:
            torch.optim.Optimizer: model optimizer
        """
        if self.learning_rate:
            optimizer_config["lr"] = self.learning_rate
        optimizer = get_optimizer(model, optimizer_config=optimizer_config)
        return optimizer

    def _get_scheduler(self, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """Creates scheduler for a given optimizer from Trainer config
        Args:
            optimizer (torch.optim.Optimizer): optimizer to be updated
        Returns:
            torch.optim.lr_scheduler._LRScheduler: optimizer scheduler
        """
        scheduler_config = self.config["scheduler"]
        scheduler = get_scheduler(optimizer, scheduler_config)
        return {"scheduler": scheduler, "monitor": self.config.get("metric_to_monitor", "valid/loss")}

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
        self.optimizer = self._get_optim(self.model, self.config["optimizer"])
        scheduler = self._get_scheduler(self.optimizer)
        return [self.optimizer], [scheduler]

    def on_epoch_end(self) -> None:
        self.epoch_num += 1

    def on_pretrain_routine_start(self) -> None:
        if not isinstance(self.logger, DummyLogger):
            for logger in self.logger:
                if isinstance(logger, TensorBoardLogger):
                    self.tensorboard_logger = logger

    def on_pretrain_routine_end(self) -> None:
        optimizers = self.optimizers()
        if isinstance(optimizers, torch.optim.Optimizer):
            optimizers = [optimizers]
        if optimizers is None or len(optimizers) == 0:
            raise RuntimeError("List of optimizers is not available on the start of the training")
        self._initial_learning_rates = [[float(pg["lr"]) for pg in opt.param_groups] for opt in optimizers]

    def _learning_rate(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer: torch.optim.Optimizer = None,
        optimizer_idx: int = None,
        optimizer_closure: Optional[Callable] = None,
        on_tpu: bool = None,
        using_native_amp: bool = None,
        using_lbfgs: bool = None,
    ) -> None:
        # Learning rate warmup
        num_warmup_steps = int(self.config.get("scheduler", {}).get("warmup_steps", 0))

        if self.trainer.global_step < num_warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / num_warmup_steps)
            optimizer = typing.cast(torch.optim.Optimizer, optimizer)
            optimizer_idx = optimizer_idx if optimizer_idx is not None else 0
            for pg_index, pg in enumerate(optimizer.param_groups):
                pg["lr"] = lr_scale * self._initial_learning_rates[optimizer_idx][pg_index]

        return super().optimizer_step(
            epoch=epoch,
            batch_idx=batch_idx,
            optimizer=optimizer,
            optimizer_idx=optimizer_idx,
            optimizer_closure=optimizer_closure,
            on_tpu=on_tpu,
            using_native_amp=using_native_amp,
            using_lbfgs=using_lbfgs,
        )

    def _get_keypoints_2d(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if OUTPUT_2D_LANDMARKS in outputs.keys():
            return outputs[OUTPUT_2D_LANDMARKS] * self._img_size
        return float(self.stride) * unravel_index(outputs[OUTPUT_LANDMARKS_HEATMAP]).flip(-1)

    def _batch_visulization(self, names, images, lm_2ds, lm_2d_projs, lm_2d_gts, lm_2d_projs_gts, epoch):
        B, V, C, H, W = images.shape

        for b in range(B):
            for v in range(V):
                name = names[v][b]
                im = unorm(images[b, v, ...]).permute(1, 2, 0).detach().cpu().numpy()
                lm_2d = lm_2ds[b, v, ...]
                lm_2d_proj = lm_2d_projs[b, v, ...]
                lm_2d_gt = lm_2d_gts[b, v, ...]
                lm_2d_proj_gt = lm_2d_projs_gts[b, v, ...]
                
                scale = 4
                img_size = images.shape[-1] * scale
                img = cv2.resize(im, (img_size, img_size))

                
                landmarks = (lm_2d * img_size).clip(min=0, max=img_size)
                landmarks_proj = (lm_2d_proj * img_size).clip(min=0, max=img_size)
                landmarks_gt = (lm_2d_gt * img_size).clip(min=0, max=img_size)
                landmarks_proj_gt = (lm_2d_proj_gt * img_size).clip(min=0, max=img_size)
                
                # im_out_srt_dir = os.path.join(out_dir, 'lm_name_{}_epoch_{}_247_landmarks.png'.format(name, str(epoch).zfill(3)))
                # cv2.imwrite(im_out_srt_dir, cv2.cvtColor(im, cv2.COLOR_RGB2BGR) * 127.5 + 128)

                # img_ = cv2.imread(im_out_srt_dir)
                img_ = np.clip(cv2.cvtColor(img, cv2.COLOR_RGB2BGR) * 255, 0, 255)
                
                for i in range(landmarks.shape[0]): cv2.circle(img_, (int(landmarks[i][0]), int(landmarks[i][1])), 4, (0, 0, 255), -1)
                for i in range(landmarks_proj.shape[0]): cv2.circle(img_, (int(landmarks_proj[i][0]), int(landmarks_proj[i][1])), 2, (0, 255, 255), -1)
                for i in range(landmarks_gt.shape[0]): cv2.circle(img_, (int(landmarks_gt[i][0]), int(landmarks_gt[i][1])), 4, (0, 0, 0), -1)
                # for i in range(landmarks_proj_gt.shape[0]): cv2.circle(img_, (int(landmarks_proj_gt[i][0]), int(landmarks_proj_gt[i][1])), 2, (0, 0, 0), -1)
                
                # out_vis_dir = out_dir + '_vis'
                # os.makedirs(out_vis_dir, exist_ok=True)
                im_out_srt_dir_vis = os.path.join(self.out_dir, 'lm_name_{}_epoch_{}_68_landmarks_vis.png'.format(name, str(epoch).zfill(3)))
                cv2.imwrite(im_out_srt_dir_vis, img_)

        
    def _step_fn(self, batch: Dict[str, Any], batch_nb: int, loader_name: str):
        weight_dad3d = self.weight_dad3d
        weight_proj = self.weight_proj
        weight_proj_ref_gt = self.weight_proj_ref_gt
        weight_proj_srt_gt = self.weight_proj_srt_gt
        weight_lm2d = self.weight_lm2d
        weight_mesh = self.weight_mesh
        names = batch['names']
        images = batch['views'] # BxVx3xHxW (512)
        krts = batch['KRTs'] # BxVx3x4
        lm_2d_gt = batch['lm2d_points_gt']
        dad3d_data_items = batch['dad3d_data_items']
        dad3d_images, dad3d_targets = self.get_input(dad3d_data_items[0])
        for i in range(1, len(dad3d_data_items)):
            imgs, tgts = self.get_input(dad3d_data_items[i])
            dad3d_images = torch.cat((dad3d_images, imgs), 0)
            for k in tgts.keys():
                dad3d_targets[k] = torch.cat((dad3d_targets[k], tgts[k]), 0)
                
        num_views_srt = self.num_views_srt # int(len(names)/4)
        num_views_srt = 4 # int(len(names)/4)
        B, V, C, H, W = images.shape
        outputs_all = self.forward(torch.cat((images.reshape(-1, C, H, W), dad3d_images), 0))
        outputs = {'OUTPUT_LANDMARKS_HEATMAP':outputs_all['OUTPUT_LANDMARKS_HEATMAP'][:-dad3d_images.shape[0], ...], \
            'OUTPUT_3DMM_PARAMS':outputs_all['OUTPUT_3DMM_PARAMS'][:-dad3d_images.shape[0], ...], \
            'OUTPUT_2D_LANDMARKS':outputs_all['OUTPUT_2D_LANDMARKS'][:-dad3d_images.shape[0], ...]}
        outputs_dad3d = {'OUTPUT_LANDMARKS_HEATMAP':outputs_all['OUTPUT_LANDMARKS_HEATMAP'][-dad3d_images.shape[0]:, ...], \
            'OUTPUT_3DMM_PARAMS':outputs_all['OUTPUT_3DMM_PARAMS'][-dad3d_images.shape[0]:, ...], \
            'OUTPUT_2D_LANDMARKS':outputs_all['OUTPUT_2D_LANDMARKS'][-dad3d_images.shape[0]:, ...]}
        dad3d_total_loss, dad3d_loss_dict = self.criterion(outputs_dad3d, dad3d_targets, self.epoch_num)
        dad3d_total_loss *= weight_dad3d
        
        lm_2d = outputs['OUTPUT_2D_LANDMARKS'].reshape(B, V, -1, 2)
        
        names_tgt = names[num_views_srt:]
        images_tgt = images[:, num_views_srt:, :, :, :]
        if num_views_srt == 0:
            lm_2d_srt = lm_2d[:, :, :, :]
            krts_srt = krts[:, :, :, :]
        else:
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
        params_3dmm_pred = outputs['OUTPUT_3DMM_PARAMS'].reshape(B, V, -1)
        vertices_3d_target = batch['flame_meshes']
        B, V, N, C = vertices_3d_target.shape
        # if self.config['single_view_gt']: # just calculate gt loss on the reference view
        #     lm_2d_loss = torch.nn.L1Loss()(lm_2d[:, -1:, :, :], lm_2d_gt[:, -1:, :, :]) * weight_lm2d
        # else:
        #     lm_2d_loss = torch.nn.L1Loss()(lm_2d, lm_2d_gt) * weight_lm2d
        lm_2d_gt_loss = torch.nn.L1Loss()(lm_2d[:, -1:, :, :], lm_2d_gt[:, -1:, :, :]) * weight_lm2d
        lm_2d_pseudo_gt_weights = torch.ones_like(lm_2d[:, :-1, :, :])
        lm_2d_pseudo_gt_weights[:, :, :17, :] = self.weight_jaw_pseudo # set the weights of jaw points, 0-16, as 0.1
        lm_2d_pseudo_gt_loss = torch.nn.L1Loss()(lm_2d[:, :-1, :, :] * lm_2d_pseudo_gt_weights, lm_2d_gt[:, :-1, :, :] * lm_2d_pseudo_gt_weights) * weight_lm2d
        lm_2d_loss = (lm_2d_gt_loss + lm_2d_pseudo_gt_loss) * 0.5
        # lm_2d_loss = torch.nn.L1Loss()(lm_2d, lm_2d_gt) * weight_lm2d
        vertices_3d_loss = self.criterion.criterions[1](params_3dmm_pred.reshape(B*V, -1), vertices_3d_target.reshape(B*V, N, C)) * weight_mesh
        if num_views_srt == 0:
            lm_2d_proj_loss = torch.nn.L1Loss()(lm_2d_gt_tgt, lm_2d_proj_tgt) * weight_proj
            total_loss = lm_2d_proj_loss + vertices_3d_loss + lm_2d_loss
            loss_dict = {'lm_2d_proj_loss':lm_2d_proj_loss, 'vertices_3d_loss':vertices_3d_loss, 'lm_2d_loss':lm_2d_loss}
        else:
            lm_2d_proj_loss = torch.nn.L1Loss()(lm_2d_tgt, lm_2d_proj_tgt) * weight_proj
            if self.config['single_view_gt']:
                # total_loss = lm_2d_proj_loss + vertices_3d_loss + lm_2d_loss
                # loss_dict = {'lm_2d_proj_loss':lm_2d_proj_loss, 'vertices_3d_loss':vertices_3d_loss, 'lm_2d_loss':lm_2d_loss}
                lm_2d_proj_ref_gt_loss = torch.nn.L1Loss()(lm_2d_proj_tgt[:, -1:, :, :], lm_2d_gt[:, -1:, :, :]) * weight_proj_ref_gt
                total_loss = lm_2d_proj_loss + lm_2d_proj_ref_gt_loss + vertices_3d_loss + lm_2d_loss + dad3d_total_loss
                loss_dict = {'lm_2d_proj_loss':lm_2d_proj_loss, 'lm_2d_proj_ref_gt_loss':lm_2d_proj_ref_gt_loss, 'vertices_3d_loss':vertices_3d_loss, 'lm_2d_loss':lm_2d_loss}
                for k in dad3d_loss_dict.keys():
                    loss_dict[k] = dad3d_loss_dict[k] * weight_dad3d

                loss_dict.update(dad3d_loss_dict)
            else:
                lm_2d_proj_srt_gt_loss = torch.nn.L1Loss()(lm_2d_gt_srt, lm_2d_proj_srt) * weight_proj_srt_gt
                total_loss = lm_2d_proj_loss + lm_2d_proj_srt_gt_loss + vertices_3d_loss + lm_2d_loss + dad3d_total_loss
                loss_dict = {'lm_2d_proj_loss':lm_2d_proj_loss, 'lm_2d_proj_srt_gt_loss':lm_2d_proj_srt_gt_loss, 'vertices_3d_loss':vertices_3d_loss, 'lm_2d_loss':lm_2d_loss, 'dad3d_loss': dad3d_total_loss}

        
        
        for metric_name, metric_value in loss_dict.items():
            self.log(
                f"{loader_name}/metrics/{metric_name}",
                metric_value,
                on_epoch=True,
            )
        
        if loader_name == 'valid':
            num_views_srt = 4
            names_tgt = names[num_views_srt:]
            images_tgt = images[:, num_views_srt:, :, :, :]
            lm_2d_srt = lm_2d[:, :num_views_srt, :, :]
            krts_srt = krts[:, :num_views_srt, :, :]
            lm_2d_tgt = lm_2d[:, num_views_srt:, :, :]
            krts_tgt = krts[:, num_views_srt:, :, :]
            lm_2d_gt_tgt = lm_2d_gt[:, num_views_srt:, :, :]
            lm_3d_srt = TriangulateDLT_BatchCam(krts_srt, lm_2d_srt)
            # lm_2d_proj_tgt = ProjectKRT_Batch(krts_tgt, lm_3d_srt.unsqueeze(1))
            lm_2d_proj = ProjectKRT_Batch(krts, lm_3d_srt.unsqueeze(1))
            lm_2d_gt_srt = lm_2d_gt[:, :num_views_srt, :, :]
            lm_3d_gt_srt = TriangulateDLT_BatchCam(krts_srt, lm_2d_gt_srt)
            # lm_2d_proj_gt_tgt = ProjectKRT_Batch(krts_tgt, lm_3d_gt_srt.unsqueeze(1))            
            lm_2d_proj_gt = ProjectKRT_Batch(krts, lm_3d_gt_srt.unsqueeze(1))            
            # self._batch_visulization(names_tgt, images_tgt, lm_2d_tgt, lm_2d_proj, lm_2d_gt_tgt, lm_2d_proj_gt, self.epoch_num)
            self._batch_visulization(names, images, lm_2d, lm_2d_proj, lm_2d_gt, lm_2d_proj_gt, self.epoch_num)
            
        # total_loss, loss_dict = self.criterion(outputs, targets, self.epoch_num)

        # process_2d_branch = OUTPUT_2D_LANDMARKS in outputs.keys() or OUTPUT_LANDMARKS_HEATMAP in outputs.keys()

        # if process_2d_branch:
        #     self.log(
        #         f"{loader_name}/metrics/heatmap_iou",
        #         self.iou_metric(
        #             outputs[OUTPUT_LANDMARKS_HEATMAP].sigmoid(),
        #             targets[TARGET_LANDMARKS_HEATMAP],
        #         ),
        #         on_epoch=True,
        #     )
            
        #     outputs_2d = self._get_keypoints_2d(outputs=outputs) 
        #     # print(outputs_2d.shape)
        #     outputs_2d = outputs_2d * targets[TARGET_2D_LANDMARKS_PRESENCE][..., None]
        #     # print(outputs_2d.shape)
        #     targets_2d = (
        #         targets[TARGET_2D_LANDMARKS] * targets[TARGET_2D_LANDMARKS_PRESENCE][..., None] * self._img_size
        #     )
        #     # print (targets_2d.shape, '======')
        #     metrics_2d = self.metrics_2d(outputs_2d, {"keypoints": targets_2d, "bboxes": targets[INPUT_BBOX_KEY]})
        #     for metric_name, metric_value in metrics_2d.items():
        #         self.log(
        #             f"{loader_name}/metrics/{metric_name}",
        #             metric_value,
        #             on_epoch=True,
        #         )

        # params_3dmm = outputs[OUTPUT_3DMM_PARAMS]
        # projected_vertices = self.head_mesh.reprojected_vertices(params_3dmm=params_3dmm, to_2d=True)
        # reprojected_pred = projected_vertices[:, self.flame_indices["face"]]
        # reprojected_gt = targets[TARGET_2D_FULL_LANDMARKS][:, self.flame_indices["face"]]
        # # print (reprojected_gt.shape, reprojected_pred.shape)
        # reprojected_metrics = self.metrics_reprojection(
        #     reprojected_pred, {"keypoints": reprojected_gt, "bboxes": targets[INPUT_BBOX_KEY]}
        # )

        # for metric_name, metric_value in reprojected_metrics.items():
        #     self.log(
        #         f"{loader_name}/metrics/{metric_name}",
        #         metric_value,
        #         on_epoch=True,
        #     )

        # pred_3d_vertices = self.head_mesh.vertices_3d(params_3dmm=params_3dmm, zero_rotation=True)
        # metrics_3d = self.metrics_3d(
        #     normalize_to_cube(pred_3d_vertices[:, self.flame_indices["face"]]),
        #     {
        #         "keypoints": normalize_to_cube(
        #             targets[TARGET_3D_MODEL_VERTICES][:, self.flame_indices["face"]]
        #         )
        #     },
        # )

        # for metric_name, metric_value in metrics_3d.items():
        #     self.log(
        #         f"{loader_name}/metrics/{metric_name}",
        #         metric_value,
        #         on_epoch=True,
        #     )


        
        # Logging
        self.log(f"{loader_name}/total_loss", total_loss, prog_bar=True, sync_dist=self.use_ddp, batch_size=B)
        if len(loss_dict):
            self.log_dict(
                dictionary=dict((f"{loader_name}/" + k, v) for k, v in loss_dict.items()),
                prog_bar=True,
                sync_dist=self.use_ddp,
                batch_size=B
            )
            
        if num_views_srt == 0:    
            return {"loss": total_loss, "lm_2d_proj_loss": lm_2d_proj_loss, 'lm_2d_loss':lm_2d_loss, 'vertices_3d_loss':vertices_3d_loss}
        else:
            if self.config['single_view_gt']:
                return_loss_dict = {"loss": total_loss, "lm_2d_proj_loss": lm_2d_proj_loss, "lm_2d_proj_ref_gt_loss": lm_2d_proj_ref_gt_loss, 'lm_2d_loss':lm_2d_loss, 'vertices_3d_loss':vertices_3d_loss}
                for k in dad3d_loss_dict.keys():
                    return_loss_dict[k] = dad3d_loss_dict[k] * weight_dad3d
                return return_loss_dict
            else:
                return {"loss": total_loss, "lm_2d_proj_loss": lm_2d_proj_loss, "lm_2d_proj_srt_gt_loss": lm_2d_proj_srt_gt_loss, 'lm_2d_loss':lm_2d_loss, 'vertices_3d_loss':vertices_3d_loss, 'dad3d_loss': dad3d_total_loss}


    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()

    def on_train_epoch_end(self, unused: Optional[Any] = None) -> None:
        super().on_train_epoch_end()
        learning_rate = self._learning_rate()
        self.log("train/learning_rate", learning_rate)

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()

    def export_jit_model(self, checkpoint_filename: str) -> torch.jit.ScriptModule:
        """
        Loads weighs of the model from given checkpoint into self.model and
        exports it via torch tracing.

        Note: If you don't want to export model, override this method and throw NotImplementedError

        Args:
            checkpoint_filename: Best checkpoint

        Returns:
            Instance of ScriptModule
        """

        load_from_lighting(self.model, checkpoint_filename)
        model_config = self.config["model"]["model_config"]
        example_input = torch.randn(1, model_config["num_channels"], model_config["img_size"], model_config["img_size"])
        return torch.jit.trace(self.model.eval().cuda(), example_input.cuda(), strict=False)

    def _build_loss(self, config: Dict) -> torch.nn.Module:
        return LossModule.from_config(config)


def get_collate_for_dataset(dataset: Union[Dataset, ConcatDataset]) -> Callable:
    """
    Returns collate_fn function for dataset. By default, default_collate returned.
    If the dataset has method get_collate_fn() we will use it's return value instead.
    If the dataset is ConcatDataset, we will check whether all get_collate_fn() returns
    the same function.

    Args:
       dataset: Input dataset

    Returns:
        Collate function to put into DataLoader
    """
    collate_fn = dataset.get_collate_fn()

    if isinstance(dataset, ConcatDataset):
        collates = [get_collate_for_dataset(ds) for ds in dataset.datasets]
        if len(set(collates)) != 1:
            raise ValueError("Datasets have different collate functions")
        collate_fn = collates[0]
    return collate_fn
