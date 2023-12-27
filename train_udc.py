import os

os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa

import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from utils.util import get_timestamp, make_source_code_snapshot, read_json
from utils.bbox_utils import BBoxRayHelper
from datasets.ray_utils import get_ray_directions, get_rays
from datasets.geo_utils import center_pose_from_avg

# dataloader
from torch.utils.data import DataLoader, WeightedRandomSampler
from datasets import dataset_dict
from omegaconf import OmegaConf
from catalyst.data.sampler import DistributedSamplerWrapper

# models
from models.nerf_model import UDCNeRF
from models.embedding_helper import EmbeddingVoxel, Embedding
from models.rendering import render_rays_udc, render_rays_edit_hack

# optimizer, scheduler, visualization
from utils import get_optimizer, get_scheduler, get_learning_rate
from utils.train_helper import visualize_val_image_udc

# losses
from models.losses import get_udcloss

# metrics
from utils.metrics import psnr, ssim, miou
import lpips

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
    #  np.random.seed(seed)
    #  random.seed(seed)
     torch.backends.cudnn.deterministic = True


class CustomWeightedRandomSampler(WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())


class UDCNeRFSystem(LightningModule):
    def __init__(self, config):
        super(UDCNeRFSystem, self).__init__()

        # config & losses
        self.config = config
        self.loss = get_udcloss(config)

        # model
        self.use_voxel_embedding = False
        self.embedding_xyz = Embedding(3, self.config.model.N_freq_xyz) # Using no voxel embedding here. 

        self.embedding_dir = Embedding(3, self.config.model.N_freq_dir)
        self.embeddings = {"xyz": self.embedding_xyz, "dir": self.embedding_dir}

        self.nerf_coarse = UDCNeRF(self.config.model, coarse=True)
        self.models = {"coarse": self.nerf_coarse}

        if config.model.N_importance > 0: # 64 by default
            self.nerf_fine = UDCNeRF(self.config.model, coarse=False)
            self.models["fine"] = self.nerf_fine

        # number of parameters
        pytorch_total_params = sum(p.numel() for p in self.nerf_coarse.parameters())
        print("[INFO]: TOTAL NUMBER OF PARAMETERS FOR COARSE MODEL: {}".format(pytorch_total_params/1024**2))
        pytorch_total_params = sum(p.numel() for p in self.nerf_fine.parameters())
        print("[INFO]: TOTAL NUMBER OF PARAMETERS for FINE MODEL: {}".format(pytorch_total_params/1024**2))

        self.models_to_train = [
            self.models,
        ]

        self.img_wh = tuple(self.config.img_wh)
        self.train_mode = None


    def forward(self, rays, instance_mask=None, extra=dict()):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        if self.train_mode:
            cur_chunk = self.config.train.chunk
        else:
            cur_chunk = self.config.train.chunk_eval
        for i in range(0, B, cur_chunk):
            extra_chunk = dict()
            for k, v in extra.items():
                if isinstance(v, torch.Tensor):
                    extra_chunk[k] = v[i : i + cur_chunk]
                else:
                    extra_chunk[k] = v
            
            rendered_ray_chunks = render_rays_udc(
                models=self.models,
                embeddings=self.embeddings, # dict of xyz & dir
                rays=rays[i : i + cur_chunk],
                instance_mask=instance_mask[i : i + cur_chunk] if instance_mask is not None else instance_mask,
                N_samples=self.config.model.N_samples, # 64
                use_disp=self.config.model.use_disp,   # false
                perturb=self.config.model.perturb, # 1
                noise_std=self.config.model.noise_std, # 1
                N_importance=self.config.model.N_importance, # 64
                chunk=cur_chunk,  # chunk size is effective in val mode
                use_zero_as_last_delta=self.config.model.use_zero_as_last_delta, # default false 
                white_back = False,
                use_alpha_for_fuse=False,     # use alpha for fused
                use_fused_alpha_for_opacity=self.config.model.use_fused_alpha_for_opacity,  # not use this term finally
                remove_bgobj = self.config.model.remove_bgobj,
                block_enable = extra["is_testnvs"] or (self.current_epoch>self.config.model.block_epoch),
                block_type = self.config.model.block_type,
                gumbel_tau = self.config.model.gumbel_tau,                 
                use_which_weight_for_fine = self.config.model.use_which_weight_for_fine, #  ['global', 'local', 'comp']
                train_mode = self.train_mode,
                **extra_chunk,
            )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def setup(self, stage):
        dataset = dataset_dict[self.config.dataset_name]
        kwargs = {
            "img_wh": tuple(self.config.img_wh),
        }
        kwargs["dataset_extra"] = self.config.dataset_extra
        self.train_dataset = dataset(split="train", **kwargs)
        self.val_dataset = dataset(split="val", **kwargs)
    

    def setup_testnvs(self):
        # load testnvs dataset
        dataset = dataset_dict[self.config.dataset_name]
        kwargs = {
            "img_wh": tuple(self.config.img_wh),
        }
        kwargs["dataset_extra"] = self.config.dataset_extra
        self.testnvs_dataset = dataset(split="test_nvs", **kwargs)
        self.val_dataset = dataset(split="val", **kwargs)

        # load pre-trained ckpts
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
    

    def setup_edit(self):
        # initialize rendering parameters
        # dataset_extra = self.ckpt_config.dataset_extra
        dataset_extra = self.config.dataset_extra
        self.near = self.config.get("near", dataset_extra.near)
        self.far = self.config.get("far", dataset_extra.far)
        self.scale_factor = dataset_extra.scale_factor
        self.pose_avg = np.concatenate(
            [np.eye(3), np.array(dataset_extra["scene_center"])[:, None]], 1
        )
        # print(self.near, self.far, self.scale_factor, self.pose_avg)

        # self.config.train.chunk = self.config.train.chunk * 4

        self.object_to_remove = []
        self.active_object_ids = [0]
        # self.active_object_ids = []
        self.object_pose_transform = {}
        self.object_bbox_ray_helpers = {}
        self.bbox_enlarge = 0.0

    
    def initialize_object_bbox(self, obj_id: int):
        # print('init bbox: ', self.config.ckpt_config_path, obj_id)
        self.object_bbox_ray_helpers[str(obj_id)] = BBoxRayHelper(
            self.config.ckpt_config_path, obj_id
        )

    def set_object_pose_transform(
        self,
        obj_id: int,
        pose: np.ndarray,
        obj_dup_id: int = 0,  # for object duplication
    ):
        self.active_object_ids.append(obj_id)
        if obj_id not in self.active_object_ids:
            self.initialize_object_bbox(obj_id)
        self.object_pose_transform[f"{obj_id}_{obj_dup_id}"] = pose
    

    def generate_rays(self, obj_id, rays_o, rays_d):
        near = self.near
        far = self.far
        if obj_id == 0:
        # if True:
            batch_near = near / self.scale_factor * torch.ones_like(rays_o[:, :1])
            batch_far = far / self.scale_factor * torch.ones_like(rays_o[:, :1])
            # rays_o = rays_o / self.scale_factor
            rays = torch.cat([rays_o, rays_d, batch_near, batch_far], 1)  # (H*W, 8)
        else:
            bbox_mask, bbox_batch_near, bbox_batch_far = self.object_bbox_ray_helpers[
                str(obj_id)
            ].get_ray_bbox_intersections(
                rays_o,
                rays_d,
                self.scale_factor,
                # bbox_enlarge=self.bbox_enlarge / self.get_scale_factor(obj_id),
                bbox_enlarge=self.bbox_enlarge,  # in physical world
            )
            # for area which hits bbox, we use bbox hit near far
            # bbox_ray_helper has scale for us, do no need to rescale
            batch_near_obj, batch_far_obj = bbox_batch_near, bbox_batch_far
            # for the invalid part, we use 0 as near far, which assume that (0, 0, 0) is empty
            batch_near_obj[~bbox_mask] = torch.zeros_like(batch_near_obj[~bbox_mask])
            batch_far_obj[~bbox_mask] = torch.zeros_like(batch_far_obj[~bbox_mask])
            rays = torch.cat(
                [rays_o, rays_d, batch_near_obj, batch_far_obj], 1
            )  # (H*W, 8)
        rays = rays.cuda()
        return rays
    
    def render_edit(
        self,
        h: int,
        w: int,
        camera_pose_Twc: np.ndarray,
        fovx_deg: float = 70,
        show_progress: bool = True,
        render_bg_only: bool = False,
        render_obj_only: bool = False,
        white_back: bool = False,
    ):
        focal = (w / 2) / np.tan((fovx_deg / 2) / (180 / np.pi))
        directions = get_ray_directions(h, w, focal).cuda()  # (h, w, 3)
        Twc = center_pose_from_avg(self.pose_avg, camera_pose_Twc)
        args = {}
        results = {}
        obj_ids = []
        rays_list = []

        # only render background
        if render_bg_only:
            self.active_object_ids = [0]

        # only render objects
        if render_obj_only:
            self.active_object_ids.remove(0)

        processed_obj_id = []
        # print('self.active_object_ids: ', self.active_object_ids)
        for obj_id in self.active_object_ids:
            # count object duplication
            obj_duplication_cnt = np.sum(np.array(processed_obj_id) == obj_id)
            if obj_id == 0:
            # if True:
                # for scene, transform is Identity
                Tow = transform = np.eye(4)
            else:
                object_pose = self.object_pose_transform[
                    f"{obj_id}_{obj_duplication_cnt}"
                ]
                # transform in the real world scale
                Tow_orig = self.get_object_bbox_helper(
                    obj_id
                ).get_world_to_object_transform()
                # transform object into center, then apply user-specific object poses
                transform = np.linalg.inv(Tow_orig) @ object_pose @ Tow_orig
                # for X_c = Tcw * X_w, when we applying transformation on X_w,
                # it equals to Tcw * (transform * X_w). So, Tow = inv(transform) * Twc
                Tow = np.linalg.inv(transform)
                # Tow = np.linalg.inv(Tow)  # this move obejct to center
            processed_obj_id.append(obj_id)
            Toc = Tow @ Twc
            # resize to NeRF scale
            Toc[:, 3] /= self.scale_factor
            Toc = torch.from_numpy(Toc).float().cuda()[:3, :4]
            # all the rays_o and rays_d has been converted to NeRF scale
            rays_o, rays_d = get_rays(directions, Toc)
            rays = self.generate_rays(obj_id, rays_o, rays_d)
            # light anchor should also be transformed
            Tow = torch.from_numpy(Tow).float()
            transform = torch.from_numpy(transform).float()
            obj_ids.append(obj_id)
            rays_list.append(rays)

        # split chunk
        B = rays_list[0].shape[0]
        chunk = self.config.train.chunk
        results = defaultdict(list)
        for i in tqdm(range(0, B, self.config.train.chunk), disable=not show_progress):
            with torch.no_grad():
                rendered_ray_chunks = render_rays_edit_hack(
                    models=self.models,
                    embeddings=self.embeddings,
                    rays_list=[r[i : i + chunk] for r in rays_list],
                    obj_instance_ids=obj_ids,
                    N_samples=self.config.model.N_samples,
                    use_disp=self.config.model.use_disp,
                    perturb=0,
                    noise_std=0,
                    N_importance=self.config.model.N_importance,
                    chunk=self.config.train.chunk,  # chunk size is effective in val mode
                    white_back=white_back,
                    scene_name=self.config.train.scene_name,
                    **args,
                )
            for k, v in rendered_ray_chunks.items():
                results[k] += [v.detach().cpu()]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)

        return results


    def get_camera_pose_by_frame_idx(self, frame_idx):
        return self.poses[frame_idx]


    def get_object_bbox_helper(self, obj_id: int):
        return self.object_bbox_ray_helpers[str(obj_id)]

    
    def get_skipping_bbox_helper(self):
        skipping_bbox_helper = {}
        for obj_id in self.object_to_remove:
            skipping_bbox_helper[str(obj_id)] = self.object_bbox_ray_helpers[
                str(obj_id)
            ]
        return skipping_bbox_helper

    def reset_active_object_ids(self):
        self.active_object_ids = [0]

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.config.train, self.models_to_train)
        scheduler = get_scheduler(self.config.train, self.optimizer)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        batch_size = self.config.train.batch_size

        samples_weight = torch.zeros(len(self.train_dataset))
        nums_k = torch.sum(self.train_dataset.all_instance_masks, 0) # k
        ids_list = torch.argmax(self.train_dataset.all_instance_masks.int(), -1) # k
        
        if self.config.model.sample_type == 'two_category':
            weights = torch.tensor([torch.sum(nums_k) / nums_k[0], torch.sum(nums_k) / torch.sum(nums_k[1:])])
            forced_p = torch.tensor(self.config.model.forced_p) # [0.5, 0.5]   [0.75, 0.25]  [0.25, 0.75]
            weights = weights * forced_p
            ids_list[ids_list>1] = 1

        elif self.config.model.sample_type == 'k_category':
            weights = torch.sum(nums_k) / nums_k
            forced_p = torch.tensor(self.config.model.forced_p) # [0.75, 0.05, 0.05, 0.05, 0.05, 0.05]  # [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
            weights = weights * forced_p

        temp_list = list(range(len(weights)))
        id2weight = dict(zip(temp_list, weights))
        # print(id2weight, ids_list.shape, ids_list.dtype)
        samples_weight = ids_list.float().apply_(id2weight.get)
        # print(samples_weight.shape, samples_weight.dtype, samples_weight.max(), samples_weight.min())

        train_sampler = CustomWeightedRandomSampler(weights=samples_weight, \
                                              num_samples=len(self.train_dataset), \
                                              replacement=True)
        
        train_loader = {
            'weighted': 
                      DataLoader(
                        self.train_dataset,
                        # shuffle=True,
                        shuffle=False,
                        num_workers=12,
                        batch_size=batch_size,
                        pin_memory=True,
                        # sampler=train_sampler,
                        sampler=DistributedSamplerWrapper(train_sampler, 
                                                          shuffle=False,
                                                          num_replicas=torch.distributed.get_world_size(),
                                                          rank=torch.distributed.get_rank()
                                                        )
                        ),
        }
        return train_loader

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=12,
            batch_size=1,  # validate one image (H*W rays) at a time
            pin_memory=True,
        )
    
    def testnvs_dataloader(self):
        return DataLoader(
            self.testnvs_dataset,
            shuffle=False,
            num_workers=12,
            batch_size=1,  # validate one image (H*W rays) at a time
            pin_memory=True,
        )
    
    def load_frame_meta(self):
        dataset_name = self.config.dataset_name
        dataset_extra = self.config.dataset_extra
        if dataset_name in ["udc_dataset_toydesk", "udc_dataset_scannet_multi"]:
            data_json_path = os.path.join(
                dataset_extra.root_dir, f"transforms_full.json"
            )
            self.dataset_meta = read_json(data_json_path)
            # load fov
            self.fov_x_deg_dataset = self.dataset_meta["camera_angle_x"] * 180 / np.pi
            # print("fov x", self.fov_x_deg_dataset)
            # load poses
            self.poses = []
            tmp_index = []
            for frame in self.dataset_meta["frames"]:
                fix_rot = np.array([1, 0, 0, 0, -1, 0, 0, 0, -1]).reshape(3, 3)
                pose = np.array(frame["transform_matrix"])
                pose[:3, :3] = pose[:3, :3] @ fix_rot
                # centralize and rescale
                # pose = center_pose_from_avg(self.pose_avg, pose)
                # pose[:, 3] /= self.scale_factor
                self.poses.append(pose)
                tmp_index.append(frame["idx"])
            sorted_idx = np.argsort(np.array(tmp_index))
            self.poses = np.array(self.poses)[sorted_idx]
        else:
            assert False, "not implemented dataset type: {}".format(dataset_name)

    def training_step(self, batch, batch_nb):
        batch = batch['weighted']
        rays, rgbs = batch["rays"], batch["rgbs"]
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)

        # 0922 edit: just in training 
        instance_mask = batch["instance_mask"]
        instance_mask = instance_mask.squeeze() # (H*W,6)

        # get mask for psnr evaluation
        mask = batch["valid_mask"].view(-1, 1).repeat(1, 3)  # (H*W, 3)
        extra_info = dict()
        extra_info["is_eval"] = False
        extra_info["is_testnvs"] = False
        # extra_info["instance_mask"] = batch["instance_mask"]
        extra_info["pass_through_mask"] = batch["pass_through_mask"]
        extra_info["rays_in_bbox"] = getattr(
            self.train_dataset, "is_rays_in_bbox", lambda _: False
        )()
        extra_info["frustum_bound_th"] = (
            self.config.model.frustum_bound
            / self.config["dataset_extra"]["scale_factor"]
        )

        self.train_mode = True
        # results = self(rays, extra_info)
        results = self(rays, instance_mask, extra_info)
        loss_sum, loss_dict = self.loss(results, batch, self.current_epoch)

        with torch.no_grad():
            typ = "fine" if "rgb_fine" in results else "coarse"
            psnr___ = psnr(results[f"rgb_merged_{typ}"], rgbs, mask)

        self.log("lr", get_learning_rate(self.optimizer))
        self.log("train/loss", loss_sum)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v)
        self.log("train/psnr_merged", psnr___, prog_bar=True)

        return loss_sum

    def validation_step(self, batch, batch_nb):
        rays, rgbs = batch["rays"], batch["rgbs"]
        mask = None
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        extra_info = dict()
        extra_info["is_eval"] = True
        extra_info["is_testnvs"] = False
        extra_info["rays_in_bbox"] = getattr(
            self.val_dataset, "is_rays_in_bbox", lambda _: False
        )()
        extra_info["frustum_bound_th"] = (
            self.config.model.frustum_bound
            / self.config["dataset_extra"]["scale_factor"]
        )

        self.train_mode = False
        # results = self(rays, extra_info)
        results = self(rays, None, extra_info)

        loss_sum, loss_dict = self.loss(results, batch)
        for k, v in loss_dict.items():
            self.log(f"val/{k}", v)
        log = {"val_loss": loss_sum}
        log.update(loss_dict)
        typ = "fine" if "rgb_fine" in results else "coarse"

        stack_image = visualize_val_image_udc(
            self.config.img_wh, batch, results, typ=typ
        )
        self.logger.experiment.add_images(
            "val/GT_pred_unmasked_attn", stack_image, self.global_step # rename
        )

        psnr___ = psnr(results[f"rgb_merged_{typ}"], rgbs, mask)
        log["val_psnr_merged"] = psnr___

        return log


    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        mean_psnr_merged = torch.stack([x["val_psnr_merged"] for x in outputs]).mean()

        self.log("val/loss", mean_loss)
        self.log("val/psnr_merged", mean_psnr_merged, prog_bar=True)


    def testnvs_step(self, batch):
        for k, v in batch.items():
            batch[k] = v.cuda()

        rays, rgbs = batch["rays"], batch["rgbs"]
        # edit here only for scene branch
        mask = None
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        extra_info = dict()
        extra_info["is_eval"] = True
        extra_info["rays_in_bbox"] = getattr(
            self.testnvs_dataset, "is_rays_in_bbox", lambda _: False
        )()
        extra_info["frustum_bound_th"] = (
            self.config.model.frustum_bound
            / self.config["dataset_extra"]["scale_factor"]
        )

        extra_info["is_testnvs"] = True
        self.train_mode = False
        results = self(rays, None, extra_info)

        loss_sum, loss_dict = self.loss(results, batch)
        log = {"test_loss": loss_sum}
        log.update(loss_dict)
        typ = "fine" if "rgb_fine" in results else "coarse"

        stack_image = visualize_val_image_udc(
            self.config.img_wh, batch, results, typ=typ
        )

        rgbs = rgbs.reshape(self.img_wh[1],self.img_wh[0],3).permute([2,0,1]).unsqueeze(0)

        merged_mask = results[f"merged_masks_{typ}"].view(self.img_wh[1],self.img_wh[0], -1).permute(2, 0, 1).unsqueeze(0) # (1,6,h,w)
        gtt_mask = batch["instance_mask"][0].view(self.img_wh[1],self.img_wh[0], -1).long().argmax(dim=-1).unsqueeze(0) # (1,h,w)


        pred_merged = results[f"rgb_merged_{typ}"].reshape(self.img_wh[1],self.img_wh[0],3).permute([2,0,1]).unsqueeze(0)
        psnr___ = psnr(pred_merged, rgbs, mask)
        ssim___ = ssim(pred_merged, rgbs)
        lpips___ = self.loss_fn_vgg(pred_merged*2-1, rgbs*2-1)
        _, miou___ = miou(merged_mask, gtt_mask, self.config.model.inst_K)
        log["test_psnr_merged"] = psnr___
        log["test_ssim_merged"] = ssim___
        log["test_lpips_merged"] = lpips___
        log["test_iou_merged"] = miou___

        for k, v in log.items():
            if v == 0 or isinstance(v, float):
                log[k] = v
            else:
                log[k] = v.item()
        stack_image = stack_image.cpu()

        return log, stack_image



def main(config):

    setup_seed(2022)
    # seed_everything(2022)

    # exp_name = get_timestamp() + "_" + config.exp_name
    exp_name = config.exp_name
    print(f"Start with exp_name: {exp_name}.")
    log_path = f"logs/{exp_name}"
    config["log_path"] = log_path

    system = UDCNeRFSystem(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=log_path,
        filename="{epoch:d}",
        monitor="val/psnr_merged",
        mode="max",
        # save_top_k=5,
        save_top_k=3, 
        # save_top_k=-1,
        save_last=True,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )

    logger = TensorBoardLogger(save_dir="logs", name=exp_name)

    trainer = Trainer(
        max_epochs=config.train.num_epochs,
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=config.ckpt_path,
        logger=logger,
        enable_model_summary=False,
        gpus=config.train.num_gpus,
        accelerator="ddp" if config.train.num_gpus > 1 else None,
        num_sanity_val_steps=1,
        benchmark=True,
        profiler="simple" if config.train.num_gpus == 1 else None,
        val_check_interval=1.0, 
        limit_train_batches=config.train.limit_train_batches,
        replace_sampler_ddp=False,
    )

    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
        make_source_code_snapshot(log_path)
    else:
        make_source_code_snapshot(log_path)
    OmegaConf.save(config=config, f=os.path.join(log_path, "run_config_snapshot.yaml"))
    trainer.fit(system)


if __name__ == "__main__":
    conf_cli = OmegaConf.from_cli()
    # print(conf_cli.dataset_config)
    conf_dataset = OmegaConf.load(conf_cli.dataset_config)
    conf_default = OmegaConf.load("config/default_conf.yml")
    # merge conf with the priority
    conf_merged = OmegaConf.merge(conf_default, conf_dataset, conf_cli)

    print("-" * 40)
    print(OmegaConf.to_yaml(conf_merged))
    print("-" * 40)

    main(config=conf_merged)
