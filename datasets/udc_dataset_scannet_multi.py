import sys

sys.path.append(".")

import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
import cv2
from PIL import Image
from torchvision import transforms as T

from datasets.ray_utils import get_ray_directions, get_rays
from datasets.geo_utils import (
    bbox_intersection_batch,
    center_pose_from_avg,
    observe_angle_distance,
)
from datasets.image_utils import compute_distance_transfrom_weights, rebalance_mask
import random

from pytorch_lightning import seed_everything

from scipy import ndimage


class ScannetMultiDataset(Dataset):
    def __init__(self, split="train", img_wh=(640, 480), dataset_extra=None):
        self.split = split
        self.img_wh = img_wh
        self.define_transforms() # to tensor

        # load dataset configuration
        self.conf = dataset_extra
        self.root_dir = self.conf["root_dir"] # data/toy_desk/our_desk_2 
        self.scene_id = self.conf.get("scene_id", "") # 
        self.scale_factor = self.conf["scale_factor"] # 16.0
        self.near = self.conf["near"] # 0.8
        self.far = self.conf["far"]   # 24

        self.ps_type = self.conf["ps_type"]
        self.ps_root = self.conf["ps_root"]

        seed_everything(2022)

        # use scene center to normalize poses
        self.pose_avg = np.concatenate(
            [np.eye(3), np.array(self.conf["scene_center"])[:, None]], 1
        )

        # remove black border caused by image undistortion
        border = 20
        w, h = self.img_wh # (640, 480)
        bmask = np.ones((h, w))
        bmask[:border, :] = 0
        bmask[-border:, :] = 0
        bmask[:, :border] = 0
        bmask[:, -border:] = 0
        self.bmask = self.transform(bmask).bool()

        # load bbox info
        self.use_bbox = self.conf["use_bbox"] # false
        if self.use_bbox:
            self.read_bbox_info()

        self.read_meta()
        self.white_back = False # false

        if self.split == "train":
            print("-" * 40)

    def read_bbox_info(self):
        # read axis_align_matrix
        scene_info_file = os.path.join(
            self.conf["scans_dir"], "{}/{}.txt".format(self.scene_id, self.scene_id)
        )
        lines = open(scene_info_file).readlines()
        for line in lines:
            if "axisAlignment" in line:
                axis_align_matrix = [
                    float(x) for x in line.rstrip().strip("axisAlignment = ").split(" ")
                ]
                break
        self.axis_align_mat = np.array(axis_align_matrix).reshape(4, 4)

        # read bbox bounds
        scene_bbox = np.load(
            os.path.join(self.conf["bbox_dir"], "{}_bbox.npy".format(self.scene_id))
        )
        for b in scene_bbox:
            if b[6] != self.conf["val_instance_id"]:
                continue
            length = np.array([b[3], b[4], b[5]]) * 0.5
            center = np.array([b[0], b[1], b[2]])
            self.bbox_bounds = np.array([center - length, center + length])

    def transform_rays_to_bbox_coordinates(self, rays_o, rays_d):
        if type(rays_o) is torch.Tensor:
            rays_o, rays_d = rays_o.numpy(), rays_d.numpy()
        # unscale
        rays_o = rays_o * self.scale_factor
        # de-centralize
        T_orig_avg = self.pose_avg.squeeze()
        rays_o_bbox = (T_orig_avg[:3, :3] @ rays_o.T).T + T_orig_avg[:3, 3]
        rays_d_bbox = (T_orig_avg[:3, :3] @ rays_d.T).T
        # convert to bbox coordinates
        T_box_orig = self.axis_align_mat
        rays_o_bbox = (T_box_orig[:3, :3] @ rays_o_bbox.T).T + T_box_orig[:3, 3]
        rays_d_bbox = (T_box_orig[:3, :3] @ rays_d.T).T
        return rays_o_bbox, rays_d_bbox

    def get_instance_mask(self, instance_path, instance_id):
        instance = cv2.resize(
            cv2.imread(instance_path, cv2.IMREAD_ANYDEPTH),
            self.img_wh,
            interpolation=cv2.INTER_NEAREST,
        )

        if isinstance(instance_id, int):
            # get 0-1 mask according to the instance id.

            # scannet0113: 
            # assert len(self.instance_ids) == 3 and self.instance_ids[0] == 0 and self.instance_ids[1] == 4 and self.instance_ids[2] == 6
            
            if len(self.instance_ids) == 3: # scannet0113_multi 
                if instance_id == 0:
                    mask = np.logical_and(
                        instance != self.instance_ids[1],
                        instance != self.instance_ids[2] 
                    )
                elif instance_id == self.instance_ids[1]:
                    mask = instance == self.instance_ids[1]
                elif instance_id == self.instance_ids[2]:
                    mask = instance == self.instance_ids[2]
            
            elif len(self.instance_ids) == 2:
                assert self.instance_ids[0] == 0

                if instance_id == 0:
                    mask = instance != self.instance_ids[1]
                    if self.conf["erode_instance_mask"]:
                        mask = ndimage.binary_erosion(mask, structure=np.ones((9,9))).astype(mask.dtype)
                    
                elif instance_id != 0:
                    # mask = instance == self.instance_ids[1]
                    mask = instance != self.instance_ids[1]
                    if self.conf["erode_instance_mask"]:
                        mask = ndimage.binary_erosion(mask, structure=np.ones((9,9))).astype(mask.dtype)
                    mask = np.logical_not(mask)


        elif isinstance(instance_id, list):
            # get 0-1 mask according to a list of instance ids. 
            mask = np.zeros_like(instance).astype(bool)
            for id in instance_id:
                mask = np.logical_or(mask, instance == id)
        return mask

    def read_meta(self):
        # Step 0. read json files for training and testing list
        data_json_path = os.path.join(
            self.root_dir,
            "transforms_{}.json".format(
                "full"
            ),
        ) # data/toy_desk/our_desk_2/transforms_full.json
        with open(data_json_path, "r") as f:
            self.meta = json.load(f)
        f.close()

        # Step 1. generate rays for each image in camera coordinate
        w, h = self.img_wh
        self.focal = (
            0.5 * w / np.tan(0.5 * self.meta["camera_angle_x"])
        )  # original focal length
        # when W=800

        self.focal *= (
            self.img_wh[0] / w
        )  # modify focal length to match size self.img_wh

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, self.focal)  # (h, w, 3)

        self.direction_orig_norm = torch.norm(self.directions, dim=-1, keepdim=True)

        # Step 2. filter image list via preset parameters and observation check
        train_start_idx = self.conf["train_start_idx"] # 0
        validate_idx = self.conf["validate_idx"]       # 131
        train_skip_step = self.conf["train_skip_step"] # 1
        train_max_size = self.conf["train_max_size"]   # 9999

        if self.split == "train":
            # only retain train split
            split_inds = np.loadtxt(
                os.path.join(self.conf["split"], "train_list_near_obj.txt")
            ).tolist()
            print("Training split count", len(split_inds))
            self.meta["frames"] = list(
                filter(lambda x: (x["idx"] in split_inds), self.meta["frames"])
            )
            # remove according to preset index
            self.meta["frames"] = list(
                filter(
                    lambda x: (
                        x["idx"] >= train_start_idx and x["idx"] != validate_idx
                    ),
                    self.meta["frames"],
                )
            )

            # remove via observation check
            def obs_check(f):
                T = np.array(f["transform_matrix"])
                if np.isnan(T.sum()) or np.isinf(T.sum()):  # remove invalid pose
                    return False
                if not self.conf["enable_observation_check"]:
                    return True
                angle, dist = observe_angle_distance(T, self.pose_avg[:3, 3])
                return (
                    angle < self.conf["max_obs_angle"]
                    and dist < self.conf["max_obs_distance"]
                )

            self.meta["frames"] = list(filter(obs_check, self.meta["frames"]))
            # no frames removed for toydesk2

            # skip frames: train_skip_step=1 for toydesk
            self.meta["frames"] = [
                self.meta["frames"][i]
                for i in np.arange(0, len(self.meta["frames"]), train_skip_step)
            ]

            # set max train size: not work for toydesk
            train_max_size = min(train_max_size, len(self.meta["frames"]))
            self.meta["frames"] = self.meta["frames"][:train_max_size]
            frames = self.meta["frames"]
            print(
                "Train idx: {} -> {}, skip: {}".format(
                    frames[0]["idx"], frames[-1]["idx"], train_skip_step
                )
            )

        elif self.split == "val":
            # we only set one frame for valid
            self.meta["frames"] = list(
                filter(lambda x: (x["idx"] == validate_idx), self.meta["frames"])
            )
            print("Valid idx: {}".format(validate_idx))
        

        elif self.split == "test_nvs":
            # only retain test split
            split_inds = np.loadtxt(
                os.path.join(self.conf["split"], "test_list_near_obj.txt")
            ).tolist()
            print('split ind: ', split_inds)
            self.meta["frames"] = list(
                filter(lambda x: (x["idx"] in split_inds), self.meta["frames"])
            ) #  images here
            frames = self.meta["frames"]


        # Step 4. create buffer of all rays and rgb data
        self.bg_instance_ids = list(self.conf.get("bg_instance_id", [])) # [0]

        if self.split == "train":
            self.image_paths = []
            self.all_rays = []
            self.all_rgbs = []
            self.all_rgbs_ps = []
            self.all_depths = []
            self.all_valid_masks = []
            self.all_instance_masks = []
            self.all_instance_masks_weight = []
            self.all_pass_through_masks = []
            self.all_frame_indices = []
            self.all_instance_ids = []

            self.instance_ids = self.conf["instance_id"] # [5,4,2,1,3]
            image_idx_set = set()

            for idx, frame in enumerate(self.meta["frames"]):
                curr_frame_instance_masks = []
                curr_frame_instance_masks_weight = []
                curr_frame_pass_through_masks = []
                curr_frame_instance_ids = []
                for i_inst, instance_id in enumerate(self.instance_ids):
                    # 0 5
                    # 1 4
                    # 2 2
                    # 3 1
                    # 4 3
                    print(
                        "\rRead meta {:05d} : {:05d} instance {:01d}".format(
                            idx, len(self.meta["frames"]) - 1, instance_id
                        ),
                        end="",
                    )
                    # only get rgbs/rays for the first one, 
                    # for others, only return instance masks...
                    sample = self.read_frame_data(
                        frame, instance_id, read_instance_only=(i_inst != 0) # false, then truex4
                    )
                    # skip duplicates, not work for toydesk2
                    if instance_id == 0 and idx in image_idx_set:
                        curr_frame_instance_masks += [sample["instance_mask"]]
                        curr_frame_instance_masks_weight += [sample["instance_mask_weight"]]
                        curr_frame_pass_through_masks += [sample["pass_through_mask"]]
                        curr_frame_instance_ids += [sample["instance_ids"]]
                        print("continue for bgs")
                        continue
                    if sample is None:
                        print(
                            "\nSkip frame {} with instance {}".format(idx, instance_id)
                        )
                        continue
                    image_idx_set.add(idx)
                    # only save for the first
                    if i_inst == 0:
                        self.all_rays += [sample["rays"]]
                        self.all_rgbs += [sample["rgbs"]]
                        self.all_rgbs_ps += [sample["rgbs_ps"]]
                        self.all_depths += [sample["depths"]]
                        self.all_valid_masks += [sample["valid_mask"]]
                        self.all_frame_indices += [
                            torch.ones_like(sample["valid_mask"]) * idx
                        ]

                    curr_frame_instance_masks += [sample["instance_mask"]]
                    curr_frame_instance_masks_weight += [sample["instance_mask_weight"]]
                    curr_frame_pass_through_masks += [sample["pass_through_mask"]]
                    curr_frame_instance_ids += [sample["instance_ids"]]

                self.all_instance_masks += [torch.stack(curr_frame_instance_masks, -1)]
                self.all_instance_masks_weight += [
                    torch.stack(curr_frame_instance_masks_weight, -1)
                ]
                self.all_pass_through_masks += [
                    torch.stack(curr_frame_pass_through_masks, -1)
                ]
                self.all_instance_ids += [torch.stack(curr_frame_instance_ids, -1)]

            print("")
            self.all_rays = torch.cat(
                self.all_rays, 0
            )  # (len(self.meta['frames])*h*w, 8)
            self.all_rgbs = torch.cat(
                self.all_rgbs, 0
            )  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs_ps = torch.cat(
                self.all_rgbs_ps, 0
            )  # (len(self.meta['frames])*h*w, 3)
            self.all_depths = torch.cat(
                self.all_depths, 0
            )  # (len(self.meta['frames])*h*w)
            self.all_valid_masks = torch.cat(
                self.all_valid_masks, 0
            )  # (len(self.meta['frames])*h*w)
            self.all_instance_masks = torch.cat(
                self.all_instance_masks, 0
            )  # (len(self.meta['frames])*h*w, 5)
            self.all_instance_masks_weight = torch.cat(
                self.all_instance_masks_weight, 0
            )  # (len(self.meta['frames])*h*w, 5)
            self.all_pass_through_masks = torch.cat(
                self.all_pass_through_masks, 0
            )  # (len(self.meta['frames])*h*w, 5)
            self.all_frame_indices = torch.cat(
                self.all_frame_indices, 0
            ).long()  # (len(self.meta['frames])*h*w)
            self.all_instance_ids = torch.cat(
                self.all_instance_ids, 0
            ).long()  # (len(self.meta['frames])*h*w, 5)

            # print('[INFO] rays, rgbs ', self.all_rays.shape, self.all_rgbs.shape, self.all_rgbs_ps.shape) # 
            # print('[INFO] depth, masks ', self.all_depths.shape, self.all_valid_masks.shape)
            # print('[INFO] instance_mask, weights', self.all_instance_masks.shape, self.all_instance_masks_weight.shape)
            # print('[INFO] pass_through_mask ', self.all_pass_through_masks.shape)
            # print('[INFO] frame_indices, instance_ids ', self.all_frame_indices.shape, self.all_instance_ids.shape)

        elif self.split == 'val' or self.split == 'test_nvs':
            self.instance_ids = self.conf["instance_id"]
            self.image_paths = []
            self.all_rgbs = []
            self.all_rgbs_ps = []
            self.all_valid_masks = []
            self.all_instance_masks = []
            self.all_instance_masks_weight = []

            self.instance_ids = self.conf["instance_id"] # [5,4,2,1,3]
            image_idx_set = set()

            for idx, frame in enumerate(self.meta["frames"]):
                curr_frame_instance_masks = []
                curr_frame_instance_masks_weight = []
                for i_inst, instance_id in enumerate(self.instance_ids):
                    # 0 5
                    # 1 4
                    # 2 2
                    # 3 1
                    # 4 3
                    print(
                        "\rRead meta {:05d} : {:05d} instance {:01d}".format(
                            idx, len(self.meta["frames"]) - 1, instance_id
                        ),
                        end="",
                    )
                    # only get rgbs/rays for the first one, 
                    # for others, only return instance masks...
                    sample = self.read_frame_data(
                        frame, instance_id, read_instance_only=(i_inst != 0) # false, then truex4
                    )
                    # skip duplicates, not work for toydesk2
                    if instance_id == 0 and idx in image_idx_set:
                        curr_frame_instance_masks += [sample["instance_mask"]]
                        curr_frame_instance_masks_weight += [sample["instance_mask_weight"]]
                        print("continue for bgs")
                        continue
                    if sample is None:
                        print(
                            "\nSkip frame {} with instance {}".format(idx, instance_id)
                        )
                        continue
                    image_idx_set.add(idx)
                    # only save for the first
                    if i_inst == 0:
                        self.all_rgbs += [sample["rgbs"]]
                        self.all_rgbs_ps += [sample["rgbs_ps"]]
                        self.all_valid_masks += [sample["valid_mask"]]

                    curr_frame_instance_masks += [sample["instance_mask"]]
                    curr_frame_instance_masks_weight += [sample["instance_mask_weight"]]

                self.all_instance_masks += [torch.stack(curr_frame_instance_masks, -1)]
                self.all_instance_masks_weight += [
                    torch.stack(curr_frame_instance_masks_weight, -1)
                ]

            print("")
            self.all_rgbs = torch.cat(
                self.all_rgbs, 0
            )  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs_ps = torch.cat(
                self.all_rgbs_ps, 0
            )  # (len(self.meta['frames])*h*w, 3)
            self.all_valid_masks = torch.cat(
                self.all_valid_masks, 0
            )  # (len(self.meta['frames])*h*w)
            self.all_instance_masks = torch.cat(
                self.all_instance_masks, 0
            )  # (len(self.meta['frames])*h*w, 5)
            self.all_instance_masks_weight = torch.cat(
                self.all_instance_masks_weight, 0
            )  # (len(self.meta['frames])*h*w, 5)
            # print('[INFO] rays, rgbs ', self.all_rgbs.shape, self.all_rgbs_ps.shape) # 
            # print('[INFO] depth, masks ', self.all_valid_masks.shape)
            # print('[INFO] instance_mask, weights', self.all_instance_masks.shape, self.all_instance_masks_weight.shape)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def read_frame_data(self, frame, instance_id, read_instance_only=False):

        valid_mask = self.bmask.flatten()  # (h*w) valid_mask

        # read instance mask
        if self.conf["use_instance_mask"]:
            instance_path = os.path.join(
                self.root_dir, f"{frame['file_path']}.{self.conf['inst_seg_tag']}.png"
            )
            instance_mask = self.get_instance_mask(instance_path, instance_id)
            if self.conf.mask_rebalance_strategy == "fg_bg_reweight":
                instance_mask_weight = rebalance_mask(
                    instance_mask,
                    fg_weight=self.conf.fg_weight,
                    bg_weight=self.conf.bg_weight,
                )
            elif self.conf.mask_rebalance_strategy == "distance_transform":
                instance_mask_weight = compute_distance_transfrom_weights(
                    instance_mask,
                    uncertain_pixel_distance=0.05 * self.img_wh[0],
                    fg_bg_balance_weight=True,
                    fg_weight=self.conf.fg_weight,
                    bg_weight=self.conf.bg_weight,
                )
            # instance_mask, uncertain_pixel_distance=0.05*self.img_wh[0], fg_bg_balance_weight=False)
            instance_mask, instance_mask_weight = self.transform(instance_mask).view(
                -1
            ), self.transform(instance_mask_weight).view(-1)
            # load pass_through mask
            pass_through_mask = self.get_instance_mask(
                instance_path, self.bg_instance_ids + [instance_id]
            )
            pass_through_mask = self.transform(pass_through_mask).view(-1)
        else:
            instance_mask = torch.ones_like(valid_mask).bool()
            instance_mask_weight = torch.zeros_like(valid_mask)
            pass_through_mask = instance_mask.clone()

        if read_instance_only:
            return {
                "instance_mask": instance_mask,
                "instance_mask_weight": instance_mask_weight,
                "instance_ids": torch.ones_like(instance_mask).long() * instance_id,
                "pass_through_mask": pass_through_mask,
            }

        # Original poses has rotation in form "right down forward", change to NDC "right up back"
        fix_rot = np.array([1, 0, 0, 0, -1, 0, 0, 0, -1]).reshape(3, 3)
        pose = np.array(frame["transform_matrix"])
        pose[:3, :3] = pose[:3, :3] @ fix_rot

        # centralize and rescale
        pose = center_pose_from_avg(self.pose_avg, pose)
        pose[:, 3] /= self.scale_factor

        c2w = torch.FloatTensor(pose)[:3, :4]

        img_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
        if not os.path.exists(img_path):
            print("Skip file which does not exist", img_path)
            return None

        img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
        img = img.resize(self.img_wh, Image.LANCZOS)
        img = self.transform(img)  # (3, H, W)
        img = img.view(3, -1).permute(1, 0)  # (H*W, 3) RGB
        img_ps = Image.open(os.path.join(self.ps_root, f"{frame['file_path']}.jpg"))
        img_ps = img_ps.resize(self.img_wh, Image.LANCZOS)
        img_ps = self.transform(img_ps)  # (3, H, W)
        if self.ps_type == 'not_normal':
            img_ps = img_ps[[2,1,0],:,:]
        img_ps = img_ps.view(3, -1).permute(1, 0)  # (H*W, 3) RGB

        depth = cv2.imread(
            os.path.join(self.root_dir, f"{frame['file_path']}.depth.png"),
            cv2.IMREAD_ANYDEPTH,
        )
        if depth is None:
            depth = np.zeros((self.img_wh[1], self.img_wh[0]))
        else:
            depth = (
                cv2.resize(depth, self.img_wh, interpolation=cv2.INTER_NEAREST) * 1e-3
            )
            depth[depth > 4] = 0
        depth = self.transform(depth).float().squeeze()  # (H, W)
        depth = depth.view(-1)  # (H*W)
        depth /= self.scale_factor
        depth *= self.direction_orig_norm.view(-1)

        rays_o, rays_d = get_rays(self.directions, c2w)

        # determine whether we constraint rays in bbox
        ray_in_bbox = self.use_bbox
        if self.conf["use_bbox_only_for_test"] and self.split == "train":
            ray_in_bbox = False

        # near and far fit to box
        if ray_in_bbox:
            rays_o_bbox, rays_d_bbox = self.transform_rays_to_bbox_coordinates(
                rays_o, rays_d
            )
            bbox_mask, batch_near, batch_far = bbox_intersection_batch(
                self.bbox_bounds, rays_o_bbox, rays_d_bbox
            )
            bbox_mask, batch_near, batch_far = (
                torch.Tensor(bbox_mask).bool(),
                torch.Tensor(batch_near[..., None]),
                torch.Tensor(batch_far[..., None]),
            )
            # scale to fit
            batch_near, batch_far = (
                batch_near / self.scale_factor,
                batch_far / self.scale_factor,
            )
            # we also mark the parts which does not intersect with bbox as invalid
            if self.conf["use_instance_mask"]:
                instance_mask = instance_mask * bbox_mask
            # determine if we need to inhibit background
            if self.conf["use_bbox_only_for_test"]:
                # for the invalid part, we use 0 as near far
                batch_near[~bbox_mask] = torch.zeros_like(batch_near[~bbox_mask])
                batch_far[~bbox_mask] = torch.zeros_like(batch_far[~bbox_mask])
            else:
                # for the invalid part, we use default near far
                # when we have 3d mask, the other rays will be suppressed by occupancy mask
                batch_near[~bbox_mask] = (
                    self.near
                    / self.scale_factor
                    * torch.ones_like(batch_near[~bbox_mask])
                )
                batch_far[~bbox_mask] = (
                    self.far
                    / self.scale_factor
                    * torch.ones_like(batch_far[~bbox_mask])
                )
        else:
            batch_near = self.near / self.scale_factor * torch.ones_like(rays_o[:, :1])
            batch_far = self.far / self.scale_factor * torch.ones_like(rays_o[:, :1])

        rays = torch.cat([rays_o, rays_d, batch_near, batch_far], 1)  # (H*W, 8)

        return {
            "rays": rays,
            "rgbs": img,
            "rgbs_ps": img_ps,
            "depths": depth,
            "c2w": c2w,
            "valid_mask": valid_mask,
            "instance_mask": instance_mask,
            "instance_mask_weight": instance_mask_weight,
            "instance_ids": torch.ones_like(depth).long() * instance_id,
            "pass_through_mask": pass_through_mask,
        }

    def is_rays_in_bbox(self):
        ray_in_bbox = self.use_bbox
        if self.conf["use_bbox_only_for_test"] and self.split == "train":
            ray_in_bbox = False
        return ray_in_bbox

    def __len__(self):
        if self.split == "train":
            return len(self.all_rays)
        if self.split == "val":
            return 4
        return len(self.meta["frames"])

    def __getitem__(self, idx):
        if self.split == "train":  # use data in the buffers
            rand_instance_id = torch.randint(0, len(self.instance_ids), (1,))
            sample = {
                "rays": self.all_rays[idx],
                "rgbs": self.all_rgbs[idx],
                "rgbs_ps": self.all_rgbs_ps[idx],
                "depths": self.all_depths[idx],
                "valid_mask": self.all_valid_masks[idx],
                "instance_mask": self.all_instance_masks[idx, :],
                "instance_mask_weight": self.all_instance_masks_weight[idx, :],
                "frame_idx": self.all_frame_indices[idx],
                "instance_ids": self.all_instance_ids[idx, :],
                "pass_through_mask": self.all_pass_through_masks[idx, :],
            }

        else:  # create data for each image separately
            if self.split == 'val':
                frame = self.meta["frames"][0]
            elif self.split == 'test_nvs':
                frame = self.meta["frames"][idx]
            
            curr_frame_instance_masks = []
            curr_frame_instance_masks_weight = []
            curr_frame_pass_through_masks = []
            curr_frame_instance_ids = []
            for i_inst, instance_id in enumerate(self.instance_ids):
                sample = self.read_frame_data(frame, instance_id, read_instance_only=(i_inst != len(self.instance_ids)-1))
                assert (
                    not sample is None
                ), "val image does not have enough areas for val_instance_id"
                curr_frame_instance_masks += [sample["instance_mask"]]
                curr_frame_instance_masks_weight += [sample["instance_mask_weight"]]
                curr_frame_pass_through_masks += [sample["pass_through_mask"]]
                curr_frame_instance_ids += [sample["instance_ids"]]
            sample["instance_mask"] = torch.stack(curr_frame_instance_masks, -1)
            sample["instance_mask_weight"] = torch.stack(curr_frame_instance_masks_weight, -1)
            sample["pass_through_mask"] = torch.stack(curr_frame_pass_through_masks, -1)
            sample["instance_ids"] = torch.stack(curr_frame_instance_ids, -1)

            sample["frame_idx"] = frame["idx"]

        return sample
