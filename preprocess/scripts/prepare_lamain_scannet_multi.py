import shutil
import numpy as np
import os
import cv2
from tqdm import trange, tqdm
from scipy import ndimage
import shutil

erode = '23x23'
FG_IDS = {
    # '0024': [9],
    # '0038': [3],
    '0113': [4,6],
    # '0192': [3]
}
# scene_idxs = sorted(list(FG_IDS.keys()))
scene_idxs = ['0113']

for scene_idx in scene_idxs:

    root = f'../../object_nerf/data/scannet_prepare/processed/processed_scannet_{scene_idx}_00/full'
    save_root = f'./preprocess/scannet{scene_idx}_multi_lamain/'
    os.makedirs(save_root, exist_ok=True)

    # K = 2 by default
    nums = [int(i.split('.')[0]) for i in sorted(os.listdir(root)) if ('instance' not in i) and ('depth' not in i)]
    num_frames = len(nums)
    # fg = FG_IDS[scene_idx][0]
    fg = FG_IDS[scene_idx]

    print('Start masking, saved in %s'%save_root)
    for num in tqdm(nums):
        idx = '%d'%num
        img_path = os.path.join(root, f"{idx}.png")

        # read instance mask
        instance_path = os.path.join(root, f"{idx}.instance-filt.png")
        assert os.path.exists(instance_path), instance_path
        instance_mask = cv2.resize( # 1296x968 -> 640x480
                    cv2.imread(instance_path, cv2.IMREAD_ANYDEPTH),
                    [640,480],
                    interpolation=cv2.INTER_NEAREST,
                )
        
        for label in range(2):
            # if label == 0:
            #     mask = instance_mask != fg
            # elif label == 1:
            #     mask = instance_mask == fg
            if label != 0:
                continue
            else:
                mask = np.logical_or(instance_mask == fg[0], instance_mask == fg[1])
            
            if erode is not None:
                mask = ndimage.binary_erosion(mask, structure=np.ones((3,3))).astype(mask.dtype)
                mask = ndimage.binary_dilation(mask, structure=np.ones((23,23))).astype(mask.dtype)
            mask = 255* mask.astype(float)
            
            label = '%04d'%label
            img_name = os.path.join(save_root, f"{idx}_{label}.png")
            mask_name = os.path.join(save_root, f"{idx}_{label}_mask.png")
            
            cv2.imwrite(mask_name, mask)
            shutil.copy(img_path, img_name)