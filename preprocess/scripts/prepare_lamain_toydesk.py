import shutil
import numpy as np
import os
import cv2
from tqdm import trange, tqdm
from scipy import ndimage



for scene_idx in ['1', '2']:
    # paths
    erode = '23x23'
    root = f'../../object_nerf/data/toy_desk/our_desk_{scene_idx}/full'
    save_root = f'./preprocess/toydesk{scene_idx}_lamain'
    os.makedirs(save_root, exist_ok=True)
    nums = [int(i.split('.')[0]) for i in sorted(os.listdir(root)) if ('instance' not in i) and ('depth' not in i)]
    num_frames = len(nums)

    # get K 
    K = 0
    for num in tqdm(nums):
        idx = '%04d'%num
        img_path = os.path.join(root, f"{idx}.png")

        # read instance mask
        instance_path = os.path.join(root, f"{idx}.instance.png")
        instance_mask = cv2.imread(instance_path, cv2.IMREAD_ANYDEPTH)
        if instance_mask.max() > K:
            K = instance_mask.max()
    print('%d frames, %d categories. '%(num_frames, K))



    print('Start masking, saved in %s'%save_root)
    for num in tqdm(nums):
        idx = '%04d'%num
        img_path = os.path.join(root, f"{idx}.png")

        # read instance mask
        instance_path = os.path.join(root, f"{idx}.instance.png")
        instance_mask = cv2.imread(instance_path, cv2.IMREAD_ANYDEPTH)

        for label in range(K):
            if label == 0:
                mask = instance_mask != label
            else:
                mask = instance_mask == label
            
            if erode is not None:
                mask = ndimage.binary_erosion(mask, structure=np.ones((3,3))).astype(mask.dtype)
                mask = ndimage.binary_dilation(mask, structure=np.ones((23,23))).astype(mask.dtype)
            
            mask = 255* mask.astype(float)
            
            label = '%04d'%label
            img_name = os.path.join(save_root, f"{idx}_{label}.png")
            mask_name = os.path.join(save_root, f"{idx}_{label}_mask.png")

            cv2.imwrite(mask_name, mask)
            shutil.copy(img_path, img_name)
        