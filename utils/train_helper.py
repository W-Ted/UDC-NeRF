import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T



def visualize_depth(depth, cmap=cv2.COLORMAP_JET, vmin=None, vmax=None):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x)  # change nan to 0
    mi = np.min(x) if vmin == None else vmin  # get minimum depth
    ma = np.max(x) if vmax == None else vmax
    x = np.clip(x, mi, ma)
    x = (x - mi) / max(ma - mi, 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_


def visualize_val_image_udc(img_wh, batch, results, typ="fine"):

    W, H = img_wh
    rgbs = batch["rgbs"]
    img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)

    rgbs_ps = batch["rgbs_ps"]
    img_gt_ps = rgbs_ps.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
        
    # img_full = results[f"rgb_{typ}"].view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
    # img_instance = results[f"rgb_instance_{typ}"].view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
    img_merged = results[f"rgb_merged_{typ}"].view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)

    # unmasked rgb map
    unmasked_imgs = []
    rgb_map_unmasked_ = results[f"rgb_unmasked_{typ}"] # hw,3,K
    for k in range(rgb_map_unmasked_.shape[-1]):
        unmasked_img = rgb_map_unmasked_[:,:,k].view(H,W,3).permute(2,0,1).cpu() # 3,h,w
        unmasked_imgs += [unmasked_img]
    
    # unmasked opacity map
    unmasked_opacitys = []
    weights_sum_unmasked_ = results[f"opacity_unmasked_{typ}"] # hw,K
    for k in range(weights_sum_unmasked_.shape[-1]):
        unmasked_opacity = weights_sum_unmasked_[:,k].view(H,W).cpu()
        unmasked_opacitys += [visualize_depth(unmasked_opacity,vmin=0,vmax=1)]

    # mask1
    gt_instance_masks =[]
    gt_unmasked_imgs = []
    mask1 = batch["instance_mask"][0] # hwx6
    for k in range(mask1.shape[1]):
        mask = mask1[:,k].view(H,W).cpu()
        gt_instance_masks += [visualize_depth(mask,vmin=0,vmax=1)]
        gt_unmasked_img = img_gt * mask
        gt_unmasked_imgs += [gt_unmasked_img]

    depth_unmasked_imgs =[]
    mask1 = results[f"depth_unmasked_{typ}"] # hwx6

    for k in range(mask1.shape[1]):
        mask = mask1[:,k].view(H,W).cpu()
        depth_unmasked_imgs += [visualize_depth(mask,vmin=0,vmax=1)]

    stack0 = torch.stack(
        [img_gt, img_gt_ps] + gt_instance_masks + gt_unmasked_imgs + gt_instance_masks
    )  # (19, 3, H, W)

    stack1 = torch.stack(
        [img_merged, unmasked_imgs[0]] + depth_unmasked_imgs + unmasked_imgs + unmasked_opacitys
    )  # (19, 3, H, W)
    stack = torch.cat([stack0,stack1],2)
    return stack


