import torch
# from kornia.losses import ssim as dssim
from kornia.losses import ssim_loss


def mse(image_pred, image_gt, valid_mask=None, reduction="mean"):
    value = (image_pred - image_gt) ** 2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == "mean":
        return torch.mean(value)
    return value


def psnr(image_pred, image_gt, valid_mask=None, reduction="mean"):
    # print(image_pred.shape, image_gt.shape)
    return -10 * torch.log10(mse(image_pred, image_gt, valid_mask, reduction))


def ssim(image_pred, image_gt, reduction="mean"):
    """
    image_pred and image_gt: (1, 3, H, W)
    """
    # dssim_ = dssim(image_pred, image_gt, 3, reduction)  # dissimilarity in [0, 1]
    dssim_ = ssim_loss(image_pred, image_gt, 5)  # dissimilarity in [0, 1]
    return 1 - 2 * dssim_  # in [-1, 1]



def miou(seg_pred, seg_gt, nclass):

    total_inter = torch.zeros(nclass)
    total_union = torch.zeros(nclass)
    total_correct = 0
    total_label = 0

    correct, labeled = batch_pix_accuracy(seg_pred, seg_gt)
    inter, union = batch_intersection_union(seg_pred, seg_gt, nclass)

    total_correct += correct
    total_label += labeled
    if total_inter.device != inter.device:
        total_inter = total_inter.to(inter.device)
        total_union = total_union.to(union.device)
    total_inter += inter
    total_union += union
    
    pixAcc = 1.0 * total_correct / (2.220446049250313e-16 + total_label)  # remove np.spacing(1)
    IoU = 1.0 * total_inter / (2.220446049250313e-16 + total_union)
    mIoU = IoU.mean().item()
    return pixAcc, mIoU


# pytorch version
def batch_pix_accuracy(output, target):
    """PixAcc"""
    # inputs are numpy array, output 4D, target 3D
    predict = torch.argmax(output.long(), 1) + 1
    target = target.long() + 1

    pixel_labeled = torch.sum(target > 0).item()
    try:
        pixel_correct = torch.sum((predict == target) * (target > 0)).item()
    except:
        print("predict size: {}, target size: {}, ".format(predict.size(), target.size()))
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled
    

def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are numpy array, output 4D, target 3D
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1  # [N,H,W] 
    target = target.float() + 1            # [N,H,W] 
    # print(predict.shape)
    # print(target.shape)

    predict = predict.float() * (target > 0).float()
    intersection = predict * (predict == target).float()
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi)
    area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi)
    area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi)
    area_union = area_pred + area_lab - area_inter
    assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
    return area_inter.float(), area_union.float()