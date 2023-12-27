import os

os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa

from utils.util import get_timestamp
import torch

from omegaconf import OmegaConf

from train_udc import UDCNeRFSystem
from tqdm import tqdm
import numpy as np
import cv2
from torchvision import transforms

# tensor to cv2 image
def tensor_to_cv2(tensor):
    tensor = torch.clamp(tensor, min=0, max=1)
    transform_list = [transforms.Normalize((0, 0, 0), (1, 1, 1)), transforms.ToPILImage()]
    transform = transforms.Compose(transform_list)
    img = transform(tensor)
    # img.save('fake_pil.jpg')
    # Convert RGB to BGR 
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img


def main(config):
    render_path = f"debug/rendered_view/render_{get_timestamp()}_{config.exp_name}/"
    os.makedirs(render_path, exist_ok=True)
    print('Write into ', render_path)

    os.makedirs(os.path.join(render_path, 'preview'), exist_ok=True)
    os.makedirs(os.path.join(render_path, 'pred'), exist_ok=True)

    system = UDCNeRFSystem(config)

    checkpoint = torch.load(config.ckpt_path)
    system.load_state_dict(checkpoint["state_dict"])

    system = system.cuda()
    system.eval()
    system.setup_testnvs()

    metric_dic = dict()
    keys = ['test_psnr_merged', 'test_ssim_merged', 'test_lpips_merged', 'test_iou_merged']
    for k in keys:
        metric_dic[k] = []

    with torch.no_grad():
        for i, batch in tqdm(enumerate(system.testnvs_dataloader())):
            log_dic, stack_image = system.testnvs_step(batch)
            print(i, log_dic)
            for k in keys:
                metric_dic[k] += [log_dic[k]]
            NUM, C, twoH, W = stack_image.shape
            H = int(twoH/2)
            stack_image = stack_image.permute([1,2,0,3]).reshape(C,twoH,-1)
            img = tensor_to_cv2(stack_image)
            img_path = os.path.join(render_path, 'preview', '%04d.jpg'%batch['frame_idx'])
            cv2.imwrite(img_path, img)

            img = tensor_to_cv2(stack_image[:,H:2*H,0:W])
            img_path = os.path.join(render_path, 'pred', '%04d.jpg'%batch['frame_idx'])
            cv2.imwrite(img_path, img)

        for k in keys:
            metric_dic[k] = sum(metric_dic[k]) / len(metric_dic[k])
        
        print(metric_dic)


if __name__ == "__main__":
    conf_cli = OmegaConf.from_cli()
    conf_dataset = OmegaConf.load(conf_cli.dataset_config)
    conf_default = OmegaConf.load("config/default_conf.yml")
    conf_merged = OmegaConf.merge(conf_default, conf_dataset, conf_cli)

    ckpt_path = conf_cli.ckpt_path
    print('ckpt path: ', ckpt_path)

    # read training config snapshot
    ckpt_conf_path = os.path.join(
        os.path.dirname(os.path.abspath(ckpt_path)),
        "run_config_snapshot.yaml",
    )
    conf_merged.ckpt_config_path = ckpt_conf_path
    conf_training = OmegaConf.create()
    conf_training.ckpt_config = OmegaConf.load(ckpt_conf_path)
    # # order: 1. merged; 2. training
    conf_merged = OmegaConf.merge(conf_training, conf_merged)
    conf_merged.ckpt_path = ckpt_path

    print("-" * 40)
    print(OmegaConf.to_yaml(conf_merged))
    print("-" * 40)

    main(config=conf_merged)

