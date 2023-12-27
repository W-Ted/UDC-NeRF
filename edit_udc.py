import os

os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa

from utils.util import get_timestamp, make_source_code_snapshot
import torch
from omegaconf import OmegaConf
from train_udc import UDCNeRFSystem
from tqdm import tqdm, trange
import numpy as np
import cv2
from torchvision import transforms

from scipy.spatial.transform import Rotation
import imageio


from PIL import Image
import torchvision.transforms as T


def move_camera_pose(pose, progress):
    # control the camera move (spiral pose)
    t = progress * np.pi * 4
    radii = 0.01
    center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5 * t)]) * radii
    pose[:3, 3] += pose[:3, :3] @ center
    return pose


def get_pure_rotation(progress_11: float, max_angle: float = 180):
    trans_pose = np.eye(4)
    trans_pose[:3, :3] = Rotation.from_euler(
        "z", progress_11 * max_angle, degrees=True
    ).as_matrix()
    return trans_pose


def get_pure_rotation_modi(progress_11: float, min_angle: float = -180, max_angle: float = 180):
    trans_pose = np.eye(4)
    trans_pose[:3, :3] = Rotation.from_euler(
        "z", min_angle + progress_11 * (max_angle-min_angle), degrees=True
    ).as_matrix()
    return trans_pose


def get_movement(progress: float, max_offset: float = 0.35, instance_id: int = 6):
    trans_pose = np.eye(4)
    if instance_id == 6:
        trans_pose[1, 3] += (1-np.sin(progress * np.pi)) * 0.35
    elif instance_id == 4:
        trans_pose[0, 3] += 0.05 
        trans_pose[1, 3] += (1-np.sin(progress * np.pi)) * 0.16 + 0.05 
    return trans_pose


def get_transformation_with_duplication_offset(progress, duplication_id: int):
    trans_pose = get_pure_rotation(np.sin(progress * np.pi * 2), max_angle=10)
    offset = 0.05
    if duplication_id > 0:
        trans_pose[0, 3] -= np.sin(progress * np.pi * 2) * offset
        trans_pose[1, 3] -= 0.2
    else:
        trans_pose[0, 3] += np.sin(progress * np.pi * 2) * offset
        trans_pose[1, 3] += 0.55
    return trans_pose


def visualize_depth(depth, cmap=cv2.COLORMAP_JET, vmin=None, vmax=None, scale=1.0, inverse=False):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x)  # change nan to 0
    mi = np.min(x) if vmin == None else vmin  # get minimum depth
    ma = np.max(x) if vmax == None else vmax
    x = np.clip(x, mi, ma)
    x = (x - mi) / max(ma - mi, 1e-8)  # normalize to 0~1
    x = x * scale  # normalize to 0~1
    if inverse == True:
        x = 1 - x
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_


def main(config):
    render_path = f"debug/rendered_view/render_{get_timestamp()}_{config.exp_name}/"
    os.makedirs(render_path, exist_ok=True)
    print('Write into ', render_path)
    
    system = UDCNeRFSystem(config)
    system.load_frame_meta()

    checkpoint = torch.load(config.ckpt_path)
    system.load_state_dict(checkpoint["state_dict"])

    system = system.cuda()
    system.eval()
    system.setup_edit()


    scene_name = config.train.scene_name
    if 'toy' in scene_name:
        vid = 'toy'
    elif 'scan' in scene_name:
        vid = 'scan'

    if vid == 'toy':   
        # for the video 
        W, H = 640, 480
        # W, H = 320, 240
        total_frames = 32
        pose_frame_idx = 124
        obj_id_list = [1,2,3,4,5]
        edit_type = {
            1: "pure_rotation",
            2: "pure_rotation",
            3: "pure_rotation",
            4: "original",
            5: "pure_rotation"
        }


    elif vid == 'scan':
        # for video
        # # move the desk: video
        W, H = 640, 480
        # W, H = 320, 240
        total_frames = 32
        pose_frame_idx = 355
        obj_id_list = [4,6]
        edit_type = {
            4: "movement",
            6: "movement"
        }

    for obj_id in obj_id_list:
        system.initialize_object_bbox(obj_id)


    with torch.no_grad():
        for idx in tqdm(range(total_frames)):
            processed_obj_id = []
            for obj_id in obj_id_list:
                obj_duplication_cnt = np.sum(np.array(processed_obj_id) == obj_id)
                progress = idx / total_frames # (0-31)/32, [0,1)

                # 1. get a trans_pose
                if edit_type[obj_id] == "duplication":
                    trans_pose = get_transformation_with_duplication_offset(
                        progress, obj_duplication_cnt
                    )
                elif edit_type[obj_id] == "pure_rotation":
                    trans_pose = get_pure_rotation_modi(progress_11=progress, min_angle=-56.25, max_angle=67.5)  #
                    # process: -1, 1  ->  rotation -180 - 180 degree around z 
                elif edit_type[obj_id] == "movement":
                    trans_pose = get_movement(progress=progress, instance_id=obj_id)

                elif edit_type[obj_id] == "original":
                    trans_pose = get_pure_rotation(progress_11=0)
                
                # 2. set the trans_pose
                system.set_object_pose_transform(obj_id, trans_pose, obj_duplication_cnt)
                processed_obj_id.append(obj_id)
            
            # print('object_pose_transform.key: ', system.object_pose_transform.keys()) 

            # render edited scene
            results = system.render_edit(
                h=H,
                w=W,
                camera_pose_Twc=move_camera_pose(
                    system.get_camera_pose_by_frame_idx(pose_frame_idx),
                    idx / total_frames,
                ),
                fovx_deg=getattr(system, "fov_x_deg_dataset", 60),
            )
            image_out_path = f"{render_path}/render_{idx:04d}.png"
            image_np = results["rgb_fine"].view(H, W, 3).detach().cpu().numpy()

            depth_np = results["depth_fine"].view(H, W).detach()
            if vid == 'scan':
                depth_vis = visualize_depth(depth_np, vmin=0, vmax=1.2, scale=0.8).permute([1,2,0]).numpy() # iccv
            elif vid == 'toy':
                depth_vis = visualize_depth(depth_np, vmin=0, vmax=0.7, scale=0.8).permute([1,2,0]).numpy() # toy using scale=0.8 # iccv
            vis = np.concatenate([image_np, depth_vis], 0)
            imageio.imwrite(image_out_path[:-4]+"_depth.png", (vis * 255).astype(np.uint8))

            system.reset_active_object_ids()
    
    # write frames into video
    imgs = []
    for idx in tqdm(range(total_frames)):
        img = cv2.imread(f"{render_path}/render_{idx:04d}_depth.png")
        imgs += [img[:,:,[2,1,0]]]
    if vid == 'toy':
        imgs = 5 * (imgs + imgs[::-1])
    elif vid == 'scan':
        imgs = 10 * imgs
    imageio.mimsave(f"{render_path}/{scene_name}.mp4", imgs, fps=15) # save mp4


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

