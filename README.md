<p align="center">
<h1 align="center"><strong>UDC-NeRF: Learning <font color="#5364cc">U</font>nified <font color="#5364cc">D</font>ecompositional and <font color="#5364cc">C</font>ompositional <font color="#5364cc">NeRF</font> for Editable Novel View Synthesis</strong></h1>
<h3 align="center">ICCV 2023</h3>

<p align="center">
    <a href="https://w-ted.github.io/">Yuxin Wang</a><sup>1</sup>,</span>
    <a href="https://wywu.github.io/">Wayne Wu</a><sup>2</sup>,
    <a href="https://www.danxurgb.net/">Dan Xu</a><sup>1</sup>
    <br>
        <sup>1</sup>HKUST,
        <sup>2</sup>Shanghai AI Lab
</p>

<div align="center">
    <a href=https://arxiv.org/abs/2308.02840><img src='https://img.shields.io/badge/arXiv-2308.02840-b31b1b.svg'></a>  
    <a href='https://w-ted.github.io/publications/udc-nerf/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  
</div>
</p>

## Demo Videos 
<details open>
  <summary>Object Manipulation compared with Object-NeRF. </summary>

https://github.com/W-Ted/UDC-NeRF/assets/31502887/156b08bb-a70d-47e9-9ab0-fcdc983ff5e7
</details>


<!-- <img width="48%" src="./assets/toy.gif" autoplay loop muted controls title="Scene Toydesk-02"></img>
<img width="48%" src="./assets/scan.gif" autoplay loop muted controls title="Scene ScanNet-0113"></img> -->

## Installation

```
git clone --recursive https://github.com/W-Ted/UDC-NeRF.git

conda create -n udcnerf python=3.6.13
conda activate udcnerf
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
pip install catalyst
```

## Dataset

Please refer to [Object-NeRF&#39;s data preparation guidance](https://github.com/zju3dv/object_nerf/tree/main/data_preparation) to prepare the dataset.

Please follow the [LaMa&#39;s guidance](https://github.com/advimman/lama/tree/main) to config the environment and download the pre-trained checkpoint. Then please refer to the following scripts to in-paint the background.

```
# step 1: prepare the images and corresponding masks. 
python preprocess/scripts/prepare_lamain_xxx.py  # Please modify the data path first.
# step 2: run LaMa to in-paint the background. 
bash preprocess/scripts/run_lama.sh
# step 3: 
python preprocess/scripts/rename_lamaout.py
```

## Editable Novel View Synthesis

Please download our [pre-trained checkpoints](https://hkustconnect-my.sharepoint.com/:u:/g/personal/ywangom_connect_ust_hk/EaoQh_pp0yBKiKq6-1m18MMBrFsqKjOuIujtVrhGQ-S56A?e=jgx0ho), and put the folder in `udc-nerf/pretrained_ckpts`. The following scripts can be used to generate demo videos in `debug/rendered_view/render_xxx_edit/`.

```
python scripts/edit_toydesk2.sh # for toydesk2, takes more than 1.5h. 
python scripts/edit_scannet0113_multi.sh # for scannet0113_multi, takes about 30min. 
```

## Training

In our experiments, we used two scenes in ToyDesk Dataset and four scenes in ScanNet Dataset, i.e, 0024, 0038, 0113, 0192.  The following scripts are two examples, and please refer to the training scripts in `scripts/` for more details.

```
python scripts/train_toydesk2.sh # for toydesk2
python scripts/train_scannet0113_multi.sh # for scannet0113_multi
```

## Evaluation

The following scripts are two examples, and please refer to the evaluation scripts in `scripts/` for more details.

```
python scripts/test_toydesk2.sh # for toydesk2
python scripts/test_scannet0113_multi.sh # for scannet0113_multi
```

## Acknowledgements

This project is built upon [Object-NeRF](https://github.com/zju3dv/object_nerf). The in-painted images are obtained by [LaMa](https://github.com/advimman/lama). Kudos to these researchers.

## Citation

```BibTeX
@inproceedings{wang2023udcnerf,
     title={Learning Unified Decompositional and Compositional NeRF for Editable Novel View Synthesis},
     author={Wang, Yuxin and Wu, Wayne and Xu, Dan},
     booktitle={ICCV},
     year={2023}
     }
```
