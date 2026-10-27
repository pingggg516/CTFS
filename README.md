# CTFS

CTFS: Collaborative Teacher Framework for Forward-Looking Sonar Image Semantic Segmentation with Extremely Limited Labels.

[![arXiv](https://img.shields.io/badge/arXiv-2603.21071-b31b1b.svg)](https://arxiv.org/abs/2603.21071)

## Installation

```bash
cd CTFS
conda create -n CTFS python=3.10.4
conda activate CTFS
pip install -r requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

## Pretrained Backbone

Download the required DINOv2 pretrained backbone and place it under the `pretrained/` directory.

[dinov2_small](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) | [dinov2_base](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth) | [dinov2_large](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth)

```bash
├── pretrained
    ├── dinov2_small.pth
    ├── dinov2_base.pth
    └── dinov2_large.pth
```

### Usage Terms

The FSSG dataset is publicly available for **non-commercial use only**.<br>

If you use this dataset, please **cite our paper** and **acknowledge this repository as the dataset source** in any resulting publications or research outputs.

## Training

Run single-GPU training with:

```bash
# --labeled-id-path: Fill in the file path
# --unlabeled-id-path: Fill in the file path
python CTFS_train.py \
  --config configs/FSSG.yaml \
  --labeled-id-path /path/to/labeled.txt \
  --unlabeled-id-path /path/to/unlabeled.txt \
  --save-path ./exp/CTFS/<experiment_name>
```

## Notes

- Make sure `data_root` in `configs/FSSG.yaml` is a valid path.
- Make sure the pretrained backbone file exists in `pretrained/`.

## Acknowledgements

This project is based on [UniMatch-V2](https://github.com/LiheYoung/UniMatch-V2) and includes DINOv2-based backbone components.

We thank the authors of the following open-source projects:
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [UniMatch-V2](https://github.com/LiheYoung/UniMatch-V2)
- [UniMatch](https://github.com/LiheYoung/UniMatch)

## Citation

If you find this project useful, please consider citing:
```bash
@InProceedings{Guo_2026_CVPR,
    author    = {Guo, Ping and Li, Chengzhou and Meng, Guanchen and Jia, Qi and Liu, Jinyuan and Liu, Zhu and Liu, Yu and Luo, Zhongxuan and Fan, Xin},
    title     = {CTFS : Collaborative Teacher Framework for Forward-Looking Sonar Image Semantic Segmentation with Extremely Limited Labels},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
    month     = {June},
    year      = {2026},
    pages     = {1660-1669}
}
```

