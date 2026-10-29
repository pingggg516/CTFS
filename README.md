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

## FSSG Dataset

The FSSG dataset contains **3,761 forward-looking sonar images** with pixel-level semantic segmentation annotations, covering **10 target categories and 1 background class**.

### Download

Download the FSSG dataset from [Baidu Netdisk](https://pan.baidu.com/s/12Awtkr8GCzO2tiNnHp08Yw?pwd=0516).

**Extraction code:** `0516`

### Directory Structure

Organize the sonar images and their corresponding segmentation masks as follows:

```text
FSSG/
├── Images/
│   ├── 00003.png
│   ├── 00005.png
│   └── ...
└── Masks/
    ├── 00003.png
    ├── 00005.png
    └── ...
```

- `Images/` contains the sonar images.
- `Masks/` contains the corresponding PNG segmentation masks.
- Each image and its mask should have the same filename and spatial dimensions. For example, `Images/00003.png` corresponds to `Masks/00003.png`.
- Preserve the original filenames to match the provided split files.

Set the dataset root in `configs/FSSG.yaml`:

```yaml
data_root: /path/to/FSSG
nclass: 11
```

### Split Files

Use the following format for the labeled, unlabeled and validation split files, with one image–mask pair per line:

```text
Images/00003.png Masks/00003.png
Images/00005.png Masks/00005.png
```

Separate the two paths with a single space. Both paths are relative to `data_root`.

Pass the training split files through `--labeled-id-path` and `--unlabeled-id-path`. The validation split is read from `splits/FSSG/val.txt`.

### Mask Format and Class Definitions

Each segmentation mask is a two-dimensional label map. The integer value at each pixel represents the semantic class of the corresponding image pixel.

| Pixel Value / Class ID | Category |
| :---: | --- |
| 0 | Background |
| 1 | Block |
| 2 | Circle Cage |
| 3 | Steel Frame |
| 4 | Concrete Column |
| 5 | Steel Plate |
| 6 | Pot |
| 7 | Square Cage |
| 8 | Tire |
| 9 | Underwater Robot |
| 10 | Diver |

For example, a mask pixel with value `8` belongs to **Tire**, while a pixel with value `10` belongs to **Diver**.

Read the original class IDs using Pillow and NumPy:

```python
from PIL import Image
import numpy as np

mask = np.array(Image.open("/path/to/FSSG/Masks/00003.png"))

print("Mask shape:", mask.shape)       # (H, W)
print("Class IDs:", np.unique(mask))   # Classes present in this mask

diver_region = (mask == 10)
```

Masks may appear nearly black when displayed as grayscale images because the class IDs range from `0` to `10`. A color palette can be applied for visualization.

Keep the original integer class IDs when loading masks for training. Do not convert the label maps to RGB or normalize their values to `[0, 1]`. If resizing is required, use nearest-neighbor interpolation.

The training configuration reserves `255` as the ignore value; it is not a semantic class.

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

