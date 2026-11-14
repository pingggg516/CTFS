
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

## FSSGv2 Dataset

The FSSGv2 dataset contains **3,761 forward-looking sonar images** with pixel-level semantic segmentation annotations, covering **10 target categories and 1 background class**.

### Note on FSSGv2

The dataset released at the link below has remained unchanged since its initial release. However, it differs from the version used in our paper: we corrected some inaccurate annotations before the release but forgot to mention this at that time. To make this clear, we now name the released dataset **FSSGv2**. If you have already downloaded it and evaluated other comparison methods on it, those results remain valid and can continue to be used.

To facilitate comparison, we retrained CTFS on FSSGv2 with 2%, 5% and 10% labeled data. The corresponding labeled and unlabeled splits, together with `val.txt` and `test.txt`, are provided in `splits/FSSG/`. The weights, training logs and test logs are available from [Baidu Netdisk](https://pan.baidu.com/s/1noKbhy4ZcnHFju1bn46h_A?pwd=0516) (extraction code: `0516`).

| Labeled ratio | Val mIoU | Test mIoU |
| :---: | :---: | :---: |
| 2% | 56.24 | 56.36 |
| 5% | 61.32 | 61.89 |
| 10% | 63.56 | 63.93 |

All runs use a batch size of 4, and the 2% setting uses a learning rate of `2.5e-6`. The other hyperparameters have not been tuned for FSSGv2, and the full configuration of each run is recorded in its training log, so further tuning may lead to better results. We sincerely apologize for any inconvenience, and please feel free to contact us if you have any questions.

### Download

Download the FSSGv2 dataset from [Baidu Netdisk](https://pan.baidu.com/s/12Awtkr8GCzO2tiNnHp08Yw?pwd=0516).

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

Each line of a split file contains one image–mask pair, separated by a single space, with both paths relative to `data_root`:

```text
Images/00003.png Masks/00003.png
Images/00005.png Masks/00005.png
```

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


### Usage Terms

The FSSGv2 dataset is publicly available for **non-commercial use only**.<br>

If you use this dataset, please **cite our paper** and **acknowledge this repository as the dataset source** in any resulting publications or research outputs.

## Training

Run single-GPU training with the following command. The validation split is read from `splits/FSSG/val.txt`.

```bash
python CTFS_train.py \
  --config configs/FSSG.yaml \
  --labeled-id-path splits/FSSG/labeled_10.txt \
  --unlabeled-id-path splits/FSSG/unlabeled_10.txt \
  --save-path ./exp/CTFS/<experiment_name>
```

## Inference

Use `inference.py` to run a trained CTFS model, either the released weights or a checkpoint from your own training (e.g. `best.pth`). To predict an image or a directory of images, run the command below; class-ID masks are saved to `outputs/mask/` and color visualizations to `outputs/color/`.

```bash
python inference.py --weights /path/to/weights.pth --input /path/to/images --output outputs
```

To evaluate a trained model on the test set:

```bash
python inference.py --weights /path/to/weights.pth --data-root /path/to/FSSGv2 --eval-split splits/FSSG/test.txt
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
