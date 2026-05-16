# CTFS

CTFS: Collaborative Teacher Framework for Forward-Looking Sonar Image Semantic Segmentation with Extremely Limited Labels.


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

## Dataset
The FSSG dataset contains 3,761 forward-looking sonar images consisting of 11 targets(including background).Please download FSSG dataset [Here](https://pan.baidu.com/s/1Oe9j98_BJnrSwnDljd2vww?pwd=uhcb) firstly.

The dataset root should contain `Images/` and `Masks/` subdirectories.

```bash
├── /path/to/dataset_root/
    ├── Images
    │   ├── 0_frame_341.png
    │   ├── 0_frame_350.png
    │   └── ...
    └── Masks
        ├── 0_frame_341.png
        ├── 0_frame_350.png
        └── ...
```

Set the dataset root in `configs/CTFS.yaml`:

```yaml
data_root: /path/to/dataset_root
```

All split list files in this project should use the same line format:

```text
Images/xxx.png Masks/xxx.png
```

The paths in each line should be relative to `data_root`.


### Class Definition
* Class names and the corresponding pixel values in the labels are as follows:
```text
0: Background
1: Block
2: Circle Cage
3: Steel Frame
4: Concrete Column
5: Steel Plate
6: Pot
7: SquareCage
8: Tire
9: Underwater Robot
10: Diver
```

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

- Make sure `data_root` in `configs/CTFS.yaml` is a valid path.
- Make sure the pretrained backbone file exists in `pretrained/`.

## Acknowledgements

This project is based on [UniMatch-V2](https://github.com/LiheYoung/UniMatch-V2) and includes DINOv2-based backbone components.

We thank the authors of the following open-source projects:
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [UniMatch-V2](https://github.com/LiheYoung/UniMatch-V2)
- [UniMatch](https://github.com/LiheYoung/UniMatch)


