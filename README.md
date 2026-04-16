# CTFS
CTFS : Collaborative Teacher Framework for Forward-Looking Sonar Image Semantic Segmentation with Extremely Limited Labels

## Getting Started

### Installation
```bash
cd CTFS
conda create -n CTFS python=3.10.4
conda activate CTFS
pip install -r requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

### Pretrained Backbone
[dinov2_small](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) | [dinov2_base](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth) | [dinov2_large](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth)
```bash
├── ./pretrained
    ├── dinov2_small.pth
    ├── dinov2_base.pth
    └── dinov2_large.pth
```
### Dataset
* The FSSG dataset contains 3,761 forward-looking sonar images consisting of 11 targets(including background).Please download FSSG dataset here firstly.
* Class names and the corresponding pixel values in the labels are as follows-:
```bash
 Pixel Value: Class Name
>>        0:  Background         
>>        1:  Block        
>>        2:  Circle Cage            
>>        3:  Steel Frame          
>>        4:  Concrete Column   
>>        5:  Steel Plate          
>>        6:  Pot      
>>        7:  SquareCage
>>        8:  Tire
>>        9:  Underwater Robot           
>>        10: Diver
```
* Make ensure the dataset structure is as follows:
```bash
├──$HOME/datasets/
    ├── Images
        ├── 0_frame_341.png
        ├── 0_frame_350.png
        ...
    └── Masks
        ├── 0_frame_341.png
        ├── 0_frame_350.png
        ...
```
Please modify your dataset path in configuration files.
## Usage



