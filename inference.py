import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model.semseg.dpt import DPT
from util.classes import CLASSES
from util.utils import intersectionAndUnion


MODEL_CONFIGS = {
    'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

PALETTE = np.array([
    [0, 0, 0],
    [230, 25, 75],
    [60, 180, 75],
    [255, 225, 25],
    [0, 130, 200],
    [245, 130, 48],
    [145, 30, 180],
    [70, 240, 240],
    [240, 50, 230],
    [210, 245, 60],
    [250, 190, 212],
], dtype=np.uint8)

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

TO_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def parse_args():
    parser = argparse.ArgumentParser(description='CTFS inference on forward-looking sonar images')
    parser.add_argument('--weights', type=str, required=True, help='Checkpoint or state_dict file (.pth)')
    parser.add_argument('--input', type=str, help='An image file or a directory of images')
    parser.add_argument('--output', type=str, default='outputs', help='Directory for predicted masks')
    parser.add_argument('--backbone', type=str, default='small', choices=list(MODEL_CONFIGS))
    parser.add_argument('--nclass', type=int, default=11)
    parser.add_argument('--key', type=str, default='model',
                        help='Key of the weights inside a full training checkpoint (model = student)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no-color', action='store_true', help='Do not save color visualizations')
    parser.add_argument('--data-root', type=str, help='FSSG root, required with --eval-split')
    parser.add_argument('--eval-split', type=str,
                        help='Split file with "image mask" pairs, e.g. splits/FSSG/val.txt; reports mIoU')
    return parser.parse_args()


def load_model(args):
    model = DPT(**{**MODEL_CONFIGS[args.backbone], 'nclass': args.nclass})
    checkpoint = torch.load(args.weights, map_location='cpu')
    if isinstance(checkpoint, dict) and args.key in checkpoint and isinstance(checkpoint[args.key], dict):
        state_dict = checkpoint[args.key]
    else:
        state_dict = checkpoint
    state_dict = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model.to(args.device).eval()


@torch.no_grad()
def predict(model, image, device):
    img = TO_TENSOR(image).unsqueeze(0).to(device)
    ori_h, ori_w = img.shape[-2:]
    new_h = int(ori_h / 14 + 0.5) * 14
    new_w = int(ori_w / 14 + 0.5) * 14
    img = F.interpolate(img, (new_h, new_w), mode='bilinear', align_corners=True)
    pred = model(img)
    pred = F.interpolate(pred, (ori_h, ori_w), mode='bilinear', align_corners=True)
    return pred.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def list_images(path):
    if os.path.isfile(path):
        return [path]
    return sorted(os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(IMAGE_EXTS))


def run_inference(model, args):
    os.makedirs(os.path.join(args.output, 'mask'), exist_ok=True)
    if not args.no_color:
        os.makedirs(os.path.join(args.output, 'color'), exist_ok=True)
    paths = list_images(args.input)
    for i, path in enumerate(paths, 1):
        pred = predict(model, Image.open(path).convert('RGB'), args.device)
        name = os.path.splitext(os.path.basename(path))[0] + '.png'
        Image.fromarray(pred).save(os.path.join(args.output, 'mask', name))
        if not args.no_color:
            Image.fromarray(PALETTE[pred]).save(os.path.join(args.output, 'color', name))
        print(f'[{i}/{len(paths)}] {path}')
    print(f'Class-ID masks saved to {os.path.join(args.output, "mask")}')


def run_evaluation(model, args):
    with open(args.eval_split) as f:
        pairs = [line.split() for line in f.read().splitlines() if line.strip()]
    inter = np.zeros(args.nclass)
    union = np.zeros(args.nclass)
    for i, (img_rel, mask_rel) in enumerate(pairs, 1):
        image = Image.open(os.path.join(args.data_root, img_rel)).convert('RGB')
        mask = np.array(Image.open(os.path.join(args.data_root, mask_rel)))
        pred = predict(model, image, args.device)
        a, b, _ = intersectionAndUnion(pred, mask, args.nclass, 255)
        inter += a
        union += b
        if i % 50 == 0 or i == len(pairs):
            print(f'[{i}/{len(pairs)}] evaluated')
    iou = inter / (union + 1e-10) * 100.0
    names = CLASSES['FSSG'] if args.nclass == len(CLASSES['FSSG']) else [str(c) for c in range(args.nclass)]
    for c, name in enumerate(names):
        print(f'Class [{c} {name}] IoU: {iou[c]:.2f}')
    print(f'mIoU: {iou.mean():.2f}')


def main():
    args = parse_args()
    if not args.input and not args.eval_split:
        raise SystemExit('Provide --input for inference and/or --eval-split with --data-root for evaluation.')
    if args.eval_split and not args.data_root:
        raise SystemExit('--eval-split requires --data-root.')
    model = load_model(args)
    if args.input:
        run_inference(model, args)
    if args.eval_split:
        run_evaluation(model, args)


if __name__ == '__main__':
    main()
