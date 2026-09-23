"""Paper Eq. 6-11 perturbations in unnormalized [0, 1] intensity space."""
import random
import numpy as np
import torch
from PIL import Image

def attenuation(x, gamma=0.25):
    h = x.shape[-2]
    scale = 1 - gamma * torch.arange(h, device=x.device, dtype=x.dtype) / h
    return x * scale.view(1, 1, h, 1)

def shadow(x, alpha=0.2, radius_ratio=0.2):
    out = x.clone()
    b, c, h, w = x.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=x.device, dtype=x.dtype),
                           torch.arange(w, device=x.device, dtype=x.dtype), indexing='ij')
    radius = radius_ratio * min(h, w)
    for idx in range(b):
        for _ in range(random.randint(1, 3)):
            x0, y0 = random.randint(0, w // 2), random.randint(0, h)
            angle, span = random.randint(-45, 45), random.randint(15, 60)
            distances = ((xx - x0)**2 + (yy - y0)**2).sqrt()
            angles = torch.atan2(yy - y0, xx - x0) * (180 / torch.pi)
            region = (angles >= angle) & (angles <= angle + span) & (distances <= radius)
            factor = torch.where(region, 1 - alpha * (1 - distances / radius), torch.ones_like(distances))
            out[idx] = out[idx] * factor
    return out

def augment_intensity(x, mode):
    if mode == 'sonar_a':
        return shadow(x)
    if mode == 'sonar_b':
        return attenuation(x)
    raise ValueError(mode)

def augment_pil(img, mode):
    # Probabilities and alpha/gamma inherited: the paper does not specify them.
    probability = 0.6 if mode == 'sonar_a' else 0.5
    if random.random() >= probability:
        return img
    x = torch.from_numpy(np.array(img, copy=True)).permute(2, 0, 1).float().unsqueeze(0) / 255
    out = augment_intensity(x, mode).squeeze(0).permute(1, 2, 0)
    return Image.fromarray((out.clamp(0, 1).numpy() * 255).astype(np.uint8))

def augment_normalized(x, mode):
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    intensity = (x * std + mean).clamp(0, 1)
    return (augment_intensity(intensity, mode) - mean) / std
