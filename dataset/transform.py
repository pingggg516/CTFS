import random

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms


def crop(img, mask, size, ignore_value=255):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask


def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def normalize(img, mask=None):
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img


def resize(img, mask, ratio_range):
    w, h = img.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask


def add_speckle_noise(img, intensity=0.08, p=0.7):
    if random.random() > p:
        return img

    img_array = np.array(img).astype(np.float32)

    intensity = float(intensity)
    intensity = max(intensity, 1e-6)
    shape = 1.0 / (intensity ** 2)
    scale = intensity ** 2
    noise = np.random.gamma(shape, scale, img_array.shape).astype(np.float32)

    noisy_img = img_array * noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    return Image.fromarray(noisy_img, mode=img.mode)


def apply_distance_attenuation(img, attenuation_factor=0.25, p=0.5):
    if random.random() > p:
        return img

    img_array = np.array(img)
    height, width = img_array.shape[:2]


    distance_mask = np.linspace(1.0, 1.0 - attenuation_factor, height)
    distance_mask = np.tile(distance_mask.reshape(-1, 1), (1, width))

    if len(img_array.shape) == 3:
        distance_mask = np.stack([distance_mask] * img_array.shape[2], axis=2)


    attenuated_img = img_array * distance_mask
    attenuated_img = np.clip(attenuated_img, 0, 255).astype(np.uint8)

    return Image.fromarray(attenuated_img)


def sonar_weak_augmentation(img, speckle_intensity=0.08, attenuation_factor=0.25):
    img = add_speckle_noise(img, intensity=speckle_intensity, p=0.7)


    img = apply_distance_attenuation(img, attenuation_factor=attenuation_factor, p=0.5)

    return img


def add_radial_noise(img, intensity=0.2, num_rays=8, p=0.5):
    if random.random() > p:
        return img

    img_array = np.array(img)
    if len(img_array.shape) == 3:
        h, w, c = img_array.shape
    else:
        h, w = img_array.shape
        c = 1


    center_x = random.randint(w//4, 3*w//4)
    center_y = random.randint(h//4, 3*h//4)


    y, x = np.ogrid[:h, :w]


    angles = np.arctan2(y - center_y, x - center_x)


    radial_pattern = np.sin(angles * num_rays) * intensity * 255


    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(h**2 + w**2)
    distance_weight = 1 - (distances / max_dist)

    radial_pattern *= distance_weight

    if c == 3:
        radial_pattern = np.stack([radial_pattern] * 3, axis=2)

    noisy_img = img_array + radial_pattern
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    return Image.fromarray(noisy_img)


def add_irregular_occlusion(img, num_patches=2, min_size=15, max_size=40, p=0.6):
    if random.random() > p:
        return img

    img_array = np.array(img)
    if len(img_array.shape) == 3:
        h, w, c = img_array.shape
    else:
        h, w = img_array.shape
        c = 1

    result = img_array.copy()

    for _ in range(num_patches):

        safe_max = min(max_size, (w - 1) // 2, (h - 1) // 2)
        if safe_max < min_size:
            return img

        center_x = random.randint(safe_max, w - 1 - safe_max)
        center_y = random.randint(safe_max, h - 1 - safe_max)
        size = random.randint(min_size, safe_max)


        a = random.randint(size//2, size)
        b = random.randint(size//3, size//2)
        angle = random.randint(0, 180)


        y, x = np.ogrid[:h, :w]


        cos_angle = np.cos(np.radians(angle))
        sin_angle = np.sin(np.radians(angle))

        x_rot = (x - center_x) * cos_angle + (y - center_y) * sin_angle
        y_rot = -(x - center_x) * sin_angle + (y - center_y) * cos_angle


        ellipse_mask = (x_rot**2 / a**2 + y_rot**2 / b**2) <= 1


        occlusion_type = random.choice(['dark', 'bright', 'noise'])

        if occlusion_type == 'dark':

            result[ellipse_mask] = result[ellipse_mask] * 0.1
        elif occlusion_type == 'bright':

            result[ellipse_mask] = np.minimum(result[ellipse_mask] + 100, 255)
        else:

            noise_patch = np.random.randint(0, 255, result[ellipse_mask].shape)
            result[ellipse_mask] = noise_patch

    return Image.fromarray(result.astype(np.uint8))


def add_acoustic_shadow(img, intensity=0.4, p=0.3):
    if random.random() > p:
        return img

    img_array = np.array(img)
    if len(img_array.shape) == 3:
        h, w, c = img_array.shape
    else:
        h, w = img_array.shape
        c = 1


    num_shadows = random.randint(1, 3)
    result = img_array.copy()

    for _ in range(num_shadows):

        start_x = random.randint(0, w//2)
        start_y = random.randint(0, h - 1)


        angle_start = random.randint(-45, 45)
        angle_span = random.randint(15, 60)

        y, x = np.ogrid[:h, :w]


        angles = np.degrees(np.arctan2(y - start_y, x - start_x))
        distances = np.sqrt((x - start_x)**2 + (y - start_y)**2)


        angle_mask = (angles >= angle_start) & (angles <= angle_start + angle_span)
        distance_mask = distances <= min(h, w) * 0.6
        shadow_mask = angle_mask & distance_mask


        shadow_intensity = intensity * (1 - distances / (min(h, w) * 0.6))
        shadow_intensity = np.clip(shadow_intensity, 0, intensity)

        if c == 3:
            for ch in range(3):
                result[:, :, ch][shadow_mask] = result[:, :, ch][shadow_mask] * (1 - shadow_intensity[shadow_mask])
        else:
            result[shadow_mask] = result[shadow_mask] * (1 - shadow_intensity[shadow_mask])

    return Image.fromarray(result.astype(np.uint8))


def sonar_strong_augmentation(img):

    img = add_speckle_noise(img, intensity=random.uniform(0.12, 0.15), p=1.0)


    img = add_radial_noise(img,
                          intensity=random.uniform(0.1, 0.3),
                          num_rays=random.randint(6, 16),
                          p=0.5)


    img = add_acoustic_shadow(img,
                             intensity=random.uniform(0.2, 0.5),
                             p=0.3)


    img = add_irregular_occlusion(img,
                                 num_patches=random.randint(1, 3),
                                 min_size=10,
                                 max_size=35,
                                 p=0.6)


    img = apply_distance_attenuation(img,
                                   attenuation_factor=random.uniform(0.3, 0.6),
                                   p=0.4)

    return img
