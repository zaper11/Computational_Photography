import rawpy
import numpy as np
import matplotlib.pyplot as plt

from part1 import images, image_name, log
from part2 import load_linear
from part3_1 import PATTERN, make_masks
from part4 import apply_gains, gains_camera
from part5 import demosaic_bilinear, to_display, CROP
from part6 import get_cam2srgb, cam_to_srgb

out_dir = '/home/juneyoung-yoon/Desktop/Assignment1/results/1/part7'
LOG = f'{out_dir}/log.txt'

TARGETS = (0.15, 0.20, 0.25, 0.30)
TARGET = 0.20

def rgb2gray(rgb):
    # linear sRGB 기준 휘도 (Rec.709 가중치)
    return rgb @ np.array([0.2126, 0.7152, 0.0722])


def adjust_brightness(rgb, m):
    # I_bright = alpha * I_linear, alpha = m / mean(I_gray). 1 초과는 clip
    alpha = m / rgb2gray(rgb).mean()
    return np.clip(rgb * alpha, 0, 1), alpha


if __name__ == '__main__':
    open(LOG, 'w').close()
    for path in images:
        with rawpy.imread(path) as raw:
            raw_image, linear = load_linear(raw)
            wb = np.array(raw.camera_whitebalance, dtype=np.float64)
            color_desc = raw.color_desc
            M = get_cam2srgb(raw)

        masks = make_masks(raw_image.shape, PATTERN)
        name = image_name(path)

        # Part 4~6: 카메라 WB -> demosaic -> linear sRGB
        bayer = apply_gains(linear, gains_camera(wb, color_desc), masks)
        srgb = cam_to_srgb(demosaic_bilinear(bayer, PATTERN), M)

        H, W = linear.shape
        crop = (slice(H // 2 - CROP, H // 2 + CROP), slice(W // 2 - CROP, W // 2 + CROP))

        results = {'original': (srgb, 1.0)}
        for m in TARGETS:
            results[f'm={m:.2f}'] = adjust_brightness(srgb, m)

        log(f'== {name} ==  mean gray (before) = {rgb2gray(srgb).mean():.4f}', LOG)
        fig, ax = plt.subplots(2, len(results), figsize=(4 * len(results), 7))
        for k, (title, (img, alpha)) in enumerate(results.items()):
            clipped = (img >= 1).any(axis=-1).mean()
            log(f'  {title:10s} alpha={alpha:.3f}  mean gray={rgb2gray(img).mean():.4f}  '
                  f'clipped={clipped:.2%}', LOG)
            ax[0, k].imshow(to_display(img))
            ax[0, k].set_title(f'{title} (α={alpha:.2f})')
            ax[1, k].imshow(to_display(img[crop]))
            ax[0, k].axis('off')
            ax[1, k].axis('off')
        plt.savefig(f'{out_dir}/{name}_brightness.png', dpi=150, bbox_inches='tight')
        plt.close()
