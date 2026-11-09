import rawpy
import numpy as np
import matplotlib.pyplot as plt

from part1 import images, image_name
from part2 import load_linear
from part3_1 import PATTERN, make_masks
from part4 import apply_gains, gains_camera

out_dir = '/home/juneyoung-yoon/Desktop/Assignment1/results/1/part5'

CROP = 100 # 확대 crop 반폭 (중심 기준 2*CROP x 2*CROP)


def box3(x):
    # 3x3 이웃 합. 바깥은 0으로 채움
    H, W = x.shape
    p = np.pad(x, 1)
    return sum(p[i:i + H, j:j + W] for i in range(3) for j in range(3))


def demosaic_bilinear(bayer, pattern):
    # (H, W) Bayer -> (H, W, 3) RGB, 채널 순서 [R, G, B]
    chans = []
    for M in make_masks(bayer.shape, pattern):
        M = M.astype(np.float32)
        est = box3(bayer * M) / box3(M)
        chans.append(np.where(M > 0, bayer, est))
    return np.stack(chans, axis=-1)


def to_display(rgb):
    # 표시용 감마
    return np.clip(rgb, 0, 1) ** (1 / 2.2)


if __name__ == '__main__':
    for path in images:
        with rawpy.imread(path) as raw:
            raw_image, linear = load_linear(raw)
            wb = np.array(raw.camera_whitebalance, dtype=np.float64)
            color_desc = raw.color_desc

        masks = make_masks(raw_image.shape, PATTERN)
        name = image_name(path)

        results = {
            'none': (1.0, 1.0, 1.0),
            'camera': gains_camera(wb, color_desc),
        }

        H, W = linear.shape
        crop = (slice(H // 2 - CROP, H // 2 + CROP), slice(W // 2 - CROP, W // 2 + CROP))

        fig, ax = plt.subplots(2, len(results), figsize=(5 * len(results), 8))
        for k, (title, gains) in enumerate(results.items()):
            bayer = apply_gains(linear, gains, masks)
            rgb = demosaic_bilinear(bayer, PATTERN)
            assert rgb.shape == (H, W, 3)
            assert np.array_equal(rgb[masks[1]][:, 1], bayer[masks[1]])   # 측정값 보존 확인(G)
            ax[0, k].imshow(to_display(rgb))
            ax[0, k].set_title(title)
            ax[1, k].imshow(to_display(rgb[crop]))
            ax[0, k].axis('off')
            ax[1, k].axis('off')
        plt.savefig(f'{out_dir}/{name}_demosaic.png', dpi=150, bbox_inches='tight')
        plt.close()
