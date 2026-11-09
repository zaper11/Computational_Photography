import os
import numpy as np
import matplotlib.pyplot as plt

from Image_Blending.part1_1 import load_image, log

IMGS = [
    '/home/juneyoung-yoon/Desktop/Computational_Photography/Assignment2/Image_Enhancement/palace.jpeg'
]
OUT_DIR = '/home/juneyoung-yoon/Desktop/Computational_Photography/Assignment2/results/Image_Enhancement/part2_2'
LOG = f'{OUT_DIR}/log.txt'

L = 256  # intensity level 수 (8-bit)


def equalize(gray):
    # gray: uint8 (M x N). h(r_k) -> c(r_k) -> s_k = c(r_k) / MN -> (L-1) * s_k 로 스케일
    h = np.bincount(gray.ravel(), minlength=L)
    c = np.cumsum(h)
    s = c / gray.size
    lut = np.round((L - 1) * s).astype(np.uint8)
    return lut[gray], lut


def rgb2ycbcr(rgb):
    # BT.601 (full range), rgb: [0, 1]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return np.stack([y, cb, cr], axis=-1)


def ycbcr2rgb(ycc):
    y, cb, cr = ycc[..., 0], ycc[..., 1] - 0.5, ycc[..., 2] - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return np.stack([r, g, b], axis=-1)


def to_u8(x):
    return np.round(np.clip(x, 0, 1) * (L - 1)).astype(np.uint8)


def equalize_luminance(rgb):
    # Y 채널만 equalization, Cb / Cr은 그대로
    ycc = rgb2ycbcr(rgb)
    y_orig = to_u8(ycc[..., 0])
    y_eq, lut = equalize(y_orig)
    ycc[..., 0] = y_eq / (L - 1)
    return np.clip(ycbcr2rgb(ycc), 0, 1), y_orig, y_eq, lut


def plot_hist(ax, gray, title):
    h = np.bincount(gray.ravel(), minlength=L)
    ax.bar(np.arange(L), h / gray.size, width=1.0, color='gray')
    ax.set_title(title)
    ax.set_xlabel('intensity r_k')
    ax.set_ylabel('h(r_k) / MN')
    ax.set_xlim(0, L - 1)
    ax2 = ax.twinx()
    ax2.plot(np.arange(L), np.cumsum(h) / gray.size, 'r-')
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel('CDF', color='r')


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    open(LOG, 'w').close()

    for path in IMGS:
        name = os.path.basename(path).rsplit('.', 1)[0]
        rgb = load_image(path)
        log(f'== {name} ==  shape={rgb.shape}', LOG)

        out, y_orig, y_eq, lut = equalize_luminance(rgb)
        log(f'  mapping range: {lut.min()} ~ {lut.max()}, monotonic={bool(np.all(np.diff(lut.astype(int)) >= 0))}', LOG)

        # 1) 결과 비교: 원본 / equalized (Y of YCbCr)
        fig, ax = plt.subplots(1, 2, figsize=(10, 8))
        for a, im, t in zip(ax, [rgb, out], ['original', 'equalized (Y of YCbCr)']):
            a.imshow(np.clip(im, 0, 1)); a.set_title(t); a.axis('off')
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/{name}_color.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # 2) Y 히스토그램 + CDF, 매핑 함수
        fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
        plot_hist(ax[0], y_orig, 'Y histogram (original)')
        plot_hist(ax[1], y_eq, 'Y histogram (equalized)')
        ax[2].plot(np.arange(L), lut, 'b-', label='s_k')
        ax[2].plot([0, L - 1], [0, L - 1], 'k--', alpha=0.4, label='identity')
        ax[2].set_title('mapping function (r_k -> s_k)')
        ax[2].set_xlabel('input r_k'); ax[2].set_ylabel('output')
        ax[2].legend(); ax[2].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/{name}_hist.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()
