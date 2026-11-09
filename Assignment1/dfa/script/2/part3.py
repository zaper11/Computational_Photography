import numpy as np
import matplotlib.pyplot as plt

from part1 import conv2d, load_gray, crop_slice
from part2 import make_grid

IMGS = [
    '/home/juneyoung-yoon/Desktop/Assignment1/images/example.jpg',
    '/home/juneyoung-yoon/Desktop/Assignment1/images/example2.jpg',
    '/home/juneyoung-yoon/Desktop/Assignment1/images/example3.jpg']
OUT_DIR = '/home/juneyoung-yoon/Desktop/Assignment1/results/2/part3'

# Sobel
def sobel_kernels():
    Sx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)
    
    Sy = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float64)
    
    return Sx, Sy


def sobel_edges(image):
    Sx, Sy = sobel_kernels()
    
    # SX, SY -> Ix, Iy
    Ix = conv2d(image, Sx)
    Iy = conv2d(image, Sy)
    
    # M = sqrt(Ix^2 + Iy^2)
    M = np.sqrt(Ix**2 + Iy**2)
    
    return Ix, Iy, M


# LoG
def log_kernel(sigma):
    r = int(np.ceil(3 * sigma))
    x, y = make_grid(r)
    r_sq = x**2 + y**2
    
    # 편미분 결과
    kernel = ((r_sq - 2 * sigma**2) / (sigma**4)) * np.exp(-r_sq / (2 * sigma**2))
    
    # zero mean 보정
    kernel = kernel - np.mean(kernel)
    
    return kernel


def zero_crossing(response, thresh=0.3):
    H, W = response.shape
    zc = np.zeros((H, W), dtype=bool)
    
    # 수평 방향 부호 변화 확인
    diff_h = np.abs(response[:, :-1] - response[:, 1:])
    sign_h = (response[:, :-1] * response[:, 1:]) < 0
    cross_h = sign_h & (diff_h > thresh)
    zc[:, :-1] |= cross_h
    zc[:, 1:] |= cross_h

    # 수직 방향 부호 변화 확인
    diff_v = np.abs(response[:-1, :] - response[1:, :])
    sign_v = (response[:-1, :] * response[1:, :]) < 0
    cross_v = sign_v & (diff_v > thresh)
    zc[:-1, :] |= cross_v
    zc[1:, :] |= cross_v

    return zc


if __name__ == '__main__':
    for path in IMGS:
        name = path.split('/')[-1].replace('.jpg', '')
        gray = load_gray(path)

        # Sobel
        Ix, Iy, M = sobel_edges(gray)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, (img, title) in zip(axes, [(gray, 'original'), (Ix, 'Ix'), (Iy, 'Iy'), (M, 'magnitude')]):
            ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/{name}_sobel.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # LoG: σ별
        sigmas = [1, 2, 4]
        fig, axes = plt.subplots(2, len(sigmas), figsize=(5 * len(sigmas), 10))
        for col, s in enumerate(sigmas):
            resp = conv2d(gray, log_kernel(s))
            edges = zero_crossing(resp)
            axes[0, col].imshow(resp, cmap='gray')
            axes[0, col].set_title(f'LoG response (sigma={s})')
            axes[1, col].imshow(edges, cmap='gray')
            axes[1, col].set_title(f'zero-crossings (sigma={s})')
        for ax in axes.ravel():
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/{name}_log.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # ---- Sobel vs LoG 비교 ----
        s_cmp = 1                               # 비교에 쓸 LoG의 sigma
        resp = conv2d(gray, log_kernel(s_cmp))
        log_edges = zero_crossing(resp)              # 위에서 쓴 thresh와 동일하게 맞추세요

        # Sobel은 M에 threshold를 적용해 이진 엣지 맵으로 만든다
        percentiles = [90]
        sobel_maps = [(M > np.percentile(M, p), f'Sobel (top {100 - p}%)') for p in percentiles]

        panels = [(gray, 'original', 'gray')] \
               + [(m, t, 'gray') for m, t in sobel_maps] \
               + [(log_edges, f'LoG zero-crossing (sigma={s_cmp})', 'gray')]

        fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.5))
        for ax, (img, title, cmap) in zip(axes, panels):
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1 if img.dtype == bool else None)
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/{name}_comparison.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # ---- 중앙 크롭 비교 (전체 이미지에서 계산한 결과를 잘라 표시) ----
        cs = crop_slice(gray.shape)

        # Sobel
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, (img, title) in zip(axes, [(gray, 'original'), (Ix, 'Ix'), (Iy, 'Iy'), (M, 'magnitude')]):
            ax.imshow(img[cs], cmap='gray', interpolation='nearest')
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/{name}_sobel_crop.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # LoG: σ별
        fig, axes = plt.subplots(2, len(sigmas), figsize=(5 * len(sigmas), 10))
        for col, s in enumerate(sigmas):
            resp_s = conv2d(gray, log_kernel(s))
            edges_s = zero_crossing(resp_s)
            axes[0, col].imshow(resp_s[cs], cmap='gray', interpolation='nearest')
            axes[0, col].set_title(f'LoG response (sigma={s})')
            axes[1, col].imshow(edges_s[cs], cmap='gray', interpolation='nearest')
            axes[1, col].set_title(f'zero-crossings (sigma={s})')
        for ax in axes.ravel():
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/{name}_log_crop.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # Sobel vs LoG
        fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.5))
        for ax, (img, title, cmap) in zip(axes, panels):
            ax.imshow(img[cs], cmap=cmap, vmin=0, vmax=1 if img.dtype == bool else None, interpolation='nearest')
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/{name}_comparison_crop.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()
