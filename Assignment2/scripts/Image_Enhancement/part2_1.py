import os
import numpy as np
import matplotlib.pyplot as plt

from Image_Blending.part1_1 import get_gaussian_kernel, conv2d, load_image, log


IMGS = [
    '/home/juneyoung-yoon/Desktop/Computational_Photography/Assignment2/Image_Enhancement/palace.jpeg'
]
OUT_DIR = '/home/juneyoung-yoon/Desktop/Computational_Photography/Assignment2/results/Image_Enhancement/part2_1'
LOG = f'{OUT_DIR}/log.txt'

ALPHAS = [1.0, 2.0, 3.0, 4.0]
SIGMA = 1.4                 # alpha 비교에 쓰는 기본 sigma
SIGMAS = [0.5, 1.0, 2.0, 3.0]  # sigma 비교 (alpha 고정)
ALPHA_FIXED = 1.5
CROP = 128
CROP_DY = -250  # 크롭 위치를 중앙에서 위(-)/아래(+)로 이동 (px)


def kernel_size(sigma):
    # 가우시안이 거의 다 들어오도록 반경 3 sigma, 홀수 크기
    return 2 * int(np.ceil(3 * sigma)) + 1


def unsharp(img, alpha, sigma):
    # I_out = I_in + alpha * (I_in - G_sigma * I_in)
    G = get_gaussian_kernel(kernel_size(sigma), sigma)
    return img + alpha * (img - conv2d(img, G))


def unsharp_single_kernel(img, alpha, sigma):
    # 단일 커널: K = (1 + alpha) * delta - alpha * G  ->  I_out = K * I_in
    size = kernel_size(sigma)
    G = get_gaussian_kernel(size, sigma)
    delta = np.zeros((size, size)); delta[size // 2, size // 2] = 1.0
    K = (1 + alpha) * delta - alpha * G
    return conv2d(img, K)


def crop_slice(shape, size=CROP):
    h, w = shape[:2]
    y0, x0 = (h - size) // 2 + CROP_DY, (w - size) // 2
    y0 = int(np.clip(y0, 0, h - size))
    return slice(y0, y0 + size), slice(x0, x0 + size)


def show_grid(panels, path, crop=None, cols=5):
    rows = int(np.ceil(len(panels) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for a in ax.ravel():
        a.axis('off')
    for a, (im, title) in zip(ax.ravel(), panels):
        im = np.clip(im, 0, 1)
        a.imshow(im[crop] if crop else im, interpolation='nearest')
        a.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    open(LOG, 'w').close()

    for path in IMGS:
        name = os.path.basename(path).rsplit('.', 1)[0]
        img = load_image(path)
        cs = crop_slice(img.shape)
        log(f'== {name} ==  shape={img.shape}', LOG)

        # 1) alpha 비교 (sigma 고정)
        panels = [(img, 'original')]
        for a in ALPHAS:
            two = unsharp(img, a, SIGMA)
            one = unsharp_single_kernel(img, a, SIGMA)
            log(f'  alpha={a:<4} sigma={SIGMA}: |two-step - single-kernel|max={np.abs(two - one).max():.2e}', LOG)
            panels.append((two, f'alpha={a}, sigma={SIGMA}'))
        show_grid(panels, f'{OUT_DIR}/{name}_alpha.png')
        show_grid(panels, f'{OUT_DIR}/{name}_alpha_crop.png', crop=cs)

        # 2) sigma 비교 (alpha 고정)
        panels = [(img, 'original')]
        for s in SIGMAS:
            out = unsharp(img, ALPHA_FIXED, s)
            one = unsharp_single_kernel(img, ALPHA_FIXED, s)
            log(f'  alpha={ALPHA_FIXED:<4} sigma={s:<4} (kernel {kernel_size(s)}x{kernel_size(s)}): '
                f'|two-step - single-kernel|max={np.abs(out - one).max():.2e}', LOG)
            panels.append((out, f'alpha={ALPHA_FIXED}, sigma={s}'))
        show_grid(panels, f'{OUT_DIR}/{name}_sigma.png')
        show_grid(panels, f'{OUT_DIR}/{name}_sigma_crop.png', crop=cs)

        # 3) BEST: alpha=ALPHA_FIXED, sigma=SIGMA (원본 vs 결과)
        best = unsharp(img, ALPHA_FIXED, SIGMA)
        log(f'  BEST alpha={ALPHA_FIXED} sigma={SIGMA}', LOG)
        panels = [(img, 'original'), (best, f'BEST (alpha={ALPHA_FIXED}, sigma={SIGMA})')]
        show_grid(panels, f'{OUT_DIR}/{name}_best.png', cols=2)
        show_grid(panels, f'{OUT_DIR}/{name}_best_crop.png', crop=cs, cols=2)
        plt.imsave(f'{OUT_DIR}/{name}_best_result.png', np.clip(best, 0, 1))
