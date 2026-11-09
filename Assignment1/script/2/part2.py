import numpy as np
import matplotlib.pyplot as plt

from part1 import conv2d, load_gray, crop_slice

IMGS = [
    '/home/juneyoung-yoon/Desktop/Assignment1/images/example.jpg',
    '/home/juneyoung-yoon/Desktop/Assignment1/images/example2.jpg',
    '/home/juneyoung-yoon/Desktop/Assignment1/images/example3.jpg']
OUT_DIR = '/home/juneyoung-yoon/Desktop/Assignment1/results/2/part2'


def make_grid(r):
    # -r .. r 좌표 meshgrid 반환
    x = np.arange(-r, r + 1)
    y = np.arange(-r, r + 1)
    return np.meshgrid(x, y)

def gaussian_kernel(sigma):
    # x, y grid 생상
    x, y = make_grid(int(np.ceil(3 * sigma)))

    # standard deviation 이용해 커널 생성
    kernel = (1 / (2 * np.pi * sigma**2))*np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel


if __name__ == '__main__':
    for path in IMGS:
        name = path.split('/')[-1].replace('.jpg', '')
        gray = load_gray(path)

        sigmas = [0.5, 1, 2, 4, 8]

        # 블러 적용
        blurred = {s: conv2d(gray, gaussian_kernel(s)) for s in sigmas}

        # 1) σ별 결과 비교 
        fig, axes = plt.subplots(1, len(sigmas) + 1, figsize=(4 * (len(sigmas) + 1), 4))
        axes[0].imshow(gray, cmap='gray'); axes[0].set_title('original')
        for ax, s in zip(axes[1:], sigmas):
            ax.imshow(blurred[s], cmap='gray')
            ax.set_title(f'sigma={s} (k={gaussian_kernel(s).shape[0]})')
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/{name}_blur_sigmas.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # 2) 커널 시각화
        fig, axes = plt.subplots(1, len(sigmas), figsize=(4 * len(sigmas), 4))
        for ax, s in zip(axes, sigmas):
            ax.imshow(gaussian_kernel(s), cmap='viridis')
            ax.set_title(f'kernel sigma={s}')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/{name}_kernels.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # 3) 중앙 크롭 σ별 비교
        cs = crop_slice(gray.shape)
        fig, axes = plt.subplots(1, len(sigmas) + 1, figsize=(4 * (len(sigmas) + 1), 4))
        axes[0].imshow(gray[cs], cmap='gray', interpolation='nearest'); axes[0].set_title('original')
        for ax, s in zip(axes[1:], sigmas):
            ax.imshow(blurred[s][cs], cmap='gray', interpolation='nearest')
            ax.set_title(f'sigma={s} (k={gaussian_kernel(s).shape[0]})')
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/{name}_blur_sigmas_crop.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()
