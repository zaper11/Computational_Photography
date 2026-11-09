import os
import numpy as np
import matplotlib.pyplot as plt

from Image_Blending.part1_1 import (gaussian_pyramid, laplacian_pyramid, collapse, load_image, log, normalize,
                                    KSIZE, kernel_sigma)

IMG_A = '/home/juneyoung-yoon/Desktop/Computational_Photography/Assignment2/Image_Blending/apple.png'
IMG_B = '/home/juneyoung-yoon/Desktop/Computational_Photography/Assignment2/Image_Blending/orange.png'
OUT_DIR = '/home/juneyoung-yoon/Desktop/Computational_Photography/Assignment2/results/Image_Blending/part1_2'
LOG = f'{OUT_DIR}/log.txt'

DEPTH = 7
DEPTHS = [1, 3, 5, 7]        # 레벨 수 비교 (커널 5x5 고정)
KSIZES = [3, 5, 9, 15]       # 커널 크기 비교 (레벨 7 고정)
CROP = 128                   # 이음새 확대 크기


def make_mask(shape, split=0.5):
    # 왼쪽 split 비율은 1 (A = apple), 나머지는 0 (B = orange)인 binary mask
    h, w = shape[:2]
    mask = np.zeros((h, w, 3))
    mask[:, :int(w * split)] = 1.0
    return mask


def blend(img_a, img_b, mask, depth=DEPTH, ksize=KSIZE):
    # L_k = L^A_k * m_k + L^B_k * (1 - m_k), m_k: mask의 Gaussian pyramid
    LA = laplacian_pyramid(img_a, depth, ksize)
    LB = laplacian_pyramid(img_b, depth, ksize)
    GM = gaussian_pyramid(mask, depth, ksize)
    LS = [la * m + lb * (1 - m) for la, lb, m in zip(LA, LB, GM)]
    return collapse(LS, ksize), LA, LB, GM, LS


def compare_settings(A, B, mask, settings, path):
    # settings: [(title, depth, ksize)]. 위: 전체 결과, 아래: 경계(이음새) 중앙 확대
    h, w = A.shape[:2]
    cy, cx, r = h // 2, w // 2, CROP // 2
    fig, ax = plt.subplots(2, len(settings), figsize=(4 * len(settings), 8.3))
    for col, (title, depth, ksize) in enumerate(settings):
        out = np.clip(blend(A, B, mask, depth, ksize)[0], 0, 1)
        ax[0, col].imshow(out); ax[0, col].set_title(title)
        ax[0, col].add_patch(plt.Rectangle((cx - r, cy - r), CROP, CROP, fill=False, edgecolor='r', lw=1.5))
        ax[1, col].imshow(out[cy - r:cy + r, cx - r:cx + r], interpolation='nearest')
        ax[1, col].set_title(f'{title} (seam crop)')
        for a in ax[:, col]:
            a.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    open(LOG, 'w').close()

    A, B = load_image(IMG_A), load_image(IMG_B)
    assert A.shape == B.shape, 'two images must have the same size'
    mask = make_mask(A.shape)

    blended, LA, LB, GM, LS = blend(A, B, mask)
    blended = np.clip(blended, 0, 1)
    # 비교용: pyramid 없이 mask로 바로 이어붙인 결과
    naive = A * mask + B * (1 - mask)

    log(f'blended: shape={blended.shape}, range=[{blended.min():.3f}, {blended.max():.3f}]', LOG)
    log(f'mean |blended - naive| = {np.abs(blended - naive).mean():.4f}', LOG)

    plt.imsave(f'{OUT_DIR}/blended.png', blended)

    # 1) 입력 / mask / 결과 비교
    fig, ax = plt.subplots(1, 5, figsize=(20, 4.5))
    for a, im, t in zip(ax, [A, B, mask, naive, blended],
                        ['image A', 'image B', 'mask', 'direct paste', 'multi-band blend']):
        a.imshow(im, cmap='gray'); a.set_title(t)
        a.set_xticks([]); a.set_yticks([])  # 테두리 유지 (흰 mask 영역이 배경에 묻히지 않게)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/comparison.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # 2) 레벨별 L^A, L^B, mask Gaussian, blended Laplacian
    rows = [('L^A', LA), ('L^B', LB), ('mask G', GM), ('blended L', LS)]
    fig, ax = plt.subplots(len(rows), DEPTH, figsize=(3 * DEPTH, 3.2 * len(rows)))
    for r, (title, pyr) in enumerate(rows):
        for k in range(DEPTH):
            if title == 'mask G' or k == DEPTH - 1:
                im = np.clip(pyr[k], 0, 1)
            else:
                im = normalize(pyr[k])
            ax[r, k].imshow(im, interpolation='nearest')
            ax[r, k].set_title(f'{title} {k}')
            ax[r, k].axis('off')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/pyramids.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # 3) 질문: pyramid 레벨 수 / Gaussian 커널 크기에 따른 변화
    compare_settings(A, B, mask, [(f'levels={d}', d, KSIZE) for d in DEPTHS], f'{OUT_DIR}/levels.png')
    compare_settings(A, B, mask, [(f'kernel {k}x{k} (sigma={kernel_sigma(k):g})', DEPTH, k) for k in KSIZES],
                     f'{OUT_DIR}/kernel_size.png')
    log(f'levels.png: levels {DEPTHS} (kernel {KSIZE}x{KSIZE}) | kernel_size.png: kernels {KSIZES} (levels {DEPTH}), sigma=(ksize-1)/4', LOG)
