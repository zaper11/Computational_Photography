import os
import numpy as np
import matplotlib.pyplot as plt

IMGS = [
    '/home/juneyoung-yoon/Desktop/Computational_Photography/Assignment2/Image_Blending/apple.png',
    '/home/juneyoung-yoon/Desktop/Computational_Photography/Assignment2/Image_Blending/orange.png']
OUT_DIR = '/home/juneyoung-yoon/Desktop/Computational_Photography/Assignment2/results/Image_Blending/part1_1'
LOG = f'{OUT_DIR}/log.txt'

DEPTH = 7


def get_gaussian_kernel(size, sigma):
    # size x size 가우시안 커널 (합 = 1)
    kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.e ** ((-1 * ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2)) / (2 * sigma ** 2)), (size, size))
    kernel /= np.sum(kernel)
    return kernel


def conv2d(image, kernel, pad_mode='symmetric'):
    # image: (m, n, c). 경계는 대칭 패딩 (zero padding 시 가장자리가 어두워짐)
    h, w = image.shape[:2]
    h_k, w_k = kernel.shape
    pad_h, pad_w = h_k // 2, w_k // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode=pad_mode)

    # 커널 뒤집기 (convolution)
    kernel = kernel[::-1, ::-1]
    out = np.zeros(image.shape, dtype=np.float64)

    # 커널 원소 (i, j)마다 (i, j)만큼 민 이미지에 가중치를 곱해 누적
    for i in range(h_k):
        for j in range(w_k):
            out += kernel[i, j] * padded[i:i + h, j:j + w]
    return out


KSIZE = 5   # Gaussian 커널 크기 (기본 5x5)


def kernel_sigma(ksize):
    # 커널 크기에 맞춘 sigma: 커널 반경 = 2 sigma. ksize=5 -> sigma=1
    return (ksize - 1) / 4


def reduce(image, ksize=KSIZE):
    # Gaussian blur 후 1/2 subsample
    kernel = get_gaussian_kernel(ksize, kernel_sigma(ksize))
    return conv2d(image, kernel)[::2, ::2]


def expand(image, shape=None, ksize=KSIZE):
    # zero 삽입으로 2배 upsample -> blur -> x4 (zero 삽입으로 줄어든 에너지 보정)
    # shape: 목표 (h, w). 홀수 크기 레벨에서 원래 shape와 맞추기 위해 사용
    h, w = (2 * image.shape[0], 2 * image.shape[1]) if shape is None else shape[:2]
    up = np.zeros((h, w, image.shape[2]), dtype=np.float64)
    src = image[:(h + 1) // 2, :(w + 1) // 2]
    up[::2, ::2][:src.shape[0], :src.shape[1]] = src

    kernel = get_gaussian_kernel(ksize, kernel_sigma(ksize))
    return conv2d(up, kernel) * 4


def gaussian_pyramid(image, depth=DEPTH, ksize=KSIZE):
    pyr = [image.astype(np.float64)]
    for _ in range(1, depth):
        pyr.append(reduce(pyr[-1], ksize))
    return pyr


def laplacian_pyramid(image, depth=DEPTH, ksize=KSIZE):
    G = gaussian_pyramid(image, depth, ksize)
    # L_k = G_k - expand(G_{k+1}), 마지막 레벨은 G 그대로
    lp = [G[k] - expand(G[k + 1], G[k].shape, ksize) for k in range(depth - 1)]
    lp.append(G[-1])
    return lp


def collapse(lp, ksize=KSIZE):
    # 가장 작은 레벨부터 expand 후 더해 올라감
    cur = lp[-1]
    for k in range(len(lp) - 2, -1, -1):
        cur = lp[k] + expand(cur, lp[k].shape, ksize)
    return cur


def load_image(path):
    img = plt.imread(path).astype(np.float64)
    if img.max() > 1:
        img /= 255.0
    if img.ndim == 2:
        img = img[..., None]
    return img[..., :3]


def log(msg, path):
    print(msg)
    with open(path, 'a') as f:
        f.write(str(msg) + '\n')


def normalize(x):
    # Laplacian 레벨 시각화용 (0 -> 0.5)
    return np.clip(x * 0.5 + 0.5, 0, 1)


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    open(LOG, 'w').close()

    for path in IMGS:
        name = os.path.basename(path).rsplit('.', 1)[0]
        img = load_image(path)

        G = gaussian_pyramid(img, DEPTH)
        L = laplacian_pyramid(img, DEPTH)
        rec = collapse(L)

        err = np.abs(rec - img)
        mse = np.mean((rec - img) ** 2)
        log(f'== {name} ==', LOG)
        log('  levels: ' + ', '.join(f'{g.shape[0]}x{g.shape[1]}' for g in G), LOG)
        log(f'  reconstruction: max|err|={err.max():.2e} MSE={mse:.2e}', LOG)

        # 1) Gaussian / Laplacian pyramid
        fig, ax = plt.subplots(2, DEPTH, figsize=(3 * DEPTH, 6.5))
        for k in range(DEPTH):
            ax[0, k].imshow(np.clip(G[k], 0, 1), interpolation='nearest')
            ax[0, k].set_title(f'Gaussian {k} ({G[k].shape[1]}x{G[k].shape[0]})')
            ax[1, k].imshow(normalize(L[k]) if k < DEPTH - 1 else np.clip(L[k], 0, 1), interpolation='nearest')
            ax[1, k].set_title(f'Laplacian {k}')
            ax[0, k].axis('off'); ax[1, k].axis('off')
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/{name}_pyramids.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # 2) 복원 결과
        fig, ax = plt.subplots(1, 3, figsize=(12, 4.5))
        for a, im, t in zip(ax, [img, np.clip(rec, 0, 1), err.max(axis=-1)],
                            ['original', 'reconstructed', f'|diff| max={err.max():.1e}']):
            a.imshow(im, cmap='gray' if im.ndim == 2 else None)
            a.set_title(t); a.axis('off')
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/{name}_reconstruction.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()
