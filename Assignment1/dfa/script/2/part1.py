import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

IMGS = [
    '/home/juneyoung-yoon/Desktop/Assignment1/images/example.jpg',
    '/home/juneyoung-yoon/Desktop/Assignment1/images/example2.jpg',
    '/home/juneyoung-yoon/Desktop/Assignment1/images/example3.jpg']
OUT_DIR = '/home/juneyoung-yoon/Desktop/Assignment1/results/2/part1'
LOG = f'{OUT_DIR}/log.txt'

CROP = 128


def box_kernel(k):
    # box filter 생성, 모든 원소 1로
    return np.ones((k, k)) / (k * k)


def conv2d(image, kernel, pad_mode='symmetric'):
    # 패딩
    h, w = image.shape
    h_k, w_k = kernel.shape

    pad_h = h_k // 2
    pad_w = w_k // 2

    padded_img = np.pad(image,((pad_h, pad_h), (pad_w, pad_w)), mode=pad_mode)

    # 커널 뒤집기
    kernel = kernel[::-1, ::-1]
    out = np.zeros((h, w))

    # 커널 원소 (i, j)마다 (i, j)만큼 민 이미지에 가중치를 곱해 누적
    for i in range(h_k):
        for j in range(w_k):
            out += kernel[i, j] * padded_img[i:i + h, j:j + w]

    return out
    

def load_gray(path, max_side=512):
    img = plt.imread(path).astype(np.float64)
    if img.max() > 1:
        img /= 255.0
    gray = img[..., :3] @ np.array([0.299, 0.587, 0.114]) if img.ndim == 3 else img
    return gray

def crop_slice(shape, size=CROP):
    # 중앙 size x size 영역
    h, w = shape
    y0, x0 = (h - size) // 2, (w - size) // 2
    return slice(y0, y0 + size), slice(x0, x0 + size)


def timeit(fn, repeat=3):
    best = float('inf')
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return out, best


def log(msg, path):
    print(msg)
    with open(path, 'a') as f:
        f.write(str(msg) + '\n')


if __name__ == '__main__':
    open(LOG, 'w').close()
    for path in IMGS:
        name = path.split('/')[-1].replace('.jpg', '')
        gray = load_gray(path)

        ks = [3, 5, 9, 15]
        t_mine, t_ref = [], []
        results = {}

        for k in ks:
            kernel = box_kernel(k)
            mine, tm = timeit(lambda: conv2d(gray, kernel))
            ref, tr = timeit(lambda: convolve2d(gray, kernel, mode='same', boundary='symm'))
            t_mine.append(tm); t_ref.append(tr)
            log(f'k={k}: mine={tm:.4f}s scipy={tr:.4f}s maxdiff={np.abs(mine-ref).max():.2e}', LOG)
            results[k] = (mine, ref)


        # 1) 결과 비교 figure
        fig, axes = plt.subplots(len(ks), 4, figsize=(16, 4 * len(ks)))
        for row, k in enumerate(ks):
            mine, ref = results[k]
            diff = np.abs(mine - ref)
            panels = [(gray, 'original'),
                      (mine, f'mine (k={k})'),
                      (ref, f'scipy (k={k})'),
                      (diff, f'|diff| max={diff.max():.1e}')]
            for col, (img, title) in enumerate(panels):
                ax = axes[row, col]
                ax.imshow(img, cmap='gray')
                ax.set_title(title)
                ax.axis('off')
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/{name}_comparison.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # 2) 커널 크기 vs 실행 시간
        plt.figure(figsize=(6, 4))
        plt.plot(ks, t_mine, 'o-', label='mine')
        plt.plot(ks, t_ref, 's-', label='scipy')
        plt.yscale('log')
        plt.xlabel('kernel size k')
        plt.ylabel('time (s)')
        plt.title('Runtime comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/{name}_runtime.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # 1-2) 중앙 크롭 결과 비교 (전체 이미지를 conv한 뒤 잘라 경계 효과 배제)
        cs = crop_slice(gray.shape)
        fig, axes = plt.subplots(len(ks), 4, figsize=(16, 4 * len(ks)))
        for row, k in enumerate(ks):
            mine, ref = results[k]
            diff = np.abs(mine - ref)
            panels = [(gray, 'original'),
                      (mine, f'mine (k={k})'),
                      (ref, f'scipy (k={k})'),
                      (diff, f'|diff| max={diff[cs].max():.1e}')]
            for col, (img, title) in enumerate(panels):
                ax = axes[row, col]
                ax.imshow(img[cs], cmap='gray', interpolation='nearest')
                ax.set_title(title)
                ax.axis('off')
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/{name}_comparison_crop.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()
