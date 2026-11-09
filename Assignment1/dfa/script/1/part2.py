import rawpy
import numpy as np
import matplotlib.pyplot as plt

from part1 import images, image_name

out_dir= '/home/juneyoung-yoon/Desktop/Assignment1/results/1/part2'

CROP = 256


def normalize(raw_image, black, white):
    # black level 제거 후 [0, 1]로 정규화. black은 스칼라 또는 픽셀별 배열.
    linear = (raw_image.astype(np.float32) - black) / (white - black)
    return np.clip(linear, 0, 1)


def load_linear(raw):
    # 열려 있는 rawpy 객체 -> (raw_image, linear). black level은 Bayer 위치별로 적용.
    raw_image = raw.raw_image_visible.copy()
    black = np.array(raw.black_level_per_channel)[raw.raw_colors_visible].astype(np.float32)
    return raw_image, normalize(raw_image, black, raw.white_level)


if __name__ == '__main__':
    for path in images:
        with rawpy.imread(path) as raw:
            raw_image, linear = load_linear(raw)

        name = image_name(path)
        plt.figure(figsize=(10, 7))
        plt.imshow(linear, cmap='gray')
        plt.savefig(f'{out_dir}/{name}_linear_gray.png', dpi=150, bbox_inches='tight')
        plt.show()

        # 중앙 크롭 저장 (part1과 같은 영역)
        h, w = linear.shape
        y0, x0 = (h - CROP) // 2, (w - CROP) // 2
        crop = linear[y0:y0 + CROP, x0:x0 + CROP]
        plt.figure(figsize=(6, 6))
        plt.imshow(crop, cmap='gray', interpolation='nearest')
        plt.title(f'Linear crop [{y0}:{y0 + CROP}, {x0}:{x0 + CROP}]')
        plt.savefig(f'{out_dir}/{name}_linear_crop.png', dpi=150, bbox_inches='tight')
        plt.close()
