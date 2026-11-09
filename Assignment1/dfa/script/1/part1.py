import rawpy
import numpy as np
import matplotlib.pyplot as plt

images=[
    '/home/juneyoung-yoon/Desktop/Assignment1/images/example.dng',
    '/home/juneyoung-yoon/Desktop/Assignment1/images/example2.dng',
    '/home/juneyoung-yoon/Desktop/Assignment1/images/example3.dng']

out_dir= '/home/juneyoung-yoon/Desktop/Assignment1/results/1/part1'
LOG = f'{out_dir}/log.txt'

CROP = 256


def image_name(path):
    return path.split('/')[-1].replace('.dng', '')


def log(msg, path):
    print(msg)
    with open(path, 'a') as f:
        f.write(str(msg) + '\n')


if __name__ == '__main__':
    open(LOG, 'w').close()
    for path in images:
        with rawpy.imread(path) as raw:
            raw_image = raw.raw_image_visible.copy()
            log(raw_image.shape, LOG)
            log(raw_image.dtype, LOG)
            log(raw.raw_pattern, LOG)
            log(raw.color_desc, LOG)
            log(raw.black_level_per_channel, LOG)
            log(raw.white_level, LOG)
            log(raw.camera_whitebalance, LOG)

        name = image_name(path)
        plt.figure(figsize=(10, 7))
        plt.imshow(raw_image, cmap='gray')
        plt.savefig(f'{out_dir}/{name}_raw_gray.png', dpi=150, bbox_inches='tight')
        plt.show()

        # 중앙 크롭 저장 (픽셀 단위 확인용)
        h, w = raw_image.shape
        y0, x0 = (h - CROP) // 2, (w - CROP) // 2
        crop = raw_image[y0:y0 + CROP, x0:x0 + CROP]
        plt.figure(figsize=(6, 6))
        plt.imshow(crop, cmap='gray', interpolation='nearest')
        plt.title(f'RAW crop [{y0}:{y0 + CROP}, {x0}:{x0 + CROP}]')
        plt.savefig(f'{out_dir}/{name}_raw_crop.png', dpi=150, bbox_inches='tight')
        plt.close()
