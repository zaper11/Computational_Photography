import rawpy
import numpy as np
import matplotlib.pyplot as plt

from part1 import images, image_name, log
from part7 import rgb2gray
from part9 import get_metadata, camera_isp

out_dir = '/home/juneyoung-yoon/Desktop/Assignment1/results/1/part10'
LOG = f'{out_dir}/log.txt'

CROP_FRAC = 0.08    # 확대 crop 반폭 (이미지 크기 대비 비율, 해상도가 달라도 같은 위치)


def jpeg_path(path):
    return path.replace('.dng', '.jpg')


def center_crop(img, frac=CROP_FRAC):
    # 비율 기준 중앙 crop. 세 이미지의 해상도가 달라도 같은 장면 영역을 자름
    H, W = img.shape[:2]
    h, w = int(H * frac), int(W * frac)
    return img[H // 2 - h:H // 2 + h, W // 2 - w:W // 2 + w]


def stats(img):
    # 평균 밝기 / 대비(표준편차) / 선명도(평균 기울기 크기)
    g = rgb2gray(img)
    gy, gx = np.diff(g, axis=0)[:, :-1], np.diff(g, axis=1)[:-1, :]
    return g.mean(), g.std(), np.sqrt(gx ** 2 + gy ** 2).mean()


if __name__ == '__main__':
    open(LOG, 'w').close()
    for path in images:
        name = image_name(path)

        with rawpy.imread(path) as raw:
            raw_image = raw.raw_image_visible.copy()
            meta = get_metadata(raw)
            # 이 파트에서만 postprocess 사용 허용
            ref = raw.postprocess(use_camera_wb=True, output_bps=16)

        mine = np.clip(camera_isp(raw_image, meta), 0, 1)
        ref = ref.astype(np.float64) / 65535.0
        jpg = plt.imread(jpeg_path(path)).astype(np.float64) / 255.0

        results = {'mine': mine, 'rawpy postprocess': ref, 'camera JPEG': jpg}

        log(f'== {name} ==', LOG)
        for title, img in results.items():
            m, s, sharp = stats(img)
            log(f'  {title:18s} shape={img.shape}  mean={m:.3f}  std={s:.3f}  sharpness={sharp:.4f}', LOG)

        fig, ax = plt.subplots(2, len(results), figsize=(6 * len(results), 9))
        for k, (title, img) in enumerate(results.items()):
            ax[0, k].imshow(img)
            ax[0, k].set_title(title)
            ax[1, k].imshow(center_crop(img))
            ax[0, k].axis('off')
            ax[1, k].axis('off')
        plt.savefig(f'{out_dir}/{name}_compare.png', dpi=150, bbox_inches='tight')
        plt.close()
