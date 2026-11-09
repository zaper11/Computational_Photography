import rawpy
import numpy as np
import matplotlib.pyplot as plt

from part1 import images, image_name, log, CROP
from part2 import load_linear
from part3_1 import PATTERN, make_masks

out_dir = '/home/juneyoung-yoon/Desktop/Assignment1/results/1/part4'
LOG = f'{out_dir}/log.txt'

WW_PCT = 100

def apply_gains(bayer, gains, masks):
    gR, gG, gB = gains
    MR, MG, MB = masks
    # 채널별 마스크로 픽셀마다 해당 색의 gain을 가진 map 생성
    gain_map = gR * MR + gG * MG + gB * MB
    return np.clip(bayer * gain_map, 0, 1)


def gains_gray_world(bayer, masks):
    # 각 채널 평균 계산
    mu = [bayer[M].mean() for M in masks]
    return mu[1] / mu[0], 1.0, mu[1] / mu[2]


def gains_white_world(bayer, masks):
    # 각 채널의 상위 WW_PCT 백분위 값(가장 밝은 값)을 흰색으로 간주
    mx = [np.percentile(bayer[M], WW_PCT) for M in masks]
    return mx[1] / mx[0], 1.0, mx[1] / mx[2]


def gains_camera(wb, color_desc):
    desc = color_desc.decode()
    r, g, b = wb[desc.index('R')], wb[desc.index('G')], wb[desc.index('B')]
    return r / g, 1.0, b / g


def preview(bayer):
    # 보간 없이 2x2 블록에서 R, B를 뽑고 G 두 개는 평균내어 절반 해상도 RGB 생성
    R = bayer[1::2, 0::2]
    B = bayer[0::2, 1::2]
    G = (bayer[0::2, 0::2] + bayer[1::2, 1::2]) / 2
    return np.clip(np.stack([R, G, B], axis=-1), 0, 1) ** (1 / 2.2)


if __name__ == '__main__':
    open(LOG, 'w').close()
    for path in images:
        with rawpy.imread(path) as raw:
            raw_image, linear = load_linear(raw)
            wb = np.array(raw.camera_whitebalance, dtype=np.float64)
            color_desc = raw.color_desc

        masks = make_masks(raw_image.shape, PATTERN)
        MR, MG, MB = masks


        name = image_name(path)

        # WB 방식별 gain (R, G, B)
        results = {
            'none': (1.0, 1.0, 1.0),
            'gray-world': gains_gray_world(linear, masks),
            'white-world': gains_white_world(linear, masks),
            'camera': gains_camera(wb, color_desc),
        }

        log(f'== {name} ==', LOG)
        fig, ax = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
        for a, (title, gains) in zip(ax, results.items()):
            log(f'  {title:12s} gains (R, G, B) = ({gains[0]:.3f}, {gains[1]:.3f}, {gains[2]:.3f})', LOG)
            # gain 적용 후 preview로 변환해 방식별 결과를 나란히 비교
            a.imshow(preview(apply_gains(linear, gains, masks)))
            a.set_title(title)
            a.axis('off')
        plt.savefig(f'{out_dir}/{name}_wb_compare.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 중앙 크롭 비교. gain은 전체 이미지 기준으로 구하고, Bayer 위상이 유지되도록 짝수 좌표에서 자름
        h, w = linear.shape
        y0, x0 = (h - CROP) // 4 * 2, (w - CROP) // 4 * 2
        crop = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
        fig, ax = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
        for a, (title, gains) in zip(ax, results.items()):
            a.imshow(preview(apply_gains(linear, gains, masks)[crop]), interpolation='nearest')
            a.set_title(title)
            a.axis('off')
        fig.suptitle(f'WB crop [{y0}:{y0 + CROP}, {x0}:{x0 + CROP}]')
        plt.savefig(f'{out_dir}/{name}_wb_compare_crop.png', dpi=150, bbox_inches='tight')
        plt.close()
