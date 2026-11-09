import rawpy
import numpy as np
import matplotlib.pyplot as plt

from part1 import images, image_name
from part2 import normalize
from part3_1 import PATTERN, make_masks
from part4 import (apply_gains, gains_camera, gains_gray_world,
                   gains_white_world, preview)
from part5 import demosaic_bilinear, to_display, CROP
from part6 import cam_to_srgb
from part7 import adjust_brightness, TARGET
from part8 import srgb_encode

out_dir = '/home/juneyoung-yoon/Desktop/Assignment1/results/1/part9'


def get_metadata(raw, wb_method='camera'):
    # rawpy 객체에서 ISP에 필요한 값만 꺼내 dict로 묶음 (픽셀 데이터는 포함하지 않음)
    return {
        'black': np.array(raw.black_level_per_channel)[raw.raw_colors_visible].astype(np.float32),
        'white': raw.white_level,
        'wb': np.array(raw.camera_whitebalance, dtype=np.float64),
        'color_desc': raw.color_desc,
        'cam2srgb': np.array(raw.color_matrix, dtype=np.float64)[:, :3],
        'pattern': PATTERN,         # Part 3.1에서 직접 판별한 값
        'wb_method': wb_method,     # 'camera' | 'gray' | 'white'
        'target': TARGET,
    }


def linearize(raw_image, metadata):
    return normalize(raw_image, metadata['black'], metadata['white'])


def white_balance(bayer, metadata):
    masks = make_masks(bayer.shape, metadata['pattern'])
    method = metadata['wb_method']
    if method == 'gray':
        gains = gains_gray_world(bayer, masks)
    elif method == 'white':
        gains = gains_white_world(bayer, masks)
    else:
        gains = gains_camera(metadata['wb'], metadata['color_desc'])
    return apply_gains(bayer, gains, masks)


def color_correction(rgb, metadata):
    return cam_to_srgb(rgb, metadata['cam2srgb'])


def adjust(rgb, metadata):
    return adjust_brightness(rgb, metadata['target'])[0]


def camera_isp(raw_image, metadata, return_stages=False):
    stages = {}
    x = stages['1 linearize'] = linearize(raw_image, metadata)
    x = stages['2 white balance'] = white_balance(x, metadata)
    x = stages['3 demosaic'] = demosaic_bilinear(x, metadata['pattern'])
    x = stages['4 color correction'] = color_correction(x, metadata)
    x = stages['5 adjust brightness'] = adjust(x, metadata)
    x = stages['6 sRGB encode'] = srgb_encode(x)
    return (x, stages) if return_stages else x


if __name__ == '__main__':
    for path in images:
        name = image_name(path)
        with rawpy.imread(path) as raw:
            raw_image = raw.raw_image_visible.copy()
            meta = get_metadata(raw)
            meta_gray = get_metadata(raw, 'gray')
            meta_white = get_metadata(raw, 'white')

        H, W = raw_image.shape
        crop = (slice(H // 2 - CROP, H // 2 + CROP), slice(W // 2 - CROP, W // 2 + CROP))

        # 1) 단계별 중간 결과 (카메라 WB)
        final, stages = camera_isp(raw_image, meta, return_stages=True)
        assert final.shape == (H, W, 3)

        shown = {
            'raw': raw_image / raw_image.max(),
            **{k: v for k, v in stages.items()},
        }
        fig, ax = plt.subplots(2, len(shown), figsize=(3.5 * len(shown), 7))
        for k, (title, img) in enumerate(shown.items()):
            if title in ('raw', '1 linearize'):
                disp = lambda a: a
                kw = {'cmap': 'gray'}
            elif title == '2 white balance':
                disp = lambda a: preview(a)      # Bayer -> 2x2 묶음 미리보기
                kw = {}
            elif title in ('3 demosaic', '4 color correction', '5 adjust brightness'):
                disp = to_display                # 아직 linear이므로 표시용 감마
                kw = {}
            else:
                disp = lambda a: np.clip(a, 0, 1)
                kw = {}
            if title == '2 white balance':
                ax[0, k].imshow(disp(img), **kw)
                ax[1, k].imshow(disp(img[crop]), **kw)
            else:
                ax[0, k].imshow(disp(img), **kw)
                ax[1, k].imshow(disp(img[crop]), **kw)
            ax[0, k].set_title(title)
            ax[0, k].axis('off')
            ax[1, k].axis('off')
        plt.savefig(f'{out_dir}/{name}_stages.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 2) WB 방식별 최종 결과 (Part 4 비교용)
        finals = {
            'gray-world': camera_isp(raw_image, meta_gray),
            'white-world': camera_isp(raw_image, meta_white),
            'camera': final,
        }
        fig, ax = plt.subplots(2, len(finals), figsize=(5 * len(finals), 8))
        for k, (title, img) in enumerate(finals.items()):
            ax[0, k].imshow(np.clip(img, 0, 1))
            ax[0, k].set_title(title)
            ax[1, k].imshow(np.clip(img[crop], 0, 1))
            ax[0, k].axis('off')
            ax[1, k].axis('off')
        plt.savefig(f'{out_dir}/{name}_wb_final.png', dpi=150, bbox_inches='tight')
        plt.close()

        plt.imsave(f'{out_dir}/{name}_mine.png', np.clip(final, 0, 1))   # Part 10 비교용
