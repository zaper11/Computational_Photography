import rawpy
import numpy as np
import matplotlib.pyplot as plt

from part1 import images, image_name, log
from part2 import load_linear
from part3_1 import PATTERN, make_masks
from part4 import apply_gains, gains_camera
from part5 import demosaic_bilinear, CROP
from part6 import get_cam2srgb, cam_to_srgb
from part7 import adjust_brightness, TARGET

out_dir = '/home/juneyoung-yoon/Desktop/Assignment1/results/1/part8'
LOG = f'{out_dir}/log.txt'


def srgb_encode(linear):
    # linear RGB [0,1] -> 비선형 sRGB [0,1] (구간별 sRGB 전달함수)
    c = np.clip(linear, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


if __name__ == '__main__':
    open(LOG, 'w').close()
    for path in images:
        with rawpy.imread(path) as raw:
            raw_image, linear = load_linear(raw)
            wb = np.array(raw.camera_whitebalance, dtype=np.float64)
            color_desc = raw.color_desc
            M = get_cam2srgb(raw)

        masks = make_masks(raw_image.shape, PATTERN)
        name = image_name(path)

        # Part 4~7: 카메라 WB -> demosaic -> linear sRGB -> 밝기 조절
        bayer = apply_gains(linear, gains_camera(wb, color_desc), masks)
        srgb_lin = cam_to_srgb(demosaic_bilinear(bayer, PATTERN), M)
        bright, alpha = adjust_brightness(srgb_lin, TARGET)

        encoded = srgb_encode(bright)

        log(f'== {name} ==', LOG)
        log(f'  mean (linear)  = {bright.mean():.4f}', LOG)
        log(f'  mean (encoded) = {encoded.mean():.4f}', LOG)

        H, W = linear.shape
        crop = (slice(H // 2 - CROP, H // 2 + CROP), slice(W // 2 - CROP, W // 2 + CROP))

        results = {'linear RGB': bright, 'sRGB (gamma encoded)': encoded}
        fig, ax = plt.subplots(2, len(results), figsize=(5 * len(results), 8))
        for k, (title, img) in enumerate(results.items()):
            ax[0, k].imshow(np.clip(img, 0, 1))      # 감마 없이 있는 그대로 표시
            ax[0, k].set_title(title)
            ax[1, k].imshow(np.clip(img[crop], 0, 1))
            ax[0, k].axis('off')
            ax[1, k].axis('off')
        plt.savefig(f'{out_dir}/{name}_gamma.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 전달함수 곡선 (리포트용, 한 번만 그리면 되지만 이미지별로 덮어씀)
        x = np.linspace(0, 1, 512)
        plt.figure(figsize=(5, 5))
        plt.plot(x, x, '--', label='linear')
        plt.plot(x, srgb_encode(x), label='sRGB encode')
        plt.xlabel('linear'); plt.ylabel('encoded'); plt.legend(); plt.grid(alpha=0.3)
        plt.savefig(f'{out_dir}/transfer_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
