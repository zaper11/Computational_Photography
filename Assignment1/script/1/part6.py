import rawpy
import numpy as np
import matplotlib.pyplot as plt

from part1 import images, image_name, log
from part2 import load_linear
from part3_1 import PATTERN, make_masks
from part4 import apply_gains, gains_camera
from part5 import demosaic_bilinear, to_display, CROP

out_dir = '/home/juneyoung-yoon/Desktop/Assignment1/results/1/part6'
LOG = f'{out_dir}/log.txt'


def get_cam2srgb(raw):
    # rawpy.color_matrix (3x4) 중 앞 3x3 = M_cam->sRGB
    return np.array(raw.color_matrix, dtype=np.float64)[:, :3]


def cam_to_srgb(rgb, M):
    # (H, W, 3) camera RGB -> (H, W, 3) linear sRGB. 픽셀마다 c_sRGB = M @ c_cam
    return np.clip(rgb @ M.T, 0, 1)


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

        log(f'== {name} ==', LOG)
        log('  M_cam->sRGB =', LOG)
        log(np.array2string(M, precision=4, suppress_small=True, prefix='  '), LOG)
        log(f'  row sums = {M.sum(axis=1)}', LOG)

        # 카메라 WB -> demosaic -> 색공간 변환
        bayer = apply_gains(linear, gains_camera(wb, color_desc), masks)
        cam_rgb = demosaic_bilinear(bayer, PATTERN)
        srgb = cam_to_srgb(cam_rgb, M)
        assert srgb.shape == cam_rgb.shape

        H, W = linear.shape
        crop = (slice(H // 2 - CROP, H // 2 + CROP), slice(W // 2 - CROP, W // 2 + CROP))

        results = {'camera RGB (before)': cam_rgb, 'linear sRGB (after)': srgb}
        fig, ax = plt.subplots(2, len(results), figsize=(5 * len(results), 8))
        for k, (title, img) in enumerate(results.items()):
            ax[0, k].imshow(to_display(img))
            ax[0, k].set_title(title)
            ax[1, k].imshow(to_display(img[crop]))
            ax[0, k].axis('off')
            ax[1, k].axis('off')
        plt.savefig(f'{out_dir}/{name}_cam2srgb.png', dpi=150, bbox_inches='tight')
        plt.close()
