import rawpy
import numpy as np
import matplotlib.pyplot as plt

from part1 import images, image_name, log
from part2 import load_linear
from part3_1 import PATTERN, make_masks

out_dir = '/home/juneyoung-yoon/Desktop/Assignment1/results/1/part3/3.2'
LOG = f'{out_dir}/log.txt'

N = 8               # crop 크기


def to_string(idx_2x2, color_desc):
    desc = color_desc.decode()
    return ''.join(desc[i] for i in np.asarray(idx_2x2).flatten())


def colored_mosaic(values, MR, MG, MB):
    out = np.stack([values * MR, values * MG, values * MB], axis=-1)
    return out / max(out.max(), 1e-6)


if __name__ == '__main__':
    open(LOG, 'w').close()
    for path in images:
        with rawpy.imread(path) as raw:
            raw_image, linear = load_linear(raw)
            meta_full = to_string(raw.raw_pattern, raw.color_desc)                    # 센서 전체 기준
            meta_visible = to_string(raw.raw_colors_visible[:2, :2], raw.color_desc)  # visible 기준

        name = image_name(path)

        M_mine = make_masks(raw_image.shape, PATTERN)
        M_meta = make_masks(raw_image.shape, meta_visible)
        assert np.all(M_mine[0].astype(int) + M_mine[1] + M_mine[2] == 1)

        same_mask = all(np.array_equal(a, b) for a, b in zip(M_mine, M_meta))
        log(f'== {name} ==', LOG)
        log(f'  3.1 결과={PATTERN}, 메타(raw_pattern)={meta_full}, 메타(visible)={meta_visible}', LOG)
        log(f'  문자열 일치={PATTERN == meta_visible}, 마스크 배열 일치={same_mask}', LOG)

        crop = (slice(0, N), slice(0, N))
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        ax[0].imshow(linear[crop], cmap='gray')
        ax[0].set_title('RAW crop')
        for a, Ms, t in zip(ax[1:3], (M_mine, M_meta), (f'3.1: {PATTERN}', f'meta: {meta_visible}')):
            a.imshow(np.stack([M[crop] for M in Ms], axis=-1).astype(float))
            a.set_title(t)
        ax[3].imshow(colored_mosaic(linear[crop], *[M[crop] for M in M_meta]))
        ax[3].set_title('RAW + CFA color')
        for a in ax:
            a.set_xticks(range(N))
            a.set_yticks(range(N))
        plt.savefig(f'{out_dir}/{name}_compare.png', dpi=150, bbox_inches='tight')
        plt.close()
