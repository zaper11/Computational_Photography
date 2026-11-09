import rawpy
import numpy as np
import matplotlib.pyplot as plt

from part1 import images, image_name, log
from part2 import load_linear

out_dir = '/home/juneyoung-yoon/Desktop/Assignment1/results/1/part3/3.1'
LOG = f'{out_dir}/log.txt'

BLOCK = 32  # 위상 평면 기준
PATTERN = 'GBRG'    # 판별결과


def block_mean(p, b):
    h, w = p.shape
    h, w = h - h % b, w - w % b
    return p[:h, :w].reshape(h // b, b, w // b, b).mean(axis=(1, 3))

def make_masks(shape, pattern):
    H, W = shape
    tile = np.array(list(pattern)).reshape(2, 2)
    yy = np.arange(H)[:, None] % 2
    xx = np.arange(W)[None, :] % 2
    cfa = tile[yy, xx]
    return (cfa == 'R'), (cfa == 'G'), (cfa == 'B')


if __name__ == '__main__':
    open(LOG, 'w').close()
    for path in images:
        with rawpy.imread(path) as raw:
            raw_image, linear = load_linear(raw)


        name = image_name(path)

        # 2x2 위상 평균/중앙값
        phases = {
            '(0,0)': linear[0::2, 0::2],
            '(0,1)': linear[0::2, 1::2],
            '(1,0)': linear[1::2, 0::2],
            '(1,1)': linear[1::2, 1::2],
        }

        # G 대각선 판별
        log(f'== {name} ==', LOG)
        for k, p in phases.items():
            log(f'  {k}: mean={p.mean():.4f}  median={np.median(p):.4f}', LOG)

        m = {k: p.mean() for k, p in phases.items()}
        log(f'왼->오 대각선 차이={abs(m["(0,0)"] - m["(1,1)"]):.4f}, '
              f'오->왼 대각선 차이={abs(m["(0,1)"] - m["(1,0)"]):.4f}', LOG)

        # 하늘, 바닥 등 기존에 아는 색에서 R,B 자연스러운지 확인

        g   = (block_mean(phases['(0,0)'], BLOCK) + block_mean(phases['(1,1)'], BLOCK)) / 2
        p01 = block_mean(phases['(0,1)'], BLOCK)
        p10 = block_mean(phases['(1,0)'], BLOCK)

        eps = 1e-6
        valid = (g > 0.02) & (g < 0.9)

        log_ratio = np.log2((p01 + eps) / (p10 + eps))
        log_ratio = np.where(valid, log_ratio, np.nan)

        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        ax[0].imshow(g, cmap='gray')
        ax[0].set_title('G (block mean)')
        im = ax[1].imshow(log_ratio, cmap='bwr', vmin=-2, vmax=2)
        ax[1].set_title('log2( (0,1) / (1,0) )')
        plt.colorbar(im, ax=ax[1])
        plt.savefig(f'{out_dir}/{name}_block_ratio.png', dpi=150, bbox_inches='tight')
        plt.show()

        # gbrg로 판별함, 마스킹 짆애

        MR, MG, MB = make_masks(raw_image.shape, PATTERN)
        assert np.all(MR.astype(int) + MG + MB == 1)
        log(f'  mask 비율 R/G/B = {MR.mean():.2f} / {MG.mean():.2f} / {MB.mean():.2f}', LOG)

        N = 8
        crop = (slice(0, N), slice(0, N))
        cfa_rgb = np.stack([MR[crop], MG[crop], MB[crop]], axis=-1).astype(float)

        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        for a, M, t in zip(ax[:3], (MR, MG, MB), ('M_R', 'M_G', 'M_B')):
            a.imshow(M[crop], cmap='gray', vmin=0, vmax=1)
            a.set_title(t)
        ax[3].imshow(cfa_rgb)
        ax[3].set_title(f'CFA ({PATTERN})')
        for a in ax:
            a.set_xticks(range(N)); a.set_yticks(range(N))
        plt.savefig(f'{out_dir}/{name}_cfa_masks.png', dpi=150, bbox_inches='tight')
        plt.show()
