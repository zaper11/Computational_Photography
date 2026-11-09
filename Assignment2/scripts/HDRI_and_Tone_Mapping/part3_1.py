import os
import numpy as np
import matplotlib.pyplot as plt

from Image_Blending.part1_1 import log
from HDRI_and_Tone_Mapping.hdr_common import DATASETS, load_bracket, load_ldr, w_tent, dt_label

OUT_DIR = '/home/juneyoung-yoon/Desktop/Computational_Photography/Assignment2/results/HDRI_and_Tone_Mapping/part3_1'  # 아래에 세트별 폴더

N_SAMPLES = 1000                # 샘플 픽셀 수 N (권장 500 ~ 1000)
LAMBDA = 50.0                   # 기본 lambda (권장 10 ~ 100)
LAMBDAS = [1, 10, 100, 1000]    # lambda 영향 비교
SEED = 0
THUMB_WIDTH = 600               # figure 표시용 다운샘플 목표 너비
CRF_ZMIN, CRF_ZMAX = 0.0, 1.0   # CRF 추정용 tent: 구간 cutoff 없이 min(z, 1-z)
COLORS = ['r', 'g', 'b']
CHANNELS = ['R', 'G', 'B']


def weight(z):
    return w_tent(z, CRF_ZMIN, CRF_ZMAX)


def sample_pixels(paths, n, seed=SEED):
    # 모든 노출에서 같은 위치의 픽셀 n개를 뽑는다. 반환: Z (n, P, 3) uint8, thumbs (표시용)
    rng = np.random.default_rng(seed)
    first = load_ldr(paths[0])
    H, W = first.shape[:2]
    ys, xs = rng.integers(0, H, n), rng.integers(0, W, n)

    Z = np.empty((n, len(paths), 3), dtype=np.uint8)
    stride = max(1, round(W / THUMB_WIDTH))
    thumbs = []
    for j, p in enumerate(paths):
        img = first if j == 0 else load_ldr(p)
        Z[:, j] = img[ys, xs]
        thumbs.append(img[::stride, ::stride].copy())
    return Z, thumbs, (H, W)


def build_system(Z, log_dt, lam):
    # Z: (N, P) 한 채널의 샘플 intensity, 미지수 v = [g(0..255), log E_1..N]
    # 행: N*P 데이터 식 + 1 스케일 식 + 254 smoothness 식
    N, P = Z.shape
    A = np.zeros((N * P + 1 + 254, 256 + N))
    b = np.zeros(N * P + 1 + 254)

    # 1.2) 데이터 식: w_ij g(I_ij) - w_ij log E_i = w_ij log dt_j
    i_idx = np.repeat(np.arange(N), P)
    j_idx = np.tile(np.arange(P), N)
    z = Z.reshape(-1).astype(int)
    w_ij = weight(z / 255)
    rows = np.arange(N * P)
    A[rows, z] = w_ij
    A[rows, 256 + i_idx] = -w_ij
    b[rows] = w_ij * log_dt[j_idx]

    # 1.3) 스케일 고정: g(128) = 0
    A[N * P, 128] = 1.0

    # 1.4) smoothness: sqrt(lambda) w(z/255) [g(z-1) - 2 g(z) + g(z+1)] = 0, z = 1..254
    zs = np.arange(1, 255)
    r = N * P + 1 + (zs - 1)
    c = np.sqrt(lam) * weight(zs / 255)
    A[r, zs - 1] = c
    A[r, zs] = -2 * c
    A[r, zs + 1] = c
    return A, b


def solve_crf(Z, log_dt, lam):
    # 채널별 least squares. 반환: g (3, 256), log E (3, N)
    g, logE = [], []
    for ch in range(3):
        A, b = build_system(Z[:, :, ch], log_dt, lam)
        v = np.linalg.lstsq(A, b, rcond=None)[0]
        g.append(v[:256])
        logE.append(v[256:])
    return np.array(g), np.array(logE)


def linearize(img, g):
    # I_lin = exp(g(I)), 채널별 lookup table
    out = np.empty(img.shape, dtype=np.float32)
    for ch in range(3):
        out[..., ch] = np.exp(g[ch]).astype(np.float32)[img[..., ch]]
    return out


def run(name):
    out_dir = f'{OUT_DIR}/{name}'
    LOG = f'{out_dir}/log.txt'
    os.makedirs(out_dir, exist_ok=True)
    open(LOG, 'w').close()
    log(f'== {name} ==', LOG)

    paths, dts = load_bracket(name)
    log_dt = np.log(dts)
    log(f'{len(paths)} exposures: ' + ', '.join(dt_label(d) for d in dts), LOG)

    Z, thumbs, (H, W) = sample_pixels(paths, N_SAMPLES)
    log(f'image {W}x{H}, N={N_SAMPLES} sampled pixels, P={len(paths)}', LOG)
    log(f'A shape = ({N_SAMPLES * len(paths) + 255}, {256 + N_SAMPLES})', LOG)

    # lambda별 CRF 추정
    lams = sorted(set(LAMBDAS + [LAMBDA]))
    sols = {}
    zz = np.arange(256)
    for lam in lams:
        g, logE = solve_crf(Z, log_dt, lam)
        sols[lam] = (g, logE)
        d2 = g[:, 2:] - 2 * g[:, 1:-1] + g[:, :-2]
        res = [np.sqrt(np.mean((weight(Z[:, :, c] / 255) * (g[c][Z[:, :, c]] - logE[c][:, None] - log_dt[None, :])) ** 2))
               for c in range(3)]
        log(f'lambda={lam:<6}: weighted data residual RMS (R,G,B) = '
            + ', '.join(f'{r:.4f}' for r in res) + f' | mean |d2 g| = {np.abs(d2).mean():.4f}', LOG)

    g, logE = sols[LAMBDA]
    np.savez(f'{out_dir}/crf.npz', g=g, lam=LAMBDA, dts=dts)

    # 1) g, f 플롯 (기본 lambda)
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.8))
    rng = np.random.default_rng(1)
    show = rng.choice(N_SAMPLES, 150, replace=False)
    for ch in range(3):
        a = ax[ch]
        pts_x = Z[show, :, ch].reshape(-1)
        pts_y = (logE[ch][show][:, None] + log_dt[None, :]).reshape(-1)
        a.scatter(pts_x, pts_y, s=3, color=COLORS[ch], alpha=0.25, label='samples: log E_i + log dt_j')
        a.plot(zz, g[ch], 'k-', lw=1.5, label='g(z)')
        a.set_title(f'g = log f^-1  ({CHANNELS[ch]}, lambda={LAMBDA:g})')
        a.set_xlabel('pixel value z'); a.set_ylabel('g(z)'); a.grid(True, alpha=0.3); a.legend(loc='lower right')
        ax[3].plot(np.exp(g[ch]), zz, COLORS[ch], label=CHANNELS[ch])
    ax[3].set_xscale('log'); ax[3].set_xlabel('exposure  E * dt  (relative)'); ax[3].set_ylabel('pixel value')
    ax[3].set_title('f: exposure -> pixel value'); ax[3].grid(True, alpha=0.3, which='both'); ax[3].legend()
    plt.tight_layout()
    plt.savefig(f'{out_dir}/crf.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # 2) lambda의 영향
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    cmap = plt.get_cmap('viridis')
    for ch in range(3):
        for k, lam in enumerate(lams):
            gl = sols[lam][0][ch]
            col = cmap(k / max(len(lams) - 1, 1))
            ax[0, ch].plot(zz, gl, color=col, label=f'lambda={lam:g}')
            ax[1, ch].plot(zz[1:-1], gl[2:] - 2 * gl[1:-1] + gl[:-2], color=col, label=f'lambda={lam:g}')
        ax[0, ch].set_title(f'g(z)  ({CHANNELS[ch]})'); ax[0, ch].set_xlabel('z'); ax[0, ch].set_ylabel('g(z)')
        ax[1, ch].set_title(f'second difference of g  ({CHANNELS[ch]})'); ax[1, ch].set_xlabel('z')
        for a in ax[:, ch]:
            a.grid(True, alpha=0.3); a.legend()
    plt.tight_layout()
    plt.savefig(f'{out_dir}/lambda_effect.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # 3) 선형화 결과 (표시용 다운샘플. LUT는 픽셀 단위라 다운샘플 후 적용해도 결과가 같다)
    P = len(paths)
    lin = [linearize(t, g) for t in thumbs]
    lin_disp_scale = np.exp(g).max()                       # 포화값(I=255) 기준 정규화
    rad = [l / dt for l, dt in zip(lin, dts)]              # I_lin / dt  ->  E 에 비례해야 함
    rad_scale = np.percentile(np.concatenate([r[..., :].ravel()[::50] for r in rad]), 99.5)
    gamma = lambda x: np.clip(x, 0, 1) ** (1 / 2.2)
    fig, ax = plt.subplots(3, P, figsize=(3 * P, 7.2), squeeze=False)
    for j in range(P):
        ax[0, j].imshow(thumbs[j]); ax[0, j].set_title(f'LDR  dt={dt_label(dts[j])}')
        ax[1, j].imshow(gamma(lin[j] / lin_disp_scale)); ax[1, j].set_title('exp(g(I))')
        ax[2, j].imshow(gamma(rad[j] / rad_scale)); ax[2, j].set_title('exp(g(I)) / dt')
    for a in ax.ravel():
        a.axis('off')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/linearized.png', dpi=110, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    log('linearized.png: rows = LDR / exp(g(I)) / exp(g(I))/dt  (display gamma 2.2)', LOG)


if __name__ == '__main__':
    for name in DATASETS:
        run(name)
