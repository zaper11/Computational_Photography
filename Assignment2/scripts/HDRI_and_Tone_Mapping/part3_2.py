import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from Image_Blending.part1_1 import log
from HDRI_and_Tone_Mapping.hdr_common import (DATASETS, load_bracket, load_ldr,
                                              w_unif, w_tent, w_gaussian, dt_label)

BASE = '/home/juneyoung-yoon/Desktop/Computational_Photography/Assignment2/results/HDRI_and_Tone_Mapping'
CRF_DIR = f'{BASE}/part3_1'     # part3_1.py가 저장한 crf.npz
OUT_DIR = f'{BASE}/part3_2'

EPS = 1e-6
WEIGHTS = {'uniform': w_unif, 'tent': w_tent, 'gaussian': w_gaussian}
CHOSEN = {'image_1': 'tent', 'office': 'tent', 'tower': 'tent'}   # .hdr로 저장할 가중치
THUMB_WIDTH = 600


def merge_hdr(paths, dts, g):
    # I_HDR = exp( sum_j w(I_ij) (log(I_ij,lin + eps) - log dt_j) / sum_j w(I_ij) ),  I_lin = exp(g(I))
    # w, log(I_lin + eps)는 8-bit 값에만 의존하므로 lookup table로 계산한다.
    z = np.arange(256) / 255
    log_lin = np.log(np.exp(g) + EPS).astype(np.float32)                 # (3, 256)
    wl = {k: f(z).astype(np.float32) for k, f in WEIGHTS.items()}        # 가중치 LUT (256,)
    dist = np.abs(z - 0.5).astype(np.float32)

    first = load_ldr(paths[0])
    H, W = first.shape[:2]
    num = {k: np.zeros((3, H, W), np.float32) for k in WEIGHTS}
    den = {k: np.zeros((3, H, W), np.float32) for k in WEIGHTS}
    best = np.full((3, H, W), np.inf, np.float32)   # 가중치 합이 0인 픽셀용: z가 0.5에 가장 가까운 노출의 값
    fallback = np.zeros((3, H, W), np.float32)

    for j, (p, dt) in enumerate(zip(paths, dts)):
        img = first if j == 0 else load_ldr(p)
        for c in range(3):
            v = img[..., c]
            term = log_lin[c][v] - np.float32(np.log(dt))    # log(I_lin + eps) - log dt_j
            for k in WEIGHTS:
                w = wl[k][v]
                num[k][c] += w * term
                den[k][c] += w
            d = dist[v]
            better = d < best[c]
            best[c][better] = d[better]
            fallback[c][better] = term[better]

    hdr, zero_frac = {}, {}
    for k in WEIGHTS:
        ok = den[k] > 0
        log_e = fallback.copy()
        np.divide(num[k], den[k], out=log_e, where=ok)       # 가중치 합이 0이면 fallback 유지
        hdr[k] = np.exp(log_e).transpose(1, 2, 0)            # (H, W, 3)
        zero_frac[k] = 1.0 - ok.mean()
    return hdr, zero_frac


LUM = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def tonemap_display(x, Lm):
    # 비교용 간단한 표시 (global Reinhard + gamma). 정식 tone mapping은 part3_3에서 구현
    # Lm을 모든 가중치가 공유해야 같은 기준으로 비교된다
    L = x @ LUM
    Ls = 0.18 / Lm * L
    Ld = Ls / (1 + Ls)
    return np.clip(x * (Ld / (L + 1e-8))[..., None], 0, 1) ** (1 / 2.2)


def preview(hdr, stride, Lm):
    return tonemap_display(hdr[::stride, ::stride], Lm)


def find_region(score, size):
    # score가 가장 큰 위치 주변 size x size 영역 (가장자리 제외)
    h = size // 2
    s = score.copy()
    s[:h] = -np.inf; s[-h:] = -np.inf; s[:, :h] = -np.inf; s[:, -h:] = -np.inf
    y, x = np.unravel_index(np.argmax(s), s.shape)
    return slice(y - h, y + h), slice(x - h, x + h)


def diff_figure(hdr, stride, Lm, out_dir, LOG):
    # 가중치 간 차이: |log2 E_k - log2 E_tent| (stops)와, 차이가 큰 곳 / 어두운 곳의 확대 비교
    H, W = hdr['tent'].shape[:2]
    size = max(96, int(0.04 * W)) // 2 * 2
    others = ['uniform', 'gaussian']
    d = {k: np.abs(np.log2(hdr[k] / hdr['tent'])).mean(axis=-1) for k in others}
    for k in others:
        log(f'  |log2 E_{k} - log2 E_tent| : mean={d[k].mean():.4f} stops, 99.9 percentile={np.percentile(d[k][::stride, ::stride], 99.9):.3f} stops', LOG)

    kb = (size // 2) | 1
    sm = cv2.blur(d['uniform'], (kb, kb))
    reg_a = find_region(sm, size)                                   # uniform vs tent 차이가 가장 큰 곳
    lum = cv2.blur(np.ascontiguousarray(hdr['tent'] @ LUM), (kb, kb))
    dark = lum <= np.percentile(lum[::stride, ::stride], 20)
    reg_b = find_region(np.where(dark, sm, -np.inf), size)          # 어두운 영역 중 차이가 가장 큰 곳
    log(f'  crop A (max diff) y={reg_a[0].start}, x={reg_a[1].start} | crop B (dark) y={reg_b[0].start}, x={reg_b[1].start} | size={size}px', LOG)

    vmax = max(0.05, float(np.percentile(d['uniform'][::stride, ::stride], 99.5)))
    fig, ax = plt.subplots(3, 3, figsize=(17, 15))
    for a, k in zip(ax[0, :2], others):
        im = a.imshow(d[k][::stride, ::stride], cmap='magma', vmin=0, vmax=vmax)
        a.set_title(f'|log2 E_{k} - log2 E_tent| (stops)')
        a.axis('off')
    fig.colorbar(im, ax=ax[0, :2], shrink=0.7, label='stops')
    ax[0, 2].imshow(preview(hdr['tent'], stride, Lm)); ax[0, 2].axis('off'); ax[0, 2].set_title('tent  (A: red, B: cyan)')
    for reg, col in ((reg_a, 'r'), (reg_b, 'c')):
        ax[0, 2].add_patch(plt.Rectangle((reg[1].start / stride, reg[0].start / stride), size / stride, size / stride,
                                         fill=False, edgecolor=col, lw=2))
    for row, (reg, tag) in enumerate(((reg_a, 'A: max difference'), (reg_b, 'B: dark region')), start=1):
        for a, k in zip(ax[row], WEIGHTS):
            a.imshow(tonemap_display(hdr[k][reg], Lm), interpolation='nearest')
            a.set_title(f'{tag} - {k}'); a.axis('off')
    plt.savefig(f'{out_dir}/weights_diff.png', dpi=130, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def run(name):
    out_dir = f'{OUT_DIR}/{name}'
    LOG = f'{out_dir}/log.txt'
    os.makedirs(out_dir, exist_ok=True)
    open(LOG, 'w').close()
    log(f'== {name} ==', LOG)

    paths, dts = load_bracket(name)
    g = np.load(f'{CRF_DIR}/{name}/crf.npz')['g']
    log(f'{len(paths)} exposures: ' + ', '.join(dt_label(d) for d in dts) + f' | CRF lambda={float(np.load(f"{CRF_DIR}/{name}/crf.npz")["lam"]):g}', LOG)

    hdr, zero_frac = merge_hdr(paths, dts, g)
    H, W = hdr['tent'].shape[:2]
    stride = max(1, round(W / THUMB_WIDTH))
    for k in WEIGHTS:
        log(f'  {k:9s}: pixels with sum(w)=0 -> fallback: {zero_frac[k]:.3%}', LOG)

    # 선택한 가중치 결과를 Radiance .hdr로 저장
    chosen = CHOSEN[name]
    hdr_path = f'{out_dir}/{name}.hdr'
    cv2.imwrite(hdr_path, np.ascontiguousarray(hdr[chosen][..., ::-1]).astype(np.float32))
    back = cv2.imread(hdr_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    lum = hdr[chosen][::stride, ::stride] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    lo, hi = np.percentile(lum, [0.1, 99.9])
    Lm = float(np.exp(np.mean(np.log(lum + 1e-8))))
    log(f'saved {name}.hdr (weight={chosen}), re-read shape={back.shape}, '
        f'dynamic range (0.1~99.9 percentile) = {np.log2(hi / lo):.1f} stops', LOG)

    # 가중치 함수와 가중치별 결과 비교
    zz = np.linspace(0, 1, 512)
    fig, ax = plt.subplots(1, 4, figsize=(22, 5.5), gridspec_kw={'width_ratios': [1, 1.4, 1.4, 1.4]})
    for k, f in WEIGHTS.items():
        ax[0].plot(zz, f(zz), label=k)
    ax[0].set_title('weighting functions (Zmin=0.05, Zmax=0.95)'); ax[0].set_xlabel('z'); ax[0].legend(); ax[0].grid(True, alpha=0.3)
    for a, k in zip(ax[1:], WEIGHTS):
        a.imshow(preview(hdr[k], stride, Lm)); a.axis('off')
        a.set_title(f'{k}' + ('  (saved)' if k == chosen else ''))
    plt.tight_layout()
    plt.savefig(f'{out_dir}/weights_compare.png', dpi=130, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    diff_figure(hdr, stride, Lm, out_dir, LOG)


if __name__ == '__main__':
    for name in DATASETS:
        run(name)
