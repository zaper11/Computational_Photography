import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from Image_Blending.part1_1 import log
from HDRI_and_Tone_Mapping.hdr_common import DATASETS, HDRI_DIR, load_bracket, load_ldr, dt_label

BASE = '/home/juneyoung-yoon/Desktop/Computational_Photography/Assignment2/results/HDRI_and_Tone_Mapping'
OUT_DIR = f'{BASE}/part3_3'

SOURCES = {n: f'{BASE}/part3_2/{n}/{n}.hdr' for n in DATASETS}
SOURCES['MMSPG_0026'] = f'{HDRI_DIR}/MMSPG_0026.hdr'

# scaled luminance의 key value a, 수업에서 얘기한 일반적으로 사용하는 값은 a = 0.18
KEY = 0.18  
DELTA = 1e-6
# Reinhard: Lwhite = B * max(L)
B_LIST = [0.1, 0.25, 0.5, 1.0]
# Logarithmic
BETA_LIST = [1, 10, 100, 1000]
# (B, beta): 전체 해상도로 저장할 설정
CHOSEN = {name: (1.0, 10) for name in SOURCES}
THUMB_WIDTH = 600
LUM = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def load_hdr(path):
    x = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    return np.maximum(np.ascontiguousarray(x[..., ::-1]), 0)      # BGR -> RGB


def reinhard(L, Lwhite):
    # Ld = L (1 + L / Lwhite^2) / (1 + L)
    return L * (1 + L / Lwhite ** 2) / (1 + L)


def logarithmic(L, beta, Lmax):
    # Ld = log(1 + beta L) / log(1 + beta Lmax)
    return np.log1p(beta * L) / np.log1p(beta * Lmax)


def tone_map(hdr, Lbar, fn):
    # 휘도 L(scaled)에 fn을 적용하고, 색은 RGB에 Ld/Lw를 곱해서 유지한다
    Lw = hdr @ LUM
    L = KEY / Lbar * Lw
    return hdr * (fn(L) / (Lw + 1e-12))[..., None]


def to_display(x):
    # [0, 1]로 clip 후 display gamma 2.2
    return np.clip(x, 0, 1) ** (1 / 2.2)


def run(name):
    out_dir = f'{OUT_DIR}/{name}'
    LOG = f'{out_dir}/log.txt'
    os.makedirs(out_dir, exist_ok=True)
    open(LOG, 'w').close()
    log(f'== {name} ==', LOG)

    hdr = load_hdr(SOURCES[name])
    H, W = hdr.shape[:2]
    Lw = hdr @ LUM
    Lbar = float(np.exp(np.mean(np.log(Lw + DELTA))))      # log-average luminance
    Lmax = float((KEY / Lbar * Lw).max())                  # max of scaled luminance
    log(f'shape={W}x{H}, log-average luminance={Lbar:.4g}, max(L) (scaled, key={KEY}) = {Lmax:.3f}', LOG)

    # 톤 매핑은 pixel 단위 + 전역 통계(Lbar, Lmax)만 쓰므로, 다운샘플해서 적용해도 결과가 같다
    stride = max(1, round(W / THUMB_WIDTH))
    small = hdr[::stride, ::stride]

    # (a) Reinhard, B 비교
    fig, ax = plt.subplots(1, len(B_LIST), figsize=(5 * len(B_LIST), 4.2), squeeze=False)
    for a, B in zip(ax[0], B_LIST):
        a.imshow(to_display(tone_map(small, Lbar, lambda L: reinhard(L, B * Lmax))))
        a.set_title(f'Reinhard  B={B}  (Lwhite={B * Lmax:.2f})'); a.axis('off')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/reinhard.png', dpi=130, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # (b) Logarithmic, beta 비교
    fig, ax = plt.subplots(1, len(BETA_LIST), figsize=(5 * len(BETA_LIST), 4.2), squeeze=False)
    for a, beta in zip(ax[0], BETA_LIST):
        a.imshow(to_display(tone_map(small, Lbar, lambda L: logarithmic(L, beta, Lmax))))
        a.set_title(f'Logarithmic  beta={beta}'); a.axis('off')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/logarithmic.png', dpi=130, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # 입력 노출 이미지 / 톤 매핑 없이 스케일만 한 결과 / 선택한 설정의 결과 비교
    B, beta = CHOSEN[name]
    panels = []
    if name in DATASETS:
        paths, dts = load_bracket(name)
        for j in sorted({0, len(paths) // 2, len(paths) - 1}):
            panels.append((load_ldr(paths[j])[::stride, ::stride], f'input exposure dt={dt_label(dts[j])}'))
    panels.append((to_display(KEY / Lbar * small), 'no tone mapping (scaled, clipped)'))
    panels.append((to_display(tone_map(small, Lbar, lambda L: reinhard(L, B * Lmax))), f'Reinhard B={B}'))
    panels.append((to_display(tone_map(small, Lbar, lambda L: logarithmic(L, beta, Lmax))), f'Logarithmic beta={beta}'))
    fig, ax = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.4), squeeze=False)
    for a, (im, t) in zip(ax[0], panels):
        a.imshow(im); a.set_title(t); a.axis('off')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/compare.png', dpi=130, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # 선택한 설정은 전체 해상도로 저장
    for tag, fn in ((f'reinhard_B{B}', lambda L: reinhard(L, B * Lmax)),
                    (f'log_beta{beta}', lambda L: logarithmic(L, beta, Lmax))):
        out = (to_display(tone_map(hdr, Lbar, fn)) * 255 + 0.5).astype(np.uint8)
        cv2.imwrite(f'{out_dir}/{name}_{tag}.png', out[..., ::-1])
    log(f'saved full-resolution results: Reinhard B={B}, Logarithmic beta={beta}', LOG)


if __name__ == '__main__':
    for name in SOURCES:
        run(name)
