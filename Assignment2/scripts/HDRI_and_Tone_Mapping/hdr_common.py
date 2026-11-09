import glob
import os
import re
import cv2
import numpy as np

HDRI_DIR = '/home/juneyoung-yoon/Desktop/Computational_Photography/Assignment2/HDRI'

# Part 2 권장값 (z in [0, 1])
ZMIN, ZMAX = 0.05, 0.95


DATASETS = ['image_1', 'office', 'tower']


# office_1 ~ office_6.jpg의 EXIF ExposureTime 값 (sec)
OFFICE_DTS = [1 / 30, 1 / 10, 1 / 3, 0.625, 1.3, 4.0]


def load_bracket(name='image_1'):
    # 반환: (이미지 경로 리스트, 노출 시간 배열) -- 노출 시간이 짧은(어두운) 순서
    if name == 'image_1':
        # image_1_x.JPG -> 1/x sec (README.txt)
        items = []
        for path in glob.glob(os.path.join(HDRI_DIR, 'image_1_*.JPG')):
            x = int(re.search(r'image_1_(\d+)\.JPG$', path).group(1))
            items.append((1.0 / x, path))
        items.sort()
        return [p for _, p in items], np.array([dt for dt, _ in items])
    if name == 'office':
        # EXIF의 ExposureTime (OFFICE_DTS)
        paths = [os.path.join(HDRI_DIR, f'office_{k}.jpg') for k in range(1, len(OFFICE_DTS) + 1)]
        return paths, np.array(OFFICE_DTS)
    if name == 'tower':
        # 노출 시간 정보 없음 -> 과제 가정: 2^(k-1), k = 1, 2, 3 (under < mean < over)
        paths = [os.path.join(HDRI_DIR, f'{n}530.png') for n in ['under', 'mean', 'over']]
        return paths, 2.0 ** np.arange(len(paths))
    raise ValueError(name)


def dt_label(dt):
    # 1/N 형태로 딱 떨어지면 1/N, 아니면 숫자
    inv = 1.0 / dt
    if dt < 1 and abs(inv - round(inv)) < 1e-2 * inv:
        return f'1/{round(inv)}'
    return f'{dt:.3g}'


def load_ldr(path):
    # uint8 RGB (H x W x 3)
    return cv2.imread(path, cv2.IMREAD_COLOR)[..., ::-1].copy()


def w_unif(z, zmin=ZMIN, zmax=ZMAX):
    z = np.asarray(z, dtype=np.float64)
    return np.where((z >= zmin) & (z <= zmax), 1.0, 0.0)


def w_tent(z, zmin=ZMIN, zmax=ZMAX):
    z = np.asarray(z, dtype=np.float64)
    return np.where((z >= zmin) & (z <= zmax), np.minimum(z, 1 - z), 0.0)


def w_gaussian(z, zmin=ZMIN, zmax=ZMAX):
    z = np.asarray(z, dtype=np.float64)
    return np.where((z >= zmin) & (z <= zmax), np.exp(-4 * (z - 0.5) ** 2 / 0.5 ** 2), 0.0)
