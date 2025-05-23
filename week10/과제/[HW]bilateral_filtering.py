import numpy as np
import cv2
import time


def convert_uint8(src):
    return np.round((((src - src.min()) / (src.max() - src.min())) * 255)).astype(np.uint8)

def my_padding(src, pad_shape, pad_type='zero'):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h + 2 * p_h, w + 2 * p_w), dtype=np.float32)
    pad_img[p_h:p_h + h, p_w:p_w + w] = src

    if pad_type == 'repetition':
        print('repetition padding')
        # up
        pad_img[:p_h, p_w:p_w + w] = src[0, :]
        # down
        pad_img[p_h + h:, p_w:p_w + w] = src[h - 1, :]
        # left
        pad_img[:, :p_w] = pad_img[:, p_w:p_w + 1]
        # right
        pad_img[:, p_w + w:] = pad_img[:, p_w + w - 1: p_w + w]
    return pad_img

def my_normalize(src):
    dst = src.copy()
    dst *= 255
    dst = np.clip(dst, 0, 255)
    return dst.astype(np.uint8)

def add_gaus_noise(src, mean=0, sigma=0.1):
    dst = src / 255
    h, w = dst.shape
    noise = np.random.normal(mean, sigma, size=(h, w))
    dst += noise
    return dst

def my_bilateral_with_patch(src, fsize, bsize, sigma_xy, sigma_r, pad_type='zero'):
    (h, w) = src.shape

    pad_f = fsize // 2
    pad_b = bsize // 2
    pad_total = pad_f + pad_b  # 유사 패치를 찾을 범위
    src_pad = my_padding(src, (pad_total, pad_total), pad_type) # 패딩

    # 계속 사용할 가우시안 커널 미리 계산, 좌우 대칭으로 0,0이 중심이 되도록 -pad부터 +pad까지로 인덱스 설정
    yy, xx = np.mgrid[-pad_f:pad_f + 1, -pad_f:pad_f + 1]
    gaussian_kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * (sigma_xy ** 2))).astype(np.float32)

    dst = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        print('\r%d / %d ...' % (i + 1, h), end="")
        for j in range(w):
            # 패딩된 이미지에서 중심 설정 -> 처음 시작이 0,0이 아니라 이미 pad_total에서 시작해야 원본의 0,0과 일치
            ci = i + pad_total
            cj = j + pad_total

            # 내 패치 설정
            my_patch = src_pad[ci - pad_f:ci + pad_f + 1,
                                cj - pad_f:cj + pad_f + 1]

            # 유사 패치 저장할 변수
            best_ssd = np.inf
            sim_patch = None
            sim_center_val = None

            # 유사 패치 찾기 시작
            for si in range(ci - pad_b, ci + pad_b + 1):
                for sj in range(cj - pad_b, cj + pad_b + 1):
                    if si == ci and sj == cj:       # 본인 패치는 스킵
                        continue
                    cand_patch = src_pad[si - pad_f:si + pad_f + 1,
                                         sj - pad_f:sj + pad_f + 1]
                    ssd = np.sum((my_patch - cand_patch) ** 2)
                    if ssd < best_ssd:
                        best_ssd = ssd
                        sim_patch = cand_patch
                        sim_center_val = src_pad[si, sj]

            # if no similar patch was found (edge cases), fall back to the reference patch
            if sim_patch is None:
                sim_patch = my_patch
                sim_center_val = src_pad[ci, cj]

            # 3) bilateral filtering on the reference patch
            ref_center_val = src_pad[ci, cj]
            range_kernel_ref = np.exp(-((my_patch - ref_center_val) ** 2) /
                                      (2 * (sigma_r ** 2))).astype(np.float32)
            weight_ref = gaussian_kernel * range_kernel_ref
            res_ref = np.sum(weight_ref * my_patch) / np.sum(weight_ref)

            # 4) bilateral filtering on the similar patch
            range_kernel_sim = np.exp(-((sim_patch - sim_center_val) ** 2) /
                                      (2 * (sigma_r ** 2))).astype(np.float32)
            weight_sim = gaussian_kernel * range_kernel_sim
            res_sim = np.sum(weight_sim * sim_patch) / np.sum(weight_sim)

            # 5) combine the two results (equal weights as 기본 설정)
            dst[i, j] = 0.5 * res_ref + 0.5 * res_sim

    print()  # move to the next line after the progress bar
    return my_normalize(np.clip(dst, 0.0, 1.0))

if __name__ == '__main__':

    src = cv2.imread('baby.png', cv2.IMREAD_GRAYSCALE)
    np.random.seed(seed=100)

    src_noise = add_gaus_noise(src, mean=0, sigma=0.1)
    start = time.time()
    dst = my_bilateral_with_patch(src_noise, 11, 17, sigma_xy=5, sigma_r=0.2)
    print('\n', time.time() - start)
    cv2.imshow('src', src)
    cv2.imshow('src_noise', convert_uint8(src_noise))
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()
