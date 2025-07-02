import numpy as np
import cv2
import time

def my_padding(src, pad_shape, pad_type='zero'):
    # zero padding인 경우
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h + 2 * p_h, w + 2 * p_w))
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


def my_filtering(src, mask, pad_type='zero'):
    (h, w) = src.shape
    (m_h, m_w) = mask.shape
    pad_img = my_padding(src, (m_h // 2, m_w // 2), pad_type)

    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(pad_img[row:m_h + row, col:m_w + col] * mask)
    return dst

def my_median_filtering(src, fsize):
    # np.median() 사용 가능
    h, w = src.shape
    dst = np.zeros((h, w))

    for row in range(h):
        for col in range(w):

            r_start = np.clip(row - fsize // 2, 0, h - 1)
            r_end = np.clip(row + fsize // 2, 0, h - 1)

            c_start = np.clip(col - fsize // 2, 0, w - 1)
            c_end = np.clip(col + fsize // 2, 0, w - 1)
            filter = src[r_start:r_end + 1, c_start:c_end + 1]

            dst[row, col] = np.median(filter)

    return dst.astype(np.uint8)

def add_snp_noise(src, prob):

    h, w = src.shape

    # np.random.rand = 0 ~ 1 사이의 값이 나옴
    noise_prob = np.random.rand(h, w)
    dst = np.zeros((h, w), dtype=np.uint8)

    for row in range(h):
        for col in range(w):
            if noise_prob[row, col] < prob:
                # pepper noise
                dst[row, col] = 0
            elif noise_prob[row, col] > 1 - prob:
                # salt noise
                dst[row, col] = 255
            else:
                dst[row, col] = src[row, col]

    return dst


def main():

    np.random.seed(seed=100)
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    # 원본 이미지에 노이즈를 추가
    src_snp_noise = add_snp_noise(src, prob=0.05)

    # nxn average filter
    filter_size = 5
    avg_filter = np.ones((filter_size, filter_size)) / (filter_size ** 2)

    # 평균 필터 적용
    average_start_time = time.time()
    dst_avg = my_filtering(src_snp_noise, avg_filter)
    dst_avg = dst_avg.astype(np.uint8)
    print('average filtering time : ', time.time() - average_start_time)

    # median filter 적용
    median_start_time = time.time()
    dst_median = my_median_filtering(src_snp_noise, filter_size)
    print('median filtering time : ', time.time() - median_start_time)

    cv2.imshow('original', src)
    cv2.imshow('Salt and pepper noise', src_snp_noise)
    cv2.imshow('noise removal(average fileter)', dst_avg)
    cv2.imshow('noise removal(median filter)', dst_median)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
