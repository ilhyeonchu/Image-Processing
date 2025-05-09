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


    ???


    for i in range(h):
        print('\r%d / %d ...' %(i, h), end="")
        for j in range(w):

            ???

    return dst

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
