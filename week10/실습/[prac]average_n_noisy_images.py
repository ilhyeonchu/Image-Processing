import cv2
import numpy as np

def add_gaus_noise(src, mean=0, sigma=0.1):
    # np.random.normal(mean, sigma, size=(h, w)) 사용
    dst = src / 255
    (h, w) = dst.shape

    ???

    return dst


def main():
    np.random.seed(seed=100)
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    (h, w) = src.shape
    N = 100
    N_imgs = np.zeros((N, h, w), dtype=np.float64)

    for i in range(N):
        N_imgs[i] = add_gaus_noise(src, mean=0, sigma=0.1)

    dst = ???

    cv2.imshow('original', src)
    cv2.imshow('noisy image', N_imgs[-1])
    cv2.imshow('noise removal', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return

if __name__ == '__main__':
    main()

