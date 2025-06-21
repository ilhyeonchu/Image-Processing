import cv2
import numpy as np
import matplotlib.pyplot as plt

def backward_fit(src, matrix):

    h, w  = src.shape
    src = src.astype(np.float32)
    # matrix 역행렬 구하기
    M_inv = np.linalg.inv(matrix)

    ???

    dst = np.clip(np.round(dst), 0 , 255).astype(np.uint8)

    return dst


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    # Rotation 20 -> shearing -> scaling
    ???

    # Shearing -> rotation 20 -> scaling
    ???

    final1 = backward_fit(???)
    final2 = backward_fit(???)

    cv2.imshow('lena_gray', src)
    cv2.imshow('final1', final1)
    cv2.imshow('final2', final2)
    # cv2.imwrite('final1.png', final1)

    cv2.waitKey()
    cv2.destroyAllWindows()
