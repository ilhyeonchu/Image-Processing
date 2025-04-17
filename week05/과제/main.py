import cv2
import matplotlib.pyplot as plt
from resizers.bilinear_resizer import BilinearResizer
from resizers.nearest_resizer import NearestResizer

def save_img(downsample_bilinear_img, upsample_bilinear_img, downsample_nearest_img, upsample_nearest_img):

    cv2.imwrite('downsample_bilinear_img.png', downsample_bilinear_img)
    cv2.imwrite('upsample_bilinear_img.png', upsample_bilinear_img)
    cv2.imwrite('downsample_nearest_img.png', downsample_nearest_img)
    cv2.imwrite('upsample_nearest_img.png', upsample_nearest_img)


def main():
    old_img = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    # TODO : Resize할 이미지 크기 입력
    downsample_shape = (222, 222)
    upsample_shape = (1024, 1024)

    ################### 아래 코드 수정 금지 #####################
    downsample_bilinear_resizer = BilinearResizer(old_img, downsample_shape)
    upsample_bilinear_resizer = BilinearResizer(old_img, upsample_shape)
    downsample_nearest_resizer = NearestResizer(old_img, downsample_shape)
    upsample_nearest_resizer = NearestResizer(old_img, upsample_shape)

    downsample_bilinear_img = downsample_bilinear_resizer.resize()
    upsample_bilinear_img = upsample_bilinear_resizer.resize()
    downsample_nearest_img = downsample_nearest_resizer.resize()
    upsample_nearest_img = upsample_nearest_resizer.resize()

    save_img(downsample_bilinear_img, upsample_bilinear_img, downsample_nearest_img, upsample_nearest_img)

if __name__ == '__main__':
    main()
