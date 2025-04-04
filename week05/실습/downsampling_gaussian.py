import cv2
import numpy as np

def my_gaussian_downsampling(old_img, block_size, gaussian_filter=None):
    # old_img의 height와 width
    h_old, w_old = old_img.shape

    # block 크기
    block_y, block_x = block_size

    # new_img의 height와 width
    h_new = h_old // block_y
    w_new = w_old // block_x

    # 빈 new_img 선언
    new_img = np.zeros((h_new, w_new), np.uint8)

    # new_img 픽셀 값 채우기
    for row in range(h_new):
        for col in range(w_new):
            # 이미지에서 block 영역 가져오기
            block = old_img[row * block_y:(row + 1) * block_y, col * block_x:(col + 1) * block_x]
            # block에 가우시안 필터 적용
            filtered_patch = block * gaussian_filter
            # 픽셀 값 구하기
            intensity = np.round(np.sum(filtered_patch))
            new_img[row, col] = intensity

    return new_img

def main():
    old_img = cv2.imread('Dog.jpg', cv2.IMREAD_GRAYSCALE)
    block_size = (7, 7)

    gaussian_1d_filter = cv2.getGaussianKernel(7, 3)
    gaussian_filter = gaussian_1d_filter @ gaussian_1d_filter.T

    new_img = my_gaussian_downsampling(old_img, block_size=block_size, gaussian_filter=gaussian_filter)

    cv2.imwrite('downsampled_smoothing.png', new_img)

if __name__ == '__main__':
    main()