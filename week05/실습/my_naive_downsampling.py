import cv2
import numpy as np

def my_naive_downsampling(old_img, block_size):
    # old_img의 height와 width
    h_old, w_old = old_img.shape

    # block 크기
    block_y, block_x = block_size

    # new_img의 height와 width
    h_new = h_old // block_y
    w_new = w_old // block_x

    # block의 중앙
    y_median = block_y // 2
    x_median = block_x // 2

    # 빈 new_img 선언
    new_img = np.zeros((h_new, w_new), np.uint8)

    # new_img 픽셀 값 채우기
    for row in range(h_new):
        for col in range(w_new):
            # 이미지에서 block 영역 가져오기
            block = old_img[(row * block_y):(row * block_y + block_y), (col * block_x):(col * block_x + block_x)]
            # block 중앙의 픽셀 값 가져오기
            intensity = block[y_median, x_median]
            new_img[row, col] = intensity

    return new_img

def main():
    old_img = cv2.imread('Dog.jpg', cv2.IMREAD_GRAYSCALE)
    block_size = (7, 7)
    new_img = my_naive_downsampling(old_img, block_size=block_size)
    cv2.imwrite('downsampled_img.png', new_img)

if __name__ == '__main__':
    main()