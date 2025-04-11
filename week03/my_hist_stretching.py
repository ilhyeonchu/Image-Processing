import numpy as np
import matplotlib.pyplot as plt
import cv2


def my_cal_Hist(img):
    """
    Argument info
    img: gray scale image (H x W)

    variable info
    hist : 이미지의 픽셀 값들의 빈도수를 세는 1차원 배열
    hist의 index는 이미지의 픽셀 값을 의미 i.e) pixel 값 18 == index 18

    return info
    hist: 입력 이미지의 각 픽셀 빈도수를 나타내는 배열
    """
    h, w = img.shape
    # 주어진 이미지의 가진 1차원 배열 생성
    # 256 : 0 ~ 255 범위의 픽셀
    hist = np.zeros(256, dtype=np.int32)

    # histogram bin
    for row in range(h):
        for col in range(w):
            intensity = img[row, col]
            hist[intensity] += 1

    return hist

def plot_histogram(src):

    # histogram 계산
    hist = my_cal_Hist(src)
    bin_x = np.arange(len(hist))
    plt.bar(bin_x, hist, width=0.8, color='g')
    plt.title('my_histogram')
    plt.xlabel('pixel intensity')
    plt.ylabel('pixel frequency')
    plt.show()

def stretching_function(start_value, end_value):
    pixel_value = np.arange(256)
    intensity_vector = np.zeros_like(pixel_value, dtype=np.float32)

    for i in range(len(start_value) - 1):  # 구간별 변환 수행
        mask = (pixel_value >= start_value[i]) & (pixel_value < start_value[i + 1])  # 현재 구간에 해당하는 입력값 찾기
        gradient = (end_value[i + 1] - end_value[i]) / (start_value[i + 1] - start_value[i])  # 해당 구간 기울기 구하기
        intensity_vector[mask] = gradient * (pixel_value[mask] - start_value[i]) + end_value[i]  # 구간별 intensity vector 찾기

    # 마지막 구간 변환
    intensity_vector[pixel_value >= start_value[-1]] = end_value[-1]
    # 예외처리
    intensity_vector = np.clip(intensity_vector, 0, 255).astype(np.uint8)

    plt.plot(pixel_value, intensity_vector)
    plt.title('stretching_function')
    plt.xlabel('input intensity')
    plt.ylabel('output intensity')
    plt.show()

    return intensity_vector

def main():

    src = cv2.imread('fruits.jpg', cv2.IMREAD_GRAYSCALE)

    low_contrast_img = np.round(src / 1.5).astype(np.uint8) + 40

    # low contrast 이미지 확인
    cv2.imshow('low contrast image', low_contrast_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # low contrast 이미지 histogram 확인
    plot_histogram(low_contrast_img)

    ################################################
    # TODO
    # 직접 값들 바꿔보기 - hyper parameter
    # start_value : 입력 픽셀 값 구간.
    # end_value : 매핑할 출력 값 구간.
    start_value = [0, 70, 150, 255]
    end_value = [0, 30, 200, 255]
    ##################################################

    # stretching_function
    # start_value구간을 end_value 구간으로 선형변환
    # ex) start_value = [0, 70, 150, 255], end_value = [0, 30, 200, 255]
    # 0 ~ 69 사이의 픽셀 값들을 0 ~ 29 사이의 값을 가지도록 선형변환 시키겠다.
    # 70 ~ 149 사이의 픽셀 값들을 30 ~ 199 사이의 값을 가지도록 선형변환 시키겠다.
    intensity_vector = stretching_function(start_value, end_value)
    stretched_dark_img = intensity_vector[low_contrast_img]

    # high contrast 이미지  확인
    cv2.imshow('my histogram stretching', stretched_dark_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # high contrast 이미지 저장
    cv2.imwrite('my_histogram_strecthing_imgs.png', stretched_dark_img)

    # high contrast 이미지 histogram 확인
    plot_histogram(stretched_dark_img)

    return

if __name__ == '__main__':
    main()