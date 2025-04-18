import numpy as np
import cv2
import matplotlib.pyplot as plt

def my_calcHist(src):

    # 실습 코드들 참고, 이미지의 명암에 대한 히스토그램을 계산
    h, w = src.shape
    hist = np.zeros(256, dtype=np.int32)
    for row in range(h):
        for col in range(w):
            intensity = src[row, col]
            hist[intensity] += 1

    return hist

    ###############################
    # TODO                        #
    # my_calcHist완성              #
    # src : input image           #
    # hist : src의 히스토그램       #
    ###############################



def my_normalize_hist(hist, pixel_num):

    # 그냥 위에서 구한 histogram을 pixel_num으로 나누면 됨
    normalized_hist = hist / pixel_num
    return normalized_hist

    ########################################################
    # TODO                                                 #
    # my_normalize_hist완성                                 #
    # hist : 히스토그램                                      #
    # pixel_num : image의 전체 픽셀 수                       #
    # normalized_hist : 히스토그램값을 총 픽셀수로 나눔         #
    ########################################################




def my_PDF2CDF(pdf):

    # cdf안에 차례대로 누적 비율?을 넣어주면 됨
    cdf = np.zeros(pdf.shape)
    cdf[0] = pdf[0]
    for i in range(1, len(pdf)):
        cdf[i] = cdf[i - 1] + pdf[i]
    return cdf

    ########################################################
    # TODO                                                 #
    # my_PDF2CDF완성                                        #
    # np.cumsum 사용시 감점                                  #
    # pdf : normalized_hist                                #
    # cdf : pdf의 누적                                      #
    ########################################################



def my_denormalize(normalized, gray_level):

    # normalized 값에 저장된 cdf를 255(gray_level) 곱해주기
    denormalized = normalized * gray_level
    return denormalized

    ########################################################
    # TODO                                                 #
    # my_denormalize완성                                   #
    # normalized : 누적된pdf값(cdf)                        #
    # gray_level : max_gray_level                          #
    # denormalized : normalized와 gray_level을 곱함        #
    ########################################################


def my_intensity_vector(denormalized_output):

    # 2주차 수업 자료에 나와있는 np.floor()를 사용해 버림으로 정수값으로 변환
    intensity_vector = np.floor(denormalized_output).astype(np.uint8)  # astype 안하면 다음에 타입 문제 발생
    return intensity_vector

    ###################################################################
    # TODO                                                            #
    # my_intensity_vector완성                                         #
    # denormalized_output : normalized와 gray_level을 곱한 값             #
    # intensity_vector : denormalized_output 버림을 하여 정수값으로 변환    #
    ####################################################################


def my_calcHist_equalization(intensity_vector, hist):

    # hist[i] 는 원래 i라는 값을 가지고 있는 픽셀 수
    # 이 값을 변환된 픽셀 값인 intensity_vector[i]의 위치에 추가시켜줌
    hist_equal = np.zeros(256, dtype=np.int32)
    for i in range(256):
        hist_equal[intensity_vector[i]] += hist[i]

    return hist_equal

    ###################################################################
    # TODO                                                            #
    # my_calcHist_equalization완성                                    #
    # intensity_vector : intensity transformation vector             #
    # hist : 히스토그램                                                #
    # hist_equal : equalization된 히스토그램                           #
    ####################################################################



def my_equal_img(src, intensity_vector):

    (h, w) = src.shape
    src_flat = np.reshape(src, (h * w)) # 1차원으로 변환
    one_hot_src = np.eye(256)[src_flat] # 원핫 인코딩
    dst = np.dot(one_hot_src, intensity_vector).reshape(h, w)   # 원핫 인코딩된 src에 intensity_vector를 곱해줌
    dst = dst.astype(np.uint8)  # 타입 변환

    return dst
    ###################################################################
    # TODO                                                            #
    # my_equal_img완성                                                 #
    # Reshape시 np.reshape 내장 함수 사용 가능                            #
    # src : input image                                               #
    # intensity_vector : intensity transformation vector              #
    # one_hot_src : one-hot encoding된 src                             #
    # dst : equalization된 결과 이미지                                   #
    ####################################################################





def my_hist_equal(src, type='original'):

    """
    1. histogram을 구한다 (my_calHist)
    2. histogram을 총 픽셀수로 나눈다(my_normalize_hist)
    3. 누적 합을 구한다 (my_PDF2CDF)
    4. 누적 시킨 값에 gray level의 최댓값(255)을 곱한다. (my_denormalize)
    5. 최댓값이 곱해진 값들을 정수화(버림)하여 intensity transformation vector를 구한다. -> (my_intensity_vector)
    6. Intensity transformation vector를 사용하여 histogram equalization 결과를 반환한다.  (my_calcHist_equalization)
    """

    (h, w) = src.shape
    max_gray_level = 255
    histogram = my_calcHist(src)
    normalized_histogram = my_normalize_hist(histogram, h * w)
    normalized_output = my_PDF2CDF(normalized_histogram)
    denormalized_output = my_denormalize(normalized_output, max_gray_level)
    intensity_vector = my_intensity_vector(denormalized_output)
    hist_equal = my_calcHist_equalization(intensity_vector, histogram)

    # show intensity function
    ###################################################################
    # TODO                                                            #
    # plt.plot(???,???)완성                                           #
    # plt.plot(x축, y축)
    # x축 : 0 ~ 255 범위의 정수
    # y축 : 0 ~ 255 각각의 정수 값에 해당하는 equalization 변환 픽셀 값
    ###################################################################

    #
    plt.plot(np.arange(256), intensity_vector)
    plt.title(type + ' intensity function')
    plt.xlabel('input intensity')
    plt.ylabel('output intensity')
    plt.show()

    ### dst : equalization 결과 image
    dst = my_equal_img(src, intensity_vector)

    return dst, hist_equal

def plot_equal_histogram(dst, hist_equal,type='original'):

    plt.figure(figsize=(8, 5))
    binX = np.arange(len(hist_equal))
    plt.title(type +' histogram equalization')
    plt.bar(binX, hist_equal, width=0.5, color='g')
    plt.show()

def plot_before_equal_histogram(src, original_hist,type='original'):
    plt.figure(figsize=(8, 5))
    binX = np.arange(len(original_hist))
    plt.title(type)
    plt.bar(binX, original_hist, width=0.5, color='g')
    plt.show()


if __name__ == '__main__':

    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    ###################################################################
    # TODO
    # src_add, src_sub, src_div, src_mul 완성
    # cv2.add, cv2.subtract cv2.divide cv2.multiply 사용 시 감점
    # src_add : src + 64
    # src_sub : src - 64
    # src_div : src / 3
    # src_mul : src * 2
    ###################################################################

    src_add = src.astype(np.float32) + 64
    src_add[src_add > 255] = 255
    src_add[src_add < 0] = 0
    src_add = np.round(src_add).astype(np.uint8)

    src_sub = src.astype(np.float32) - 64
    src_sub[src_sub > 255] = 255
    src_sub[src_sub < 0] = 0
    src_sub = np.round(src_sub).astype(np.uint8)

    src_div = src.astype(np.float32) / 3
    src_div[src_div > 255] = 255
    src_div[src_div < 0] = 0
    src_div = np.round(src_div).astype(np.uint8)

    src_mul = src.astype(np.float32) * 2
    src_mul[src_mul > 255] = 255
    src_mul[src_mul < 0] = 0
    src_mul = np.round(src_mul).astype(np.uint8)

    # original
    src_hist = my_calcHist(src)
    dst, hist_src_equal = my_hist_equal(src)

    # src_add
    src_add_hist = my_calcHist(src_add)
    dst_add, hist_add_equal = my_hist_equal(src_add,type='src_add')

    # src_sub
    src_sub_hist = my_calcHist(src_sub)
    dst_sub, hist_sub_equal = my_hist_equal(src_sub,type='src_sub')

    # src_div
    src_div_hist = my_calcHist(src_div)
    dst_div, hist_div_equal = my_hist_equal(src_div,type='src_div')

    # src_mul
    src_mul_hist = my_calcHist(src_mul)
    dst_mul, hist_mul_equal = my_hist_equal(src_mul, type='src_mul')

    cv2.imshow('original',src)
    cv2.imshow('original equal result', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite('original.png', src)
    cv2.imwrite('original_equal_result.png', dst)

    cv2.imshow('src_add', src_add)
    cv2.imshow('src_add equal result', dst_add)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite('src_add.png', src_add)
    cv2.imwrite('src_add_equal_result.png', dst_add)

    cv2.imshow('src_sub', src_sub)
    cv2.imshow('src_sub equal result', dst_sub)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite('src_sub.png', src_sub)
    cv2.imwrite('src_sub_equal_result.png', dst_sub)

    cv2.imshow('src_div', src_div)
    cv2.imshow('src_div equal result', dst_div)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite('src_div.png', src_div)
    cv2.imwrite('src_div_equal_result.png', dst_div)

    cv2.imshow('src_mul', src_mul)
    cv2.imshow('src_mul equal result', dst_mul)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite('src_mul.png', src_mul)
    cv2.imwrite('src_mul_equal_result.png', dst_mul)

    plot_before_equal_histogram(src, src_hist, type='original')
    plot_equal_histogram(dst, hist_src_equal, type='original')

    plot_before_equal_histogram(src_add, src_add_hist, type='src_add')
    plot_equal_histogram(dst_add, hist_add_equal, type='src_add')

    plot_before_equal_histogram(src_sub, src_sub_hist, type='src_sub')
    plot_equal_histogram(dst_sub, hist_sub_equal, type='src_sub')

    plot_before_equal_histogram(src_div, src_div_hist, type='src_div')
    plot_equal_histogram(dst_div, hist_div_equal, type='src_div')

    plot_before_equal_histogram(src_mul, src_mul_hist, type='src_mul')
    plot_equal_histogram(dst_mul, hist_mul_equal, type='src_mul')

