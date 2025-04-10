import cv2
import kernel
import padding
import matplotlib.pyplot as plt
import numpy as np
import time


def visualize_outputs(image, image_average, image_gaussian, image_sharp):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.title("Original image")
    plt.imshow(image, cmap="gray")

    plt.subplot(1, 4, 2)
    plt.title("Average kernel")
    plt.imshow(image_average, cmap="gray")

    plt.subplot(1, 4, 3)
    plt.title("Gaussian kernel")
    plt.imshow(image_gaussian, cmap="gray")

    plt.subplot(1, 4, 4)
    plt.title("Sharpening kernel")
    plt.imshow(image_sharp, cmap="gray")
    plt.show()

def filter2D(image, kernel, padding_size=None, padding_type=None):
    if padding_size is not None:
        assert padding_type is not None
        padded_image = padding_type()(image, *padding_size)
    else:
        padding_size = (0, 0, 0, 0)
        padded_image = image # 패딩이 없는 경우 padded image는 입력으로 들어온 image 그대로

    kernel_values = kernel.values
    image_height, image_width   = image.shape[:2]
    kernel_height, kernel_width = kernel_size

    ####### 윗 부분 수정 금지! #######

    filtered_image_height   = image_height    # 필터링 이후의 이미지 높이
    filtered_image_width    = image_width    # 필터링 이후의 이미지 너비
    filtered_image = np.zeros((filtered_image_height, filtered_image_width), dtype=image.dtype)
    for row in range(filtered_image_height):
        for column in range(filtered_image_width):
            # TODO: <-- 이 부분에 코드가 한 줄 이상 들어갑니다! -->
            # 커널의 크기에 맞춰서 영역 지정
            start_row = row
            start_column = column
            end_row = row + kernel_height
            end_column = column + kernel_width
            
            # 위에서 지정한 영역에 해당하는 패딩된 이미지의 값을 가져와서 필터링링
            filtered_image[row, column] = np.sum(padded_image[start_row:end_row, start_column:end_column] * kernel_values)

    return filtered_image # 필터링 된 이미지 반환


if __name__ == "__main__":
    plain_image     = cv2.imread("Lena.png",        cv2.IMREAD_GRAYSCALE)
    noised_image    = cv2.imread("Lena_noise.png",  cv2.IMREAD_GRAYSCALE)

    kernel_size     = (3,3)

    # TODO : 커널 크기에 따라 패딩 크기 계산 할것
    
    height = kernel_size[0]
    width  = kernel_size[1]
    
    # 패딩 크기 계산, 정사각형이라는 조건이 없으므로 가로 세로 따로 계산
    if(height % 2 == 0):
        top = (height -1) // 2
        bottom = height -1 - top
    else:
        top = (height -1) // 2
        bottom = height - top - 1
    
    if(width % 2 == 0):
        left = (width -1) // 2
        right = width -1 - left
    else:
        left = (width -1) // 2
        right = width - left - 1
    
    padding_size = (top, bottom, left, right)

    gaussian_sigma  = 1.0

    ######################## 아래 부분 수정 금지! ########################

    # 각 커널 객체 생성
    average_kernel      = kernel.Average(kernel_size)
    gaussian_kernel     = kernel.Gaussian(kernel_size, gaussian_sigma)
    sharpening_kernel   = kernel.Sharp(kernel_size)

    ########################  필터링 1. 노이즈 없는 이미지 + zero padding ########################
    filtering_start = time.time()
    print("[Plain image - Zero padding] Filtering start.")
    plain_image_average     = filter2D(plain_image, average_kernel, padding_size, padding_type=padding.Zero)
    plain_image_gaussian    = filter2D(plain_image, gaussian_kernel, padding_size, padding_type=padding.Zero)
    plain_image_sharp       = filter2D(plain_image, sharpening_kernel, padding_size, padding_type=padding.Zero)
    filtering_end = time.time() - filtering_start
    print(f"[Plain image - Zero padding] Filtering finished.")
    print(f"Elapsed time : {filtering_end} second")

    visualize_outputs(plain_image, plain_image_average, plain_image_gaussian, plain_image_sharp)

    print("="*20)

    ######################## 필터링 2. 노이즈 없는 이미지 + repetition padding ########################
    filtering_start = time.time()
    print("[Plain image - Repetition padding] Filtering start.")
    plain_image_average     = filter2D(plain_image, average_kernel, padding_size, padding_type=padding.Repeat)
    plain_image_gaussian    = filter2D(plain_image, gaussian_kernel, padding_size, padding_type=padding.Repeat)
    plain_image_sharp       = filter2D(plain_image, sharpening_kernel, padding_size, padding_type=padding.Repeat)
    filtering_end = time.time() - filtering_start
    print(f"[Plain image - Repetition padding] Filtering finished.")
    print(f"Elapsed time : {filtering_end} second")
    visualize_outputs(plain_image, plain_image_average, plain_image_gaussian, plain_image_sharp)

    print("=" * 20)

    ######################## 필터링 3. 노이즈 있는 이미지 + zero padding ########################
    filtering_start = time.time()
    print("[Noised image - Zero padding] Filtering start.")
    noised_image_average    = filter2D(noised_image, average_kernel, padding_size, padding_type=padding.Zero)
    noised_image_gaussian   = filter2D(noised_image, gaussian_kernel, padding_size, padding_type=padding.Zero)
    noised_image_sharp      = filter2D(noised_image, sharpening_kernel, padding_size, padding_type=padding.Zero)
    filtering_end = time.time() - filtering_start
    print(f"[Noised image - Zero padding] Filtering finished.")
    print(f"Elapsed time : {filtering_end} second")
    visualize_outputs(noised_image, noised_image_average, noised_image_gaussian, noised_image_sharp)

    print("=" * 20)

    ######################## 필터링 4. 노이즈 있는 이미지 + repetition padding ########################
    filtering_start = time.time()
    print("[Noised image - Repetition padding] Filtering start.")
    noised_image_average    = filter2D(noised_image, average_kernel, padding_size, padding_type=padding.Repeat)
    noised_image_gaussian   = filter2D(noised_image, gaussian_kernel, padding_size, padding_type=padding.Repeat)
    noised_image_sharp      = filter2D(noised_image, sharpening_kernel, padding_size, padding_type=padding.Repeat)
    filtering_end = time.time() - filtering_start
    print(f"[Noised image - Repetition padding] Filtering finish.")
    print(f"Elapsed time : {filtering_end} second")
    visualize_outputs(noised_image, noised_image_average, noised_image_gaussian, noised_image_sharp)

    print("=" * 20)
