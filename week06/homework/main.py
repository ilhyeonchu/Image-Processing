import cv2
import filter
import padding
import matplotlib.pyplot as plt
import numpy as np
from utils.visualize import *

def filter2D(image, filter_values, padding_size=None, padding_type=None):
    if padding_size is not None:
        assert padding_type is not None
        padded_image = padding_type()(image, *padding_size)
    else:
        padding_size = (0, 0, 0, 0)
        padded_image = image

    image_height, image_width = image.shape[:2]
    filter_height, filter_width = filter_values.shape[:2]

    filtered_image_height = image_height
    filtered_image_width = image_width
    filtered_image = np.zeros((filtered_image_height, filtered_image_width), dtype=image.dtype)
    for row in range(filtered_image_height):
        for column in range(filtered_image_width):
            start_row = row
            start_column = column
            end_row = row + filter_height
            end_column = column + filter_width

            region = padded_image[start_row:end_row, start_column:end_column]
            filtered_image[row, column] = np.sum(region * filter_values)

    return filtered_image

def get_magnitude(grad_x, grad_y):
    return np.sqrt(grad_x**2 + grad_y**2)

def min_max_scaling(image):
    return (image - image.min()) / (image.max() - image.min())

def create_dog_filter(filter_size, sigma):
    gaussian_filter = filter.Gaussian(filter_size, sigma).values

    x_dir_derivative_filter = filter.Derivative(filter.Direction.X).values
    y_dir_derivative_filer = filter.Derivative(filter.Direction.Y).values

    x_dir_derivative_filter = x_dir_derivative_filter.reshape(1,-1)
    y_dir_derivative_filer = y_dir_derivative_filer.reshape(-1,1)

    dog_x = np.outer(gaussian_filter[:, 0], x_dir_derivative_filter[0])
    dog_y = np.outer(y_dir_derivative_filer[:, 0], gaussian_filter[0])

    return dog_x, dog_y

def create_log_filter(filter_size, sigma):
    gaussian_filter = filter.Gaussian(filter_size, sigma).values

    x_dir_laplace_filter = filter.Laplace(filter.Direction.X).values
    y_dir_laplace_filer = filter.Laplace(filter.Direction.Y).values

    log_x = x_dir_laplace_filter
    log_y = y_dir_laplace_filer

    return log_x, log_y

def find_zero_crossing(log_output):
    zero_padding = padding.Zero()
    log_output_padded = zero_padding(log_output, 1, 1, 1, 1)

    zero_crossing_plane = np.zeros_like(log_output)
    for i in range(log_output.shape[0]):
        for j in range(log_output.shape[1]):
            # 4개의 픽셀 평균 p1-p4까지로 구해서 저장
            p1 = (log_output_padded[i, j] + log_output_padded[i, j+1] + log_output_padded[i+1, j] + log_output_padded[i+1, j+1]) / 4
            p2 = (log_output_padded[i,j+1] + log_output_padded[i,j+2] + log_output_padded[i+1,j+1] + log_output_padded[i+1, j+2]) / 4
            p3 = (log_output_padded[i+1, j] + log_output_padded[i+1, j+1] + log_output_padded[i+2, j] + log_output_padded[i+2, j+1]) / 4
            p4 = (log_output_padded[i+1, j+1] + log_output_padded[i+1, j+2] + log_output_padded[i+2, j+1] + log_output_padded[i+2, j+2]) / 4

            # 최소 최대가 각각 양,음 수인지 체크
            values = [p1, p2, p3, p4]

            if max(values) > 0 and min(values) < 0:
                zero_crossing_plane[i, j] = 1


    return zero_crossing_plane

if __name__ == "__main__":
    noised_image = cv2.imread("./noise_Lena.png",  cv2.IMREAD_GRAYSCALE)
    noised_image = noised_image / 255.

    ################################# Derivative of gaussian #################################
    dog_filter_size = (13,13)
    dog_filter_sigma = 3
    dog_filtering_padding_size = [6,6,6,6] # [13,13]이 필터 크기이므로 모서리의 상하좌우에 6씩 추가해야

    dog_x, dog_y = create_dog_filter(dog_filter_size, dog_filter_sigma)

    visualize_dog_filters(dog_x, dog_y)

    dog_x_output = filter2D(noised_image, dog_x, dog_filtering_padding_size, padding.Zero)
    dog_y_output = filter2D(noised_image, dog_y, dog_filtering_padding_size, padding.Zero)

    gradient_magnitude = get_magnitude(dog_x_output, dog_y_output)
    gradient_magnitude = min_max_scaling(gradient_magnitude)

    dog_x_output = np.abs(dog_x_output)
    dog_x_output = min_max_scaling(dog_x_output)

    dog_y_output = np.abs(dog_y_output)
    dog_y_output = min_max_scaling(dog_y_output)

    visualize_gradient_outputs(noised_image, dog_x_output, dog_y_output, gradient_magnitude)

    ################################# Laplacian of gaussian #################################
    log_filter_size = (9, 9)
    log_filter_sigma = 1.4**0.5
    log_filtering_padding_size = [4,4,4,4] # [9,9]이 필터 크기이므로 모서리의 상하좌우에 4씩 추가해야
    log_x, log_y = create_log_filter(log_filter_size, log_filter_sigma)

    visualize_log_filters(log_x, log_y)

    log_filter = log_x + log_y

    log_output = filter2D(noised_image, log_filter, log_filtering_padding_size, padding.Zero)

    visualize_log_output(noised_image, log_output)

    ################################# Find zero crossing #################################
    log_filter_size = [15, 15]  # 필터 사이즈 15x15로 주어짐
    log_filter_sigma = 2     # 1-4 사이의 값
    log_filtering_padding_size = [7, 7, 7, 7]   # [15,15]이 필터 크기이므로 모서리의 상하좌우에 7씩 추가해야
    log_x, log_y = create_log_filter(log_filter_size, log_filter_sigma)

    log_filter = log_x + log_y

    log_output = filter2D(noised_image, log_filter, log_filtering_padding_size, padding.Zero)
    zero_crossing = find_zero_crossing(log_output)

    visualize_zero_crossing_output(noised_image, log_output, zero_crossing)

    ################################# Binarization #################################
    threshold = 0.2 # 0-0.5 사이의 값 단위는 0.1
    binarization_output = (gradient_magnitude > threshold).astype(np.uint8) # 기준값(threshold)보다 크면 1, 작으면 0으로 변환
    visualize_binarization_output(noised_image, gradient_magnitude, binarization_output)