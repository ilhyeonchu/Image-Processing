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

    filtered_image_height = ???
    filtered_image_width = ???
    filtered_image = np.zeros((filtered_image_height, filtered_image_width), dtype=image.dtype)
    for row in range(filtered_image_height):
        for column in range(filtered_image_width):
            ???

    return filtered_image

def get_magnitude(grad_x, grad_y):
    return np.sqrt(grad_x**2 + grad_y**2)

def min_max_scaling(image):
    return (image - image.min()) / (image.max() - image.min())

def create_dog_filter(filter_size, sigma):
    gaussian_filter = filter.Gaussian(filter_size, sigma).values

    x_dir_derivative_filter = filter.Derivative(filter.Direction.X).values
    y_dir_derivative_filer = filter.Derivative(filter.Direction.Y).values

    dog_x = ???
    dog_y = ???

    return dog_x, dog_y

def create_log_filter(filter_size, sigma):
    gaussian_filter = filter.Gaussian(filter_size, sigma).values

    x_dir_laplace_filter = filter.Laplace(filter.Direction.X).values
    y_dir_laplace_filer = filter.Laplace(filter.Direction.Y).values

    log_x = ???
    log_y = ???

    return log_x, log_y

def find_zero_crossing(log_output):
    zero_padding = padding.Zero()
    log_output_padded = zero_padding(log_output, 1, 1, 1, 1)

    zero_crossing_plane = np.zeros_like(log_output)
    for i in range(log_output.shape[0]):
        for j in range(log_output.shape[1]):
            ???

    return zero_crossing_plane

if __name__ == "__main__":
    noised_image = cv2.imread("./noise_Lena.png",  cv2.IMREAD_GRAYSCALE)
    noised_image = noised_image / 255.

    ################################# Derivative of gaussian #################################
    dog_filter_size = (13,13)
    dog_filter_sigma = 3
    dog_filtering_padding_size = ???

    dog_x, dog_y = create_dog_filter(dog_filter_size, dog_filter_sigma)

    visualize_dog_filters(dog_x, dog_y)

    dog_x_output = filter2D(noised_image, dog_x, dog_filtering_padding_size, padding.Repeat)
    dog_y_output = filter2D(noised_image, dog_y, dog_filtering_padding_size, padding.Repeat)

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
    log_filtering_padding_size = ???
    log_x, log_y = create_log_filter(log_filter_size, log_filter_sigma)

    visualize_log_filters(log_x, log_y)

    log_filter = log_x + log_y

    log_output = filter2D(noised_image, log_filter, log_filtering_padding_size, padding.Repeat)

    visualize_log_output(noised_image, log_output)

    ################################# Find zero crossing #################################
    log_filter_size = ???
    log_filter_sigma = ???
    log_filtering_padding_size = ???
    log_x, log_y = create_log_filter(log_filter_size, log_filter_sigma)

    log_filter = log_x + log_y

    log_output = filter2D(noised_image, log_filter, log_filtering_padding_size, padding.Repeat)
    zero_crossing = find_zero_crossing(log_output)

    visualize_zero_crossing_output(noised_image, log_output, zero_crossing)

    ################################# Binarization #################################
    threshold = ???
    binarization_output = ???
    visualize_binarization_output(noised_image, gradient_magnitude, binarization_output)