import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_sobel_filter():
    blurring = np.array([[1], [2], [1]])
    derivative_filter = np.array([[-1], [0], [1]])
    sobel_x = np.dot(blurring, derivative_filter.T)
    sobel_y = np.dot(derivative_filter, blurring.T)
    return sobel_x, sobel_y


def min_max_scaling(src):
    return (src - src.min()) / (src.max() - src.min())


def calculate_magnitude(sobel_x, sobel_y):
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return magnitude


def show_images(images, titles):
    plt.figure(figsize=(10, 10))
    for idx, (image, title) in enumerate(zip(images, titles)):
        image = image.clip(0, 255)
        plt.subplot(2,2,idx+1)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')

    plt.show()

def show_sobel_filtering_output(image):
    sobel_filter_x, sobel_filter_y = generate_sobel_filter()
    gradient_x = cv2.filter2D(image, -1, sobel_filter_x)
    gradient_y = cv2.filter2D(image, -1, sobel_filter_y)

    magnitude = calculate_magnitude(gradient_x, gradient_y)
    magnitude = min_max_scaling(magnitude)

    gradient_x = np.abs(gradient_x)
    gradient_x = min_max_scaling(gradient_x)

    gradient_y = np.abs(gradient_y)
    gradient_y = min_max_scaling(gradient_y)

    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.title("Input image")

    plt.subplot(2, 2, 2)
    plt.imshow(gradient_x, cmap="gray")
    plt.axis("off")
    plt.title("x-direction gradient")

    plt.subplot(2, 2, 3)
    plt.imshow(magnitude, cmap="gray")
    plt.axis("off")
    plt.title("Gradient magnitude")

    plt.subplot(2, 2, 4)
    plt.imshow(gradient_y, cmap="gray")
    plt.axis("off")
    plt.title("y-direction gradient")
    plt.show()


if __name__ == "__main__":
    clean_image = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    clean_image = clean_image / 255.

    noise_image = cv2.imread('noise_Lena.png', cv2.IMREAD_GRAYSCALE)
    noise_image = noise_image / 255.

    show_sobel_filtering_output(clean_image)
    show_sobel_filtering_output(noise_image)
