import cv2
import numpy as np
import matplotlib.pyplot as plt


def min_max_scaling(src):
    return (src - src.min()) / (src.max() - src.min())


def get_DoG_filter_by_equation(fsize, sigma):
    DoG_x = np.zeros((fsize, fsize), np.float64)
    DoG_y = np.zeros((fsize, fsize), np.float64)
    half = fsize // 2
    for y in range(-half, half + 1):
        for x in range(-half, half + 1):
            DoG_x[y + half, x + half] = -1 / (2 * np.pi * sigma**2) * x * np.exp(-(x**2 + y**2) / (2 * sigma**2))
            DoG_y[y + half, x + half] = -1 / (2 * np.pi * sigma**2) * y * np.exp(-(x**2 + y**2) / (2 * sigma**2))

    return DoG_y, DoG_x


def calculate_magnitude(grad_x, grad_y):
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return magnitude

def merge_images(images, titles):
    plt.figure(figsize=(10, 10))

    for idx, (image, title) in enumerate(zip(images, titles)):
        image = image.clip(0, 255)
        plt.subplot(2, 2, idx+1)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')

    plt.show()


if __name__ == "__main__":
    image = cv2.imread('noise_Lena.png', cv2.IMREAD_GRAYSCALE)
    image = image.astype(np.float32) / 255.

    f_size = 13
    sigma = 3

    DoG_y, DoG_x = get_DoG_filter_by_equation(f_size, sigma)

    DoG_gradient_y = cv2.filter2D(image, -1, DoG_y)
    DoG_gradient_x = cv2.filter2D(image, -1, DoG_x)
    DoG_expression_magnitude = calculate_magnitude(DoG_gradient_x, DoG_gradient_y)
    DoG_expression_magnitude = min_max_scaling(DoG_expression_magnitude)

    DoG_gradient_x = np.abs(DoG_gradient_x)
    DoG_gradient_x = min_max_scaling(DoG_gradient_x)

    DoG_gradient_y = np.abs(DoG_gradient_y)
    DoG_gradient_y = min_max_scaling(DoG_gradient_y)

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(DoG_gradient_x, cmap="gray")
    plt.title("DoG x-direction gradient")
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(DoG_gradient_y, cmap="gray")
    plt.title("DoG y-direction gradient")
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(min_max_scaling(DoG_x), cmap="gray")
    plt.title("DoG x-direction filter")
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(min_max_scaling(DoG_y), cmap="gray")
    plt.title("DoG y-direction filter")
    plt.axis('off')


    merge_images([image, DoG_gradient_x, DoG_expression_magnitude, DoG_gradient_y], ["Input image", "x-direction gradient", "Gradient magnitude", "y-direction gradient"])
