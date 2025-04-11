import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_images(images, titles):
    plt.figure(figsize=(10, 10))
    for idx, (image, title) in enumerate(zip(images, titles)):
        image = image.clip(0, 255)
        plt.subplot(2,2,idx+1)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')

    plt.show()

if __name__ == "__main__":
    clean_image = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    noised_image = cv2.imread("noise_Lena.png", cv2.IMREAD_GRAYSCALE)

    clean_image = clean_image / 255.
    noised_image = noised_image / 255.

    laplace_filter = ???

    filtered_clean = cv2.filter2D(clean_image, -1, laplace_filter)
    filtered_noise = cv2.filter2D(noised_image, -1, laplace_filter)

    plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    plt.imshow(clean_image, cmap="gray")
    plt.title("[Input] Clean")
    plt.axis("off")

    plt.subplot(2,2,2)
    plt.imshow(filtered_clean, cmap="gray")
    plt.title("[Output] Clean")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(noised_image, cmap="gray")
    plt.title("[Input] Noise")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(filtered_noise, cmap="gray")
    plt.title("[Output] Noise")
    plt.axis("off")
    plt.show()
