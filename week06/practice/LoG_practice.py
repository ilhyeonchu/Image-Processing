import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_log_filter(fsize, sigma):
    half = fsize//2
    idx = np.arange(-half, half+1)
    xs = np.tile(idx, (fsize, 1))
    ys = xs.T

    r_squared = xs ** 2 + ys ** 2
    log_filter = -1 / (np.pi * sigma ** 4) * (1 - r_squared / (2 * sigma ** 2)) * np.exp(-r_squared / (2 * sigma ** 2))

    return log_filter

if __name__ == "__main__":
    image = cv2.imread('noise_Lena.png', cv2.IMREAD_GRAYSCALE)

    image = image / 255.

    laplace_filter = np.full((9,9), 1, dtype=np.float32)
    laplace_filter[4,4] = -80

    log_filter = get_log_filter(9, 1.4**0.5)

    plt.imshow(log_filter, cmap="gray", interpolation="bicubic")
    plt.show()

    lap_image = cv2.filter2D(image, -1, laplace_filter)
    log_image = cv2.filter2D(image, -1, log_filter)

    plt.figure(figsize=(10, 5))
    plt.subplot(1,3,1)
    plt.imshow(image[105:155,100:150], cmap="gray")
    plt.axis("off")
    plt.title("Input image")

    plt.subplot(1, 3, 2)
    plt.imshow(lap_image[105:155,100:150], cmap="gray")
    plt.axis("off")
    plt.title("Laplace filter output")

    plt.subplot(1, 3, 3)
    plt.imshow(log_image[105:155,100:150], cmap="gray")
    plt.axis("off")
    plt.title("LoG filter output")
    plt.show()
