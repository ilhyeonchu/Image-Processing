import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_hist(src):
    hist = np.zeros((256,), np.float32)

    src_flatten = src.reshape(-1)

    for index in range(len(src_flatten)):
        value = src_flatten[index]
        hist[value] += 1

    return hist

def thresholding(src, threshold):
    dst = (src > threshold).astype(np.float32)
    dst = dst * 255

    return dst

def calc_within_class_variance(probs, threshold):
    intensity_level = probs.shape[0]

    intensity = np.arange(intensity_level)
    intensity_probs = intensity * probs

    if threshold == 0:
        left_prob = 0
        left_variance = 0
    else:
        left_prob = ???
        left_mean = ???
        left_variance = ???

    if threshold == 255:
        right_prob = 0
        right_variance = 0
    else:
        right_prob = ???
        right_mean = ???
        right_variance = ???

    within_class_variance = left_prob * left_variance + right_prob * right_variance

    return within_class_variance

def otsu_method_by_within_class_variance(src):
    hist = get_hist(src)
    probs = ???

    variances = []
    for k in range(probs.shape[0]):
        variances.append(calc_within_class_variance(probs, k))

    return np.argmin(variances)

def main():
    src = cv2.imread('../meat.png', cv2.IMREAD_GRAYSCALE)
    src_color = cv2.imread('../meat.png')

    cv2_threshold, _ = cv2.threshold(src, -1, 255, cv2.THRESH_OTSU) # Between-class variance

    threshold = otsu_method_by_within_class_variance(src)
    fat = thresholding(src, threshold)

    h, w = fat.shape
    fat_3ch = np.zeros((h, w, 3), dtype=np.uint8)
    fat_3ch[:, :, 1] = fat

    final_color = cv2.addWeighted(src_color, 1, fat_3ch, 0.5, 0)


    # Visualize threshold
    plt.hist(src.reshape(-1), bins=256)
    plt.vlines(threshold, 0, 5000, colors="red", label="Within")
    plt.text(threshold, 5150, s=f"{int(threshold)}", horizontalalignment='left', color="red")
    plt.vlines(cv2_threshold, 0, 5000, colors="blue", label="between")
    plt.text(cv2_threshold, 5150, s=f"{int(cv2_threshold)}", horizontalalignment='right', color="blue")
    plt.legend()
    plt.show()

    plt.figure(figsize=(15,5))

    plt.subplot(1, 5, 1)
    plt.imshow(src, cmap="gray")
    plt.title("Input gray")
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.imshow(src_color[...,::-1])
    plt.title("Input color")
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.imshow(fat, cmap="gray")
    plt.title("fat")
    plt.axis("off")

    plt.subplot(1, 5, 4)
    plt.imshow(fat_3ch)
    plt.title("fat_3ch")
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.imshow(final_color[...,::-1])
    plt.title("final_color")
    plt.axis("off")

    plt.show()
    return


if __name__ == '__main__':
    main()