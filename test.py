import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_test(points:list):
    plt.plot(points)
    plt.show()

def img_show_test(img_height:int, img_width:int):
    src = np.zeros((img_height, img_width), dtype=np.uint8)
    cv2.imshow("src", src)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    plot_test([0, 1, 2, 3, 4])
    img_show_test(512, 512)
