import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_out_list(output_list=[], titles=[], figsize=(10, 10)):
  plt.rcParams['figure.figsize'] = figsize
  row = 1
  col = len(output_list)

  for i in range(len(output_list)):
    image_index = i + 1
    plt.subplot(row, col, image_index)
    plt.imshow(output_list[i], cmap='gray', vmin=0, vmax=255)
    plt.title(titles[i])
  plt.show()

def backward_fit(src, matrix):

    h, w  = src.shape
    src = src.astype(np.float32)
    M_inv = np.linalg.inv(matrix)

    ???

    dst = np.clip(np.round(dst), 0, 255).astype(np.uint8)

    return dst


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    twisted_Lena = cv2.imread('final1.png', cv2.IMREAD_GRAYSCALE)

    # translation
    M1 = ???
    M2 = ???
    M3 = ???
    M4 = ???

    final1 = backward_fit(src, M1)
    final2 = backward_fit(src, M2)
    final3 = backward_fit(src, M3)
    final4 = backward_fit(src, M4)

    twisted_final1 = backward_fit(twisted_Lena, M1)
    twisted_final2 = backward_fit(twisted_Lena, M2)
    twisted_final3 = backward_fit(twisted_Lena, M3)
    twisted_final4 = backward_fit(twisted_Lena, M4)

    plot_out_list([src, final1, final2, final3, final4], ['Original', 'final1', 'final2', 'final3', 'final4'], figsize=(15, 15))
    plot_out_list([src, twisted_final1, twisted_final2, twisted_final3, twisted_final4], ['Original', 'final1', 'final2', 'final3', 'final4'], figsize=(15, 15))

    cv2.waitKey()
    cv2.destroyAllWindows()