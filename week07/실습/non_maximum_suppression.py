import numpy as np
import cv2

def min_max_scaling(src):
    return ((src - src.min()) / (src.max() - src.min()))

def get_sobel_filter():
    blurring = np.array([[1], [2], [1]])
    derivative_1d = np.array([[1, 0, -1]])

    Sobel_x = blurring @ derivative_1d
    Sobel_y = derivative_1d.T @ blurring.T

    return Sobel_x, Sobel_y

def calculate_magnitude(gradient_x, gradient_y):
    magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    return magnitude

def non_maximum_suppression_NN(gradient_x, gradient_y, magnitude):
    h, w = magnitude.shape
    large_magnitude = np.zeros(magnitude.shape)
    gradient_x_2 = gradient_x.copy()
    # np.arctan 사용시 라디안 값 나옴. 범위는 (-π/2, π/2)
    angle = np.arctan(gradient_y / gradient_x_2)

    for row in range(h):
        for col in range(w):
            neighbor = [magnitude[row, col]]
            # Case 1.
            if -np.pi / 8 <= angle[row, col] <= np.pi / 8:
                neighbor.append(magnitude[row, min(col + 1, w - 1)])
                neighbor.append(magnitude[row, max(col - 1, 0)])
            # Case 2-1.
            elif -3 * np.pi / 8 <= angle[row, col] < -np.pi / 8:
                neighbor.append(magnitude[max(row - 1, 0), min(col + 1, w - 1)])
                neighbor.append(magnitude[min(row + 1, h - 1), max(col - 1, 0)])
            # Case 2-2.
            elif np.pi / 8 < angle[row, col] <= 3 * np.pi / 8:
                neighbor.append(magnitude[min(row + 1, h - 1), min(col + 1, w - 1)])
                neighbor.append(magnitude[max(row - 1, 0), max(col - 1, 0)])
            # Case 3.
            else:
                neighbor.append(magnitude[min(row + 1, h - 1), col])
                neighbor.append(magnitude[max(row - 1, 0), col])
            # 이웃 magnitude보다 크거나 같다면 유지
            if magnitude[row, col] == max(neighbor):
                large_magnitude[row, col] = magnitude[row, col]

    return large_magnitude

def main():
    src = cv2.imread('cameraman.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    Sobel_x, Sobel_y = get_sobel_filter()
    gradient_x = cv2.filter2D(src, -1, Sobel_x)
    gradient_y = cv2.filter2D(src, -1, Sobel_y)
    magnitude = calculate_magnitude(gradient_x=gradient_x, gradient_y=gradient_y)
    nms_NN_result = non_maximum_suppression_NN(gradient_x, gradient_y, magnitude)

    abs_gradient_x = np.abs(gradient_x)
    abs_gradient_y = np.abs(gradient_y)
    cv2.imshow('abs_gradient_x', min_max_scaling(abs_gradient_x))
    cv2.imshow('abs_gradient_y', min_max_scaling(abs_gradient_y))
    cv2.imshow('magnitude', min_max_scaling(magnitude))
    cv2.imshow('nms_NN', min_max_scaling(nms_NN_result))
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()