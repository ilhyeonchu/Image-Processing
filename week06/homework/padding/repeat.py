import numpy as np
from .padding import Padding


class Repeat(Padding):
    def __init__(self):
        super().__init__()

    def __call__(self, image: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:

        # 04주차 복사

        padded_image = np.zeros((image.shape[0] + top + bottom, image.shape[1] + left + right), dtype=image.dtype)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                padded_image[i + top, j + left] = image[i, j]

        # 패딩으로 늘어난 부분을 repetition으로 채우기
        for i in range(top):
            padded_image[i, :] = padded_image[top, :] # 0행부터 top-1행까지 채우기
        for i in range(image.shape[0] + top, image.shape[0] + top + bottom):
            padded_image[i, :] = padded_image[image.shape[0] + top - 1, :] # 원본 이미지 끝난 부분부터 마지막 행까지 채우기

        # 패딩으로 늘어난 부분을 repetition으로 채우기 (좌우)
        for i in range(left):
            padded_image[:, i] = padded_image[:, left]
        for i in range(image.shape[1] + left, image.shape[1] + left + right):
            padded_image[:, i] = padded_image[:, image.shape[1] + left - 1]

        return padded_image