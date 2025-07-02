import numpy as np
from .padding import Padding


class Zero(Padding):
    def __init__(self):
        super().__init__()

    def __call__(self, image: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:

        # 04주차 복사
        # 패딩해 저장할 값이 0인 배열을 생성
        padded_image = np.zeros((image.shape[0] + top + bottom, image.shape[1] + left + right), dtype=image.dtype)

        # 원본 이미지 넣기
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                padded_image[i + top, j + left] = image[i, j]

        return padded_image