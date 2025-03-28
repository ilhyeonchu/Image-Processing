import numpy as np


class Padding:
    def __init__(self):
        pass

    def __call__(self, image: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
        """
        :param image: (image height, image width)
        :param top: 이미지 윗 부분 패딩 크기
        :param bottom: 이미지 아랫 부분 패딩 크기
        :param left: 이미지 왼쪽 부분 패딩 크기
        :param right: 이미지 오른쪽 부분 패딩 크기
        :return padded_image: 패딩이 적용된 이미지
        """
        pass