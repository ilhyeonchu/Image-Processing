from typing import Tuple
import numpy as np


class Gaussian:
    def __init__(self, filter_size:Tuple, sigma:float):
        """
        :param filter_size: filter의 높이 및 너비
        :param sigma:
        """

        height, width = filter_size
        center_y = height // 2
        center_x = width // 2
        gaussian_filter = np.zeros((height, width), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                # 중앙에서 떨어진 거리 dy,dx를 이용해 가우시안의 지수 부분 완성
                dy = y - center_y
                dx = x - center_x
                gaussian_filter[y, x] = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))

        # 지수 부분은 완성되어 있으니 계수부분, 정규화
        gaussian_filter /= (2 * np.pi * sigma**2)   # 반복문 밖에서 하는게 성능이 좋다는데 이유가?
        gaussian_filter /= np.sum(gaussian_filter)

        self.values = gaussian_filter # 생성된 filter를 self.values에 저장