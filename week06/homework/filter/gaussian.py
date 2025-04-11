from typing import Tuple
import numpy as np


class Gaussian:
    def __init__(self, filter_size:Tuple, sigma:float):
        """
        :param filter_size: filter의 높이 및 너비
        :param sigma:
        """

        gaussian_filter = ???

        self.values = gaussian_filter # 생성된 filter를 self.values에 저장