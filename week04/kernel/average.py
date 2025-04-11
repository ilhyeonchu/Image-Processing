from typing import Tuple
import numpy as np


class Average:
    def __init__(self, kernel_size:Tuple):
        """
        :param kernel_size: (kernel height, kernel width)
        """

        # TODO: <-- 이 부분에 코드가 한 줄 이상 들어갑니다! -->
        
        kernel_height, kernel_width = kernel_size
        kernel = np.ones((kernel_height, kernel_width), dtype=np.float32) / (kernel_height * kernel_width)
        self.values = kernel # 생성된 kernel을 self.values에 저장