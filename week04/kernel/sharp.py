from typing import Tuple
import numpy as np

class Sharp:
    def __init__(self, kernel_size: Tuple):
        """
        :param kernel_size: (kernel height, kernel width)
        """

        # TODO: <-- 이 부분에 코드가 한 줄 이상 들어갑니다! -->
        kernel_height, kernel_width = kernel_size
        kernel1 = np.zeros((kernel_height, kernel_width), dtype=np.float32)
        kernel1[kernel_height // 2, kernel_width // 2] = 2.0
        kernel2 = np.ones((kernel_height, kernel_width), dtype=np.float32)
        kernel2 /= (kernel_height * kernel_width)
        kernel = kernel1 - kernel2
        
        self.values = kernel # 생성된 kernel을 self.values에 저장