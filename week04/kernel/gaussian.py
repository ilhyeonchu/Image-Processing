from typing import Tuple
import numpy as np

class Gaussian:
    def __init__(self, kernel_size:Tuple, sigma:float):
        """
        :param kernel_size: (kernel height, kernel width)
        :param sigma:
        """

        # TODO: <-- 이 부분에 코드가 한 줄 이상 들어갑니다! -->
        
        kernel_height, kernel_width = kernel_size
        kernel = np.zeros((kernel_height, kernel_width), dtype=np.float32)
        c_h = kernel_height // 2
        c_w = kernel_width // 2
        
        sum = 0.0
        
        for i in range(kernel_height):
            for j in range(kernel_width):
                
                x = i - c_h
                y = j - c_w
                
                expon = -(x * x + y * y) / (2 * sigma * sigma)
                kernel[i, j] = np.exp(expon) / (2 * np.pi * sigma * sigma)  # e를 어떻게 표현하지지
                sum += kernel[i, j]
        
        kernel /= sum
        

        self.values = kernel # 생성된 kernel을 self.values에 저장