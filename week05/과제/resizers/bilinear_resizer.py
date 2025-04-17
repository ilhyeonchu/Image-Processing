import numpy as np
from .base_resizer import BaseResizer

class BilinearResizer(BaseResizer):
    def resize(self):
        new_img = self.new_img
        old_img = self.old_img
        a_y, a_x, b_y, b_x = self.a_y, self.a_x, self.b_y, self.b_x

        for row in range(self.new_shape[0]):
            for col in range(self.new_shape[1]):
                # TODO : Bilinear interpolation 구현
                y = row * a_y + b_y
                x = col * a_x + b_x
                
                # 왼쪽 위 좌표, 오른쪽 아래 좌표(범위 안벗어나게 min 사용용)
                y_floor = int(y)
                x_floor = int(x)
                y_ceil = min(y_floor + 1, old_img.shape[0] - 1)
                x_ceil = min(x_floor + 1, old_img.shape[1] - 1)
                # 주변 4개의 픽셀 값
                f1 = old_img[y_floor, x_floor]
                f2 = old_img[y_floor, x_ceil]
                f3 = old_img[y_ceil, x_floor]
                f4 = old_img[y_ceil, x_ceil]
                
                # 가중치 계산산
                w1 = (y_ceil - y) * (x_ceil - x)
                w2 = (y_ceil - y) * (x - x_floor)
                w3 = (y - y_floor) * (x_ceil - x)
                w4 = (y - y_floor) * (x - x_floor)
                
                intensity = f1 * w1 + f2 * w2 + f3 * w3 + f4 * w4

                new_img[row, col] = round(intensity)

        return new_img
