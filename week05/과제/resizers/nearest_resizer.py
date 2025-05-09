import numpy as np
from .base_resizer import BaseResizer

class NearestResizer(BaseResizer):
    def resize(self):
        new_img = self.new_img
        old_img = self.old_img
        a_y, a_x, b_y, b_x = self.a_y, self.a_x, self.b_y, self.b_x

        for row in range(self.new_shape[0]):
            for col in range(self.new_shape[1]):
                # TODO : NN interpolation 구현
                
                # nearest이므로 0.5를 더해주고 소수점을 버리면 가장 가까운 좌표를 찾을 수 있음(수업자료)
                y = int(row * a_y + b_y + 0.5)
                x = int(col * a_x + b_x + 0.5)
                # 범위를 벗어나지 않게 min 사용
                y = min(y, old_img.shape[0] - 1)
                x = min(x, old_img.shape[1] - 1)

                # old_img에서 값 가져오기
                intensity = old_img[y, x]

                new_img[row, col] = intensity

        return new_img
