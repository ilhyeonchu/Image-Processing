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
                y = ???
                x = ???

                y_floor = ???
                x_floor = ???
                y_ceil = ???
                x_ceil = ???

                intensity = ???

                new_img[row, col] = round(intensity)

        return new_img
