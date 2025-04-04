import numpy as np

class BaseResizer:
    def __init__(self, old_img, new_shape):
        self.old_shape = old_img.shape
        self.old_img = old_img
        self.new_shape = new_shape
        # TODO : 빈 배열 만들기
        self.new_img = np.zeros(???, dtype=np.uint8)
        self.a_y, self.a_x, self.b_y, self.b_x = self._match_up_coordinates(self.old_shape, self.new_shape)

    def _match_up_coordinates(self, old_shape, new_shape):
        # TODO : a_y, a_x, b_y, b_x 구하기
        a_y = ???
        a_x = ???
        b_y = ???
        b_x = ???
        return a_y, a_x, b_y, b_x