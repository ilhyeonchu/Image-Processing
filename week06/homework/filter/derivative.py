import numpy as np
from .direction import Direction

class Derivative:
    def __init__(self, direction: Direction):
        """
        :param direction: filter의 방향, Direction.X인 경우 (1,3), Direction.Y인 경우 (3,1) 생성
        """

        # 자료에 나와있는 수치대로 filter 만들기
        if direction == Direction.X:
            derivative_filter = np.array([[-1, 0, 1]])
        elif direction == Direction.Y:
            derivative_filter = np.array([[-1], [0], [1]])
        else:
            raise Exception

        self.values = derivative_filter # 생성된 filter를 self.values에 저장