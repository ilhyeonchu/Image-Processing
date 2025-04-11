import numpy as np
from .direction import Direction

class Derivative:
    def __init__(self, direction: Direction):
        """
        :param direction: filter의 방향, Direction.X인 경우 (1,3), Direction.Y인 경우 (3,1) 생성
        """

        if direction == Direction.X:
            derivative_filter = ???
        elif direction == Direction.Y:
            derivative_filter = ???
        else:
            raise Exception

        self.values = derivative_filter # 생성된 filter를 self.values에 저장