import numpy as np
from .direction import Direction

class Laplace:
    def __init__(self, direction: Direction):
        """
        :param direction: filter의 방향, Direction.X인 경우 (1,3), Direction.Y인 경우 (3,1) 생성
        """

        # TODO: <-- 이 부분에 코드가 한 줄 이상 들어갑니다! -->

        # 어차피 계산은 main.py에서 다 하고 적용하므로 필터만 정의
        if direction == Direction.X:
            laplace_filter = np.array([[1, -2, 1]])
        elif direction == Direction.Y:
            laplace_filter = np.array([[1], [-2], [1]])
        else:
            raise Exception

        self.values = laplace_filter # 생성된 filter를 self.values에 저장