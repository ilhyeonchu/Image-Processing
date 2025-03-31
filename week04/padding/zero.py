import numpy as np
from .padding import Padding


class Zero(Padding):
    def __init__(self):
        super().__init__()

    def __call__(self, image: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:

        # TODO: <-- 이 부분에 코드가 한 줄 이상 들어갑니다! -->

        return padded_image # 패딩된 이미지 반환