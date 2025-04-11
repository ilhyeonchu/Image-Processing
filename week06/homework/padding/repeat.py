import numpy as np
from .padding import Padding


class Repeat(Padding):
    def __init__(self):
        super().__init__()

    def __call__(self, image: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:

        padded_image = ???

        return padded_image