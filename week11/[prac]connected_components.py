import numpy as np
import cv2
from collections import deque
def labeling(B, neighbor):
    ???
    
    return label_image, num_features

def main():
    example_2D = np.array([
        [1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [1, 0, 1, 1, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 1, 1, 1],
        [1, 1, 1, 0, 1, 0, 0, 1, 1, 1]
    ], np.uint8)

    label_image, num_features = labeling(example_2D * 255, neighbor=8)
    print(example_2D)
    print(label_image)

if __name__ == '__main__':
    main()