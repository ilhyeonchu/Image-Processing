import cv2
import numpy as np
from collections import deque


def get_hist(src):
    ???
    return hist

def otsu_method(src):
    ???
    return threshold

def dilation(B, S):
    ???
    return dst

def erosion(B, S):
    ???
    return dst

def opening(B, S):
    ???
    return dst

def closing(B, S):
    ???
    return dst

def main():
    original = cv2.imread('cell.png')
    gray_scale = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    B_test = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]])
    S = np.array(
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]])

    mask = ???
    original[mask == 0] -= 127
    cv2.imwrite('result.png', original)

if __name__ == '__main__':
    main()
