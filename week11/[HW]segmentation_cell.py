import cv2
import numpy as np
from collections import deque


def get_hist(src):
    hist = np.zeros(256, dtype=np.int32)
    for r in range(src.shape[0]):
        for c in range(src.shape[1]):
            hist[src[r, c]] += 1
    return hist

def otsu_method(src):
    hist = get_hist(src)
    total_pixels = src.shape[0] * src.shape[1]

    current_max_variance = 0
    threshold = 0

    # 정규화된 히스토그램 계산
    normalized_hist = hist / total_pixels

    for t in range(1, 256): # 가능한 모든 임계값(1부터 255)에 대해 반복
        # 클래스 1 (배경)
        w1 = np.sum(normalized_hist[:t])
        # 클래스 2 (전경)
        w2 = np.sum(normalized_hist[t:])

        if w1 == 0 or w2 == 0:
            continue

        # 클래스 1의 평균
        mu1_numerator = np.sum(np.arange(t) * normalized_hist[:t])
        mu1 = mu1_numerator / w1

        # 클래스 2의 평균
        mu2_numerator = np.sum(np.arange(t, 256) * normalized_hist[t:])
        mu2 = mu2_numerator / w2

        # 클래스 간 분산 계산
        between_class_variance = w1 * w2 * ((mu1 - mu2) ** 2)

        if between_class_variance > current_max_variance:
            current_max_variance = between_class_variance
            threshold = t
    return threshold

def dilation(B, S):
    h, w = B.shape
    s_h, s_w = S.shape
    pad_h, pad_w = s_h // 2, s_w // 2  # 구조 요소 중심

    dst = np.zeros_like(B, dtype=np.uint8)

    for r_b in range(h):
        for c_b in range(w):
            if B[r_b, c_b] == 1:  # 입력 이미지의 전경 픽셀에 대해
                for r_s in range(s_h):
                    for c_s in range(s_w):
                        if S[r_s, c_s] == 1:  # 구조 요소의 전경 픽셀에 대해
                            # 결과 이미지에 마킹할 위치 계산
                            target_r, target_c = r_b + r_s - pad_h, c_b + c_s - pad_w
                            if 0 <= target_r < h and 0 <= target_c < w:
                                dst[target_r, target_c] = 1
    return dst

def erosion(B, S):
    h, w = B.shape
    s_h, s_w = S.shape
    pad_h, pad_w = s_h // 2, s_w // 2  # 구조 요소 중심

    dst = np.zeros_like(B, dtype=np.uint8)

    for r_dst in range(h):
        for c_dst in range(w):
            is_match = True  # 구조 요소가 완전히 맞는다고 가정
            for r_s in range(s_h):
                for c_s in range(s_w):
                    if S[r_s, c_s] == 1:  # 구조 요소의 전경 픽셀에 대해
                        source_r, source_c = r_dst + r_s - pad_h, c_dst + c_s - pad_w

                        # 이미지 경계를 벗어나거나, 입력 이미지의 해당 픽셀이 배경이면 불일치
                        if not (0 <= source_r < h and 0 <= source_c < w and B[source_r, source_c] == 1):
                            is_match = False
                            break  # 내부 루프 탈출
                if not is_match:
                    break  # 외부 루프 탈출

            if is_match:  # 구조 요소가 완전히 일치하면 결과 픽셀을 전경으로 설정
                dst[r_dst, c_dst] = 1
    return dst

def opening(B, S):
    eroded_image = erosion(B, S)
    opened_image = dilation(eroded_image, S)
    dst = opened_image
    return dst

def closing(B, S):
    dilated_image = dilation(B, S)
    closed_image = erosion(dilated_image, S)
    dst = closed_image
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

    threshold_value = otsu_method(gray_scale)
    binary_mask = np.where(gray_scale > threshold_value, 1, 0).astype(np.uint8)
    opened_mask = opening(binary_mask, S)
    final_mask = closing(opened_mask, S)
    mask = final_mask
    original[mask == 0] -= 127

    cv2.imwrite('binary_mask_initial.png', binary_mask * 255)
    cv2.imwrite('opened_mask.png', opened_mask * 255)
    cv2.imwrite('final_mask.png', final_mask * 255)

    cv2.imwrite('result.png', original)


if __name__ == '__main__':
    main()
