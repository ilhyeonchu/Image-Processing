import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def get_histogram(img:np.ndarray, mask:Optional[np.ndarray]=None):
    """
    :param img: Histogram 을 계산할 이미지
    :param mask: Mask 가 있을 경우, "Mask 가 있는 곳에 대해서만" histogram 을 계산
    :return: Masking 여부에 따른 image 의 histogram
    """
    assert img.dtype == np.uint8, "이미지 타입이 uint8이 아닙니다."

    values = img.reshape(-1)

    if mask is not None:
        assert mask.dtype == np.bool_, "마스크 타입이 bool이 아닙니다."
        values = img[mask].reshape(-1)

    histogram = np.zeros((256,), dtype=np.float32)

    # TODO: <-- 이 부분에 코드가 한 줄 이상 들어갑니다! -->
    # 각 밝기 값의 빈도수 세기
    for intensity in values:
        histogram[intensity] += 1

    return histogram


def find_threshold(img:np.ndarray, mask:Optional[np.ndarray]=None):
    """
    :param img: Threshold 를 계산할 이미지
    :param mask: Mask 가 있을 경우, "Mask 가 있는 곳에 대해서만" threshold 를 계산
    :return: Masking 여부에 따른 image 의 threshold
    """
    histogram = get_histogram(img, mask)

    intensities = np.arange(256) # 존재할 수 있는 밝기값 (0~255)
    probs = histogram / (np.sum(histogram) + 1e-8) # 각 밝기 값이 존재할 확률, 이론 자료의 "p"에 해당

    left_prob = 0   # Threshold 기준 왼쪽 범위의 밝기 값이 등장할 확률, 이론 자료의 "q1"에 해당
    left_mean = 0   # Threshold 기준 왼쪽 범위에 대한 평균, 이론 자료의 "m1"에 해당

    right_prob = 1  # Threshold 기준 오른쪽 범위의 밝기 값이 등장할 확률, 이론 자료의 "q2"에 해당
    right_mean = np.sum(intensities * probs) # Threshold 기준 오른쪽 범위에 대한 평균, 이론 자료의 "m2"에 해당

    between_class_variances = []

    for intensity in range(256):
        # TODO: <-- 이 부분에 코드가 한 줄 이상 들어갑니다! -->
        # Intensity 에 대해 between-class variance 계산
        # 실습 자료의 수식에 따라 moving average 를 적용
        # Divide-by-zero 에러를 피하기 위해 분모에 1e-8을 더해줄 것!
        p = probs[intensity]
        left_prob += p
        right_prob -= p

        if left_prob > 0:
            left_mean = (left_mean * (left_prob - p) + intensity * p) / (left_prob + 1e-8)
        if right_prob > 0:
            right_mean = (right_mean * (right_prob + p) - intensity * p) / (right_prob + 1e-8)

        sigma_b = left_prob * right_prob * (left_mean - right_mean) ** 2
        between_class_variances.append(sigma_b)

    threshold = int(np.argmax(between_class_variances))

    return threshold

if __name__ == '__main__':
    meat = cv2.imread('../meat.png')
    meat_gray = cv2.imread('../meat.png', cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread('../mask.png', cv2.IMREAD_GRAYSCALE) > 0

    pad = np.zeros_like(meat_gray)

    # Masking 을 적용했을 때의 threshold 찾기
    threshold_without_mask = find_threshold(meat_gray)
    fat_without_mask = (meat_gray >= threshold_without_mask).astype(np.uint8)
    fat_without_mask_3ch = np.stack([pad, fat_without_mask*255, pad], axis=-1)
    no_mask_result = cv2.addWeighted(meat, 1, fat_without_mask_3ch.astype(np.uint8), 0.5, 0)
    fat_ratio_no_mask = np.round(np.sum(fat_without_mask * mask) / np.sum(mask), decimals=4)

    # Masking 을 적용하지 않았을 때의 threshold 찾기
    threshold_with_mask = find_threshold(meat_gray, mask)
    fat_with_mask = (meat_gray >= threshold_with_mask).astype(np.uint8)
    fat_with_mask_3ch = np.stack([pad, fat_with_mask*255, pad], axis=-1)
    mask_result = cv2.addWeighted(meat, 1, fat_with_mask_3ch.astype(np.uint8), 0.5, 0)
    fat_ratio_mask = np.round(np.sum(fat_with_mask * mask) / np.sum(mask), decimals=4)

    # Visualization
    print(f"[Fat ratio]")
    print(f"No masking : {fat_ratio_no_mask}")
    print(f"Masking    : {fat_ratio_mask}")

    plt.figure(figsize=(10, 7))
    plt.subplot(2, 3, 1)
    plt.imshow(meat[..., ::-1])
    plt.title("[Input]")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(fat_without_mask, cmap="gray")
    plt.title("[Segmentation] No masking")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(no_mask_result[..., ::-1])
    plt.title("[Result] No masking")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(meat[..., ::-1])
    plt.title("[Input]")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(fat_with_mask * mask, cmap="gray")
    plt.title("[Segmentation] Masking")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(mask_result[..., ::-1] * mask[..., None])
    plt.title("[Result] Masking")
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.subplot(2, 3, 1)
    plt.imshow(meat[100:160, 135:190, ::-1])
    plt.title("[Input]")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.title("[Result] No masking")
    plt.imshow(no_mask_result[100:160, 135:190, ::-1])
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.title("[Result] Masking")
    plt.imshow(mask_result[100:160, 135:190, ::-1])
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.title("[Input]")
    plt.imshow(meat[213:275, 130:170, ::-1])
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.title("[Result] No masking")
    plt.imshow(no_mask_result[213:275, 130:170, ::-1])
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.title("[Result] Masking")
    plt.imshow(mask_result[213:275, 130:170, ::-1])
    plt.axis("off")

    plt.show()
