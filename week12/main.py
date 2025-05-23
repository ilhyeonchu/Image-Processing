import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from jpeg import JPEG

def pad_right_and_bottom(image, pad_right, pad_bottom):
    """
    주어진 이미지의 오른쪽 및 아래에 zero padding을 수행
    """
    h, w, c = image.shape
    padded_image = ???

    return padded_image

if __name__ == "__main__":
    BLOCK_SIZE = 8 # TODO : block size를 다양하게 실험
    QUANTIZATION_SCALES = [1, 5, 10]

    img = cv2.imread("image.jpg")[...,::-1]
    pad_bottom = ???
    pad_right = ???

    ######################### 아래는 수정 금지 #########################
    padded_img = pad_right_and_bottom(img, pad_right, pad_bottom)

    outputs = []
    for quant_scale in QUANTIZATION_SCALES:
        output = []
        print(f"\nScale {quant_scale} processing...")
        for channel in tqdm(range(3)):
            inp = padded_img[..., channel]
            encoding_result = JPEG.encode(inp, BLOCK_SIZE, quant_scale)
            decoding_result = JPEG.decode(encoding_result, padded_img.shape[:2], BLOCK_SIZE, quant_scale)
            output.append(decoding_result)
        output = np.stack(output, axis=-1)
        output = output[:img.shape[0], :img.shape[1]]
        outputs.append(output)

    cv2.imwrite("./decompressed_image.png", outputs[0]) # 전체 이미지 확인용

    plt.figure(figsize=(10, 8))
    plt.subplot(3, 3, 1)
    plt.imshow(img[158:243,55:226])
    plt.subplot(3, 3, 2)
    plt.imshow(outputs[0][158:243,55:226])
    plt.subplot(3, 3, 3)
    diff = img.astype(np.float32) - outputs[0].astype(np.float32) + 128
    diff = (diff - diff.min()) / (diff.max() - diff.min())
    plt.imshow(diff[158:243,55:226])

    plt.subplot(3, 3, 4)
    plt.imshow(img[158:243, 55:226])
    plt.subplot(3, 3, 5)
    plt.imshow(outputs[1][158:243, 55:226])
    plt.subplot(3, 3, 6)
    diff = img.astype(np.float32) - outputs[1].astype(np.float32) + 128
    diff = (diff - diff.min()) / (diff.max() - diff.min())
    plt.imshow(diff[158:243, 55:226])

    plt.subplot(3, 3, 7)
    plt.imshow(img[158:243, 55:226])
    plt.subplot(3, 3, 8)
    plt.imshow(outputs[2][158:243, 55:226])
    plt.subplot(3, 3, 9)
    diff = img.astype(np.float32) - outputs[2].astype(np.float32) + 128
    diff = (diff - diff.min()) / (diff.max() - diff.min())
    plt.imshow(diff[158:243, 55:226])
    plt.show()