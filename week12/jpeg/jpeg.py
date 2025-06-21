import numpy as np
import cv2
try:
    from .dct import DiscreteCosineTransform as DCT
    from .scanning import ZigzagScanning as ZS
except ImportError as e:
    from dct import DiscreteCosineTransform as DCT
    from scanning import ZigzagScanning as ZS

class JPEG:
    QUANTIZATION_MATRIX = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                    [12, 12, 14, 19, 26, 58, 60, 55],
                                    [14, 13, 16, 24, 40, 57, 69, 56],
                                    [14, 17, 22, 29, 51, 87, 80, 62],
                                    [18, 22, 37, 56, 68, 109, 103, 77],
                                    [24, 35, 55, 64, 81, 104, 113, 92],
                                    [49, 64, 78, 87, 103, 121, 120, 101],
                                    [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.float64)

    @classmethod
    def get_quantization_matrix(cls, block_size, quantization_scale):
        """
        주어진 quantization matrix 은 8x8의 형태만 존재

        Block 크기가 달라지면 이에 맞는 quantization matrix를 구해야하나, 이를 구하는 수식은 제공되지 않음
        [참고] https://stackoverflow.com/questions/29215879/how-can-i-generalize-the-quantization-matrix-in-jpeg-compression

        따라서 8x8 형태의 quantization matrix를 interpolation을 통해 크기를 조정해서 사용

        @@@@@@@@@@@ 이때 "cv2.resize" 를 사용할것! @@@@@@@@@@@


        :param block_size:
        :param quantization_scale: Scale을 얼마나 줄지에 대한 값
        :return:
        """
        scaled_matrix = np.clip(cls.QUANTIZATION_MATRIX * quantization_scale, 1, 255)  # 0으로 나누기 방지를 위해 최소값을 1로 설정
        # 양자화 행렬을 block_size에 맞게 리사이즈 (cv2.INTER_LINEAR 사용)
        resized_matrix = cv2.resize(scaled_matrix, (block_size, block_size), interpolation=cv2.INTER_LINEAR)
        return resized_matrix

    @classmethod
    def encode(cls, img, block_size, quantization_scale=1):
        quantization_matrix = cls.get_quantization_matrix(block_size, quantization_scale)
        img_blocks = cls.image_to_blocks(img, block_size)
        compressed_blocks = []
        for img_block in img_blocks:
            img_block = img_block.copy().astype(np.float64) - 128
            dct_result = DCT.spatial_to_frequency(img_block)
            quantization_result = np.round(dct_result / quantization_matrix).astype(np.int64)
            compressed_block = ZS.encode(quantization_result)
            compressed_blocks.append(compressed_block)

        compressed_img = cls.get_residual(compressed_blocks, img.shape[:2], block_size)
        return compressed_img

    @classmethod
    def decode(cls, compressed_img, img_size, block_size, quantization_scale=1):
        quantization_matrix = cls.get_quantization_matrix(block_size, quantization_scale)
        compressed_blocks = cls.get_inverse_residual(compressed_img, img_size, block_size)
        decompressed_blocks = []
        for compressed_block in compressed_blocks:
            inverse_scanning_result = ZS.decode(compressed_block, block_size)
            inverse_quantization_result = inverse_scanning_result * quantization_matrix
            inverse_quantization_result = inverse_quantization_result.astype(np.float64)
            inverse_dct_result = DCT.frequency_to_spatial(inverse_quantization_result)
            decompressed_block = inverse_dct_result + 128

            decompressed_blocks.append(decompressed_block)

        decompressed_blocks = np.array(decompressed_blocks)
        decompressed_img = cls.blocks_to_image(decompressed_blocks, img_size, block_size)

        decompressed_img = np.round(decompressed_img)
        decompressed_img = np.clip(decompressed_img, 0, 255).astype(np.uint8)

        return decompressed_img

    @classmethod
    def image_to_blocks(cls, img, block_size):
        """
        이미지를 블럭 크기에 맞게 자르는 메서드
        
        Args:
            img: 입력 이미지 (H, W) or (H, W, C)
            block_size: 블록 크기
            
        Returns:
            numpy.ndarray: 블록으로 분할된 이미지 (n_blocks, block_size, block_size)
        """
        h, w = img.shape[:2]
        
        # 2D 이미지인 경우 채널 차원 추가
        if len(img.shape) == 2:
            img = img[..., np.newaxis]
            
        # 블록으로 분할
        n_blocks_h = h // block_size
        n_blocks_w = w // block_size
        
        # (n_blocks_h, block_size, n_blocks_w, block_size, C)
        blocks = img.reshape(n_blocks_h, block_size, n_blocks_w, block_size, -1)
        # (n_blocks_h, n_blocks_w, block_size, block_size, C)
        blocks = blocks.transpose(0, 2, 1, 3, 4)
        # (n_blocks_h * n_blocks_w, block_size, block_size, C)
        blocks = blocks.reshape(-1, block_size, block_size, blocks.shape[-1])
        
        # 채널이 1개인 경우 마지막 차원 제거
        if blocks.shape[-1] == 1:
            blocks = blocks[..., 0]
            
        return blocks

    @classmethod
    def blocks_to_image(cls, blocks, img_size, block_size):
        """
        블록들을 원본 이미지 형태로 변환하는 메서드
        
        Args:
            blocks: 블록으로 분할된 이미지 (n_blocks, block_size, block_size)
            img_size: 원본 이미지 크기 (H, W)
            block_size: 블록 크기
            
        Returns:
            numpy.ndarray: 원본 이미지 형태로 변환된 배열 (H, W)
        """
        h, w = img_size
        n_blocks_h = h // block_size
        n_blocks_w = w // block_size
        
        # 2D 블록인 경우 채널 차원 추가
        if len(blocks.shape) == 3:
            blocks = blocks[..., np.newaxis]
            
        # 블록을 원래 이미지 형태로 재구성
        # (n_blocks_h, n_blocks_w, block_size, block_size, C)
        blocks = blocks.reshape(n_blocks_h, n_blocks_w, block_size, block_size, -1)
        # (n_blocks_h, block_size, n_blocks_w, block_size, C)
        blocks = blocks.transpose(0, 2, 1, 3, 4)
        # (n_blocks_h * block_size, n_blocks_w * block_size, C)
        image = blocks.reshape(h, w, -1)
        
        # 채널이 1개인 경우 마지막 차원 제거
        if image.shape[-1] == 1:
            image = image[..., 0]
            
        return image

    @classmethod
    def get_residual(cls, compressed_blocks, img_shape, block_size):
        """
        블록 간의 잔차를 계산하는 메서드
        
        Args:
            compressed_blocks: Zigzag scanning이 적용된 블록들
            img_shape: 입력 이미지의 크기 (H, W)
            block_size: 블록 크기
            
        Returns:
            list: 잔차가 적용된 블록들의 리스트
        """
        img_h, img_w = img_shape

        vertical_block_num = img_h // block_size
        horizontal_block_num = img_w // block_size

        assert len(compressed_blocks) == vertical_block_num * horizontal_block_num, "블럭 개수가 다릅니다."

        residuals = []
        prev_block = None
        
        for i, block in enumerate(compressed_blocks):
            if i == 0:
                # 첫 번째 블록은 그대로 유지
                residuals.append(block.copy() if isinstance(block, list) else block.tolist())
            else:
                # 이전 블록과의 차이를 계산
                residual_block = []
                min_len = min(len(block), len(prev_block))
                for j in range(min_len):
                    if isinstance(block[j], str) and block[j] == 'EOB':
                        residual_block.append('EOB')
                        break
                    if isinstance(prev_block[j], str) or isinstance(block[j], str):
                        continue
                    residual_block.append(block[j] - prev_block[j])
                
                # EOB가 없으면 추가
                if 'EOB' not in residual_block and residual_block:
                    residual_block.append('EOB')
                residuals.append(residual_block)
            
            prev_block = block.copy() if isinstance(block, list) else block.tolist()
            
        return residuals

    @classmethod
    def get_inverse_residual(cls, residuals, img_shape, block_size):
        """
        잔차로부터 원본 블록들을 복원하는 메서드
        
        Args:
            residuals: 잔차가 적용된 블록들의 리스트
            img_shape: 이미지 크기 (H, W)
            block_size: 블록 크기
            
        Returns:
            list: 복원된 블록들의 리스트
        """
        img_h, img_w = img_shape

        vertical_block_num = img_h // block_size
        horizontal_block_num = img_w // block_size

        assert len(residuals) == vertical_block_num * horizontal_block_num, "블럭 개수가 다릅니다."

        compressed_blocks = []
        prev_block = None
        
        for i, residual in enumerate(residuals):
            if i == 0:
                # 첫 번째 블록은 그대로 유지
                compressed_blocks.append(residual.copy() if isinstance(residual, list) else residual.tolist())
            else:
                # 이전 블록에 잔차를 더해 원본 블록 복원
                recovered_block = []
                min_len = min(len(residual), len(prev_block))
                for j in range(min_len):
                    if isinstance(residual[j], str) and residual[j] == 'EOB':
                        recovered_block.append('EOB')
                        break
                    if isinstance(prev_block[j], str) or (j < len(residual) and isinstance(residual[j], str)):
                        continue
                    recovered_block.append(residual[j] + prev_block[j])
                
                # EOB가 없으면 추가
                if 'EOB' not in recovered_block and recovered_block:
                    recovered_block.append('EOB')
                compressed_blocks.append(recovered_block)
            
            prev_block = compressed_blocks[-1].copy() if isinstance(compressed_blocks[-1], list) else compressed_blocks[-1].tolist()
            
        return compressed_blocks


"""
                                        아래는 테스트 코드입니다.
                                        이 파일을 실행시키면 구현한 메서드에 대해 테스트가 가능합니다.
"""
if __name__ == "__main__":
    test1_input = [[26, 3, 5, -6, 2, 3, "EOB"],
                  [1, 3, 2, 9, -53, -23, 12, 30, "EOB"],
                  [-6, 2, "EOB"],
                  [-13, 14, -18, 22, 42, "EOB"]]

    test1_answer = [[26, 3, 5, -6, 2, 3, 'EOB'],
                    [-25, 0, -3, 15, -55, -26, 12, 30, 'EOB'],
                    [-32, -1, -5, 6, -2, -3, 'EOB'],
                    [-7, 12, -18, 22, 42, 'EOB']]

    test2_answer = [[26, 3, 5, -6, 2, 3, 'EOB'],
                    [1, 3, 2, 9, -53, -23, 12, 30, 'EOB'],
                    [-6, 2, 0, 0, 0, 0, 'EOB'],
                    [-13, 14, -18, 22, 42, 0, 'EOB']]

    print("\033[97m" + "-" * 30)
    print("[JPEG]")
    print("Residual computation test", end="........... ")
    test1_output = JPEG.get_residual(test1_input, (2,2), 1)
    if test1_answer == test1_output:
        print("\033[32m" + "PASSED!")
    else:
        print("\033[31m" + "FAILED!")

    print("\033[97m" + "Inverse residual computation test", end="... ")
    test2_output = JPEG.get_inverse_residual(test1_answer, (2, 2), 1)

    if (test1_input == test2_output) or (test2_answer == test2_output):
        print("\033[32m" + "PASSED!")
    else:
        print("\033[31m" + "FAILED!")
    print("\033[97m" +"-"*30)
