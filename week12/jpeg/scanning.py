import numpy as np


class ZigzagScanning:
    @classmethod
    def encode(cls, quantized_values):
        """
        Quantization이 완료된 블록을 zigzag 스캔하여 1차원 배열로 변환
        
        Args:
            quantized_values: Quantization 이 완료된 block (N x N)
            
        Returns:
            list: Zigzag 스캔 결과 (1차원 배열 + 'EOB')
        """
        assert quantized_values.shape[0] == quantized_values.shape[1]
        
        size = quantized_values.shape[0]
        result = []
        
        # 8x8 블록에 대한 하드코딩된 zigzag 인덱스
        zigzag_indices = [
            (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
            (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
            (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
            (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
            (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
            (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
            (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
            (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
        ]
        
        # EOB(End of Block) 찾기
        eob_found = False
        for i, j in zigzag_indices:
            if i < size and j < size:  # 블록 크기가 8x8보다 작을 수 있으므로 확인
                val = int(quantized_values[i, j])
                result.append(val)
                if val != 0:
                    eob_found = True
                    last_non_zero = len(result) - 1
        
        # EOB 추가 (0이 아닌 마지막 값 다음에 EOB 추가)
        if eob_found:
            result = result[:last_non_zero + 1]
        result.append('EOB')
        
        return result

    @classmethod
    def decode(cls, encoded_values, block_size):
        """
        Zigzag 스캔된 1차원 배열을 원래의 2차원 블록으로 복원
        
        Args:
            encoded_values: Zigzag scanning의 결과 (1차원 배열 + 'EOB')
            block_size: 복원할 블록의 크기 (N x N)
            
        Returns:
            numpy.ndarray: 복원된 2차원 블록
        """
        # EOB 제거
        if 'EOB' in encoded_values:
            encoded_values = encoded_values[:encoded_values.index('EOB')]
        
        # 0으로 초기화된 블록 생성
        block = np.zeros((block_size, block_size), dtype=np.int64)
        
        size = block_size
        idx = 0
        
        # 대각선 방향으로 순회하면서 값 채우기
        for s in range(2 * size - 1):
            if s % 2 == 0:  # 아래에서 위로
                if s < size:
                    row = s
                    col = 0
                else:
                    row = size - 1
                    col = s - size + 1
                
                while row >= 0 and col < size and idx < len(encoded_values):
                    block[row, col] = encoded_values[idx]
                    row -= 1
                    col += 1
                    idx += 1
            else:  # 위에서 아래로
                if s < size:
                    row = 0
                    col = s
                else:
                    row = s - size + 1
                    col = size - 1
                
                while row < size and col >= 0 and idx < len(encoded_values):
                    block[row, col] = encoded_values[idx]
                    row += 1
                    col -= 1
                    idx += 1
        
        return block


"""
                                        아래는 테스트 코드입니다.
                                        이 파일을 실행시키면 구현한 메서드에 대해 테스트가 가능합니다.
"""
if __name__ == "__main__":
    encoding_input = np.array([
        [-26, -3, -6, 2, 2, 0, 0, 0],
        [1, -2, -4, 0, 0, 0, 0, 0],
        [-3, 1, 5, -1, -1, 0, 0, 0],
        [-4, 1, 2, -1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ], np.int64)
    encoding_output = [-26, -3, 1, -3, -2, -6, 2, -4, 1, -4, 1, 1, 5, 0, 2, 0, 0, -1, 2, 0, 0, 0, 0, 0, -1, -1, 'EOB']

    test_encode = ZigzagScanning.encode(encoding_input)

    print("\033[97m" +"-"*30)
    print("[Zigzag scanning]")
    print("Encoding test", end="... ")
    if test_encode == encoding_output:
        print("\033[32m" + "PASSED!")
    else:
        print("\033[31m" + "FAILED!")

    test_decode = ZigzagScanning.decode(encoding_output, 8)
    print("\033[97m" +"Decoding test", end="... ")
    if np.sum(test_decode == encoding_input) == 64:
        print("\033[32m" + "PASSED!")
    else:
        print("\033[31m" + "FAILED!")
    print("\033[97m" +"-"*30)

