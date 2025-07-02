import numpy as np


class ZigzagScanning:
    @classmethod
    def encode(cls, quantized_values):
        """
        :param quantized_values: Quantization 이 완료된 block
        :return: Scanning 결과
        """
        assert quantized_values.shape[0] == quantized_values.shape[1]

        block_size = quantized_values.shape[0]
        v_min, v_max = 0, block_size - 1
        h_min, h_max = 0, block_size - 1
        i, j = 0, 0
        encode_result = []
        while (i <= v_max) and (j <= h_max):
            encode_result.append(quantized_values[i, j])
            if (i + j) % 2 == 0:  # Going up
                if j == h_max:
                    i += 1
                elif i == v_min:
                    j += 1
                else:
                    i -= 1
                    j += 1
            else:  # Going down
                if i == v_max:
                    j += 1
                elif j == h_min:
                    i += 1
                else:
                    i += 1
                    j -= 1

        # Remove trailing zeros and add EOB
        while encode_result and encode_result[-1] == 0:
            encode_result.pop()
        encode_result.append('EOB')
        return encode_result

    @classmethod
    def decode(cls, encoded_values, block_size):
        """
        :param encoded_values: Zigzag scanning의 결과
        :param block_size: Quantization 된 block
        :return:
        """

        v_min, v_max = 0, block_size - 1
        h_min, h_max = 0, block_size - 1
        i, j = 0, 0
        decode_output = np.zeros((block_size, block_size), dtype=np.int64)
        values = encoded_values[:-1]  # Remove EOB
        k = 0
        while (i <= v_max) and (j <= h_max):
            if k < len(values):
                decode_output[i, j] = values[k]
                k += 1
            if (i + j) % 2 == 0:  # Going up
                if j == h_max:
                    i += 1
                elif i == v_min:
                    j += 1
                else:
                    i -= 1
                    j += 1
            else:  # Going down
                if i == v_max:
                    j += 1
                elif j == h_min:
                    i += 1
                else:
                    i += 1
                    j -= 1

        return np.array(decode_output)


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

