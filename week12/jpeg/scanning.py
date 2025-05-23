import numpy as np


class ZigzagScanning:
    @classmethod
    def encode(cls, quantized_values):
        """
        :param quantized_values: Quantization 이 완료된 block
        :return: Scanning 결과
        """
        assert quantized_values.shape[0] == quantized_values.shape[1]

        ???

        return encode_result

    @classmethod
    def decode(cls, encoded_values, block_size):
        """
        :param encoded_values: Zigzag scanning의 결과
        :param block_size: Quantization 된 block
        :return:
        """

        ???

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

