import numpy as np

class DiscreteCosineTransform:
    @classmethod
    def constant(cls, value, size):
        """
        Value 값에 따라 상수를 다르게 적용.
        이론 자료의 수식을 참고
        
        Args:
            value: 상수 인덱스 (u 또는 v)
            size: 블록 크기
            
        Returns:
            상수 값 (1/sqrt(2) if value == 0 else 1.0)
        """
        assert size > 0, "크기는 0일 수 없습니다."
        
        if value == 0:
            return 1.0 / np.sqrt(2)
        else:
            return 1.0

    @classmethod
    def spatial_to_frequency(cls, spatial_domain):
        """
        :param spatial_domain: 이미지로부터 가져온 block
        :return: 주파수의 형태로 변환된 block
        """
        assert spatial_domain.shape[0] == spatial_domain.shape[1]

        domain_size = spatial_domain.shape[0]

        frequency_domain = np.zeros_like(spatial_domain, dtype=np.float64)
        
        # DCT 공식 적용
        for v in range(domain_size):
            for u in range(domain_size):
                sum_val = 0.0
                for y in range(domain_size):
                    for x in range(domain_size):
                        # DCT 수식 적용
                        cos_x = np.cos((2*x + 1) * u * np.pi / (2 * domain_size))
                        cos_y = np.cos((2*y + 1) * v * np.pi / (2 * domain_size))
                        sum_val += spatial_domain[y, x] * cos_x * cos_y
                
                # 상수 항 적용
                c_u = cls.constant(u, domain_size)
                c_v = cls.constant(v, domain_size)
                frequency_domain[v, u] = 0.25 * c_u * c_v * sum_val

        return frequency_domain

    @classmethod
    def frequency_to_spatial(cls, frequency_domain):
        """
        :param frequency_domain: 주파수에 대한 block
        :return: 이미지에 대한 block
        """
        assert frequency_domain.shape[0] == frequency_domain.shape[1]

        domain_size = frequency_domain.shape[0]

        spatial_domain = np.zeros_like(frequency_domain, dtype=np.float64)
        
        # 역 DCT 공식 적용
        for y in range(domain_size):
            for x in range(domain_size):
                sum_val = 0.0
                for v in range(domain_size):
                    for u in range(domain_size):
                        # 역 DCT 수식 적용
                        c_u = cls.constant(u, domain_size)
                        c_v = cls.constant(v, domain_size)
                        cos_x = np.cos((2*x + 1) * u * np.pi / (2 * domain_size))
                        cos_y = np.cos((2*y + 1) * v * np.pi / (2 * domain_size))
                        sum_val += c_u * c_v * frequency_domain[v, u] * cos_x * cos_y
                
                spatial_domain[y, x] = 0.25 * sum_val

        return spatial_domain


"""
                                        아래는 테스트 코드입니다.
                                        이 파일을 실행시키면 구현한 메서드에 대해 테스트가 가능합니다.
"""
if __name__ == "__main__":
    DCT = DiscreteCosineTransform

    example_block = np.array([
        [52, 55, 61, 66, 70, 61, 64, 73],
        [63, 59, 66, 90, 109, 85, 69, 72],
        [62, 59, 68, 113, 144, 104, 66, 73],
        [63, 58, 71, 122, 154, 106, 70, 69],
        [67, 61, 68, 104, 126, 88, 68, 70],
        [79, 65, 60, 70, 77, 68, 58, 75],
        [85, 71, 64, 59, 55, 61, 65, 83],
        [87, 79, 69, 68, 65, 76, 78, 94]
    ], np.float64)

    test1_input = example_block - 128
    test1_answer = np.array([[-414, -29, -62, 25, 55, -20, -1, 2],
                                    [6, -21, -62, 8, 12, -7, -6, 7],
                                    [-46, 8, 77, -26, -30, 10, 6, -5],
                                    [-49, 12, 34, -14, -10, 6, 1, 1],
                                    [11, -8, -12, -2, 0, 1, -5, 2],
                                    [-10, 1, 3, -3, 0, 0, 2, 0],
                                    [-3, -1, 1, 0, 1, -4, 2, -3],
                                    [-1, -1, 0, -3, 0, 0, -1, 0]], np.int32)

    test2_input = np.array([[-416., -33., -60.,  32., 48., 0., 0., 0.],
                                   [12., -24., -56., 0., 0., 0., 0., 0.],
                                   [-42., 13., 80., -24., -40.,  0., 0., 0.],
                                   [-42., 17., 44., 0., 0., 0., 0., 0.],
                                   [18., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0.]], np.float64)

    test2_answer = [[-63, -63, -64, -65, -63, -58, -55, -53],
                     [-73, -73, -60, -39, -31, -42, -54, -59],
                     [-76, -79, -53, -7, 7, -22, -52, -61],
                     [-64, -78, -54, 1, 18, -19, -53, -58],
                     [-49, -74, -66, -23, -9, -38, -61, -58],
                     [-44, -70, -76, -56, -47, -61, -67, -58],
                     [-43, -59, -70, -69, -65, -65, -60, -51],
                     [-42, -48, -57, -65, -64, -56, -47, -41]]

    test1_output = np.round(DCT.spatial_to_frequency(test1_input)).astype(np.int32)
    print("\033[97m" + "-" * 30)
    print("[Discrete Cosine Transform]")
    print("Spatial domain --> frequency domain test", end="... ")
    if np.array(test1_output == test1_answer).sum() == 64:
        print("\033[32m" + "PASSED!")
    else:
        print("\033[31m" + "FAILED!")

    test2_output = np.round(DCT.frequency_to_spatial(test2_input)).astype(np.int32)
    print("\033[97m" + "Frequency domain --> spatial domain test", end="... ")
    if np.sum(test2_output == test2_answer) == 64:
        print("\033[32m" + "PASSED!")
    else:
        print("\033[31m" + "FAILED!")
    print("\033[97m" + "-" * 30)