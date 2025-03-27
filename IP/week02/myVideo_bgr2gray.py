import cv2
import numpy as np
from tqdm import tqdm

def get_video_frames(video_path):
    """
    :param video_path:
    :return: video_frames
    """
    cap = cv2.VideoCapture(VIDEO_INPUT_FILE_PATH)  # 동영상 캡처 객체 생성

    # 저장된 비디오 한 프레임씩 읽기
    video_frames = []
    if cap.isOpened():  # 객체 초기화 정상
        while True:
            read_status, frame = cap.read()  # 다음 프레임 읽기

            # 프레임 비정상적으로 읽어진 경우
            if not read_status:
                break

            # 프레임이 정상적으로 읽어진 경우
            video_frames.append(frame)
    else:
        raise "비디오를 읽는 도중 오류가 발생했습니다."

    cap.release()
    return video_frames

def calc_bgr2gray(src):
    """
    :param src: color video
    :return: gray video
    """
    h, w, c = src.shape
    dst = np.zeros((h, w))

    # 과제 부분 시작
    for row in range(h):
        for col in range(w):
            ##############################################
            # TODO
            # B * 0.0721 + G * 0.7154 + R * 0.2125 공식을 사용
            # calc_bgr2gray 함수 완성
            # 과제 제출의 경우, 2중 for문 형태로 제출
            # 물론 2중 for문이 아닌 형태로 만들 수 있습니다.
            ##############################################

            #  <----- 이 부분에 코드가 한 줄 이상 들어갑니다. ----->
    # 과제 부분 끝

    dst = (dst+0.5).astype(np.uint8) # round, dtype 변환
    return dst

if __name__ == "__main__":
    VIDEO_INPUT_FILE_PATH = 'jin.avi'  # 동영상 경로
    VIDEO_OUTPUT_FILE_PATH = './gray_recorded.avi' # MacOS의 경우, './gray_recorded.mp4'

    color_frames = get_video_frames(VIDEO_INPUT_FILE_PATH)                              # 비디오의 모든 프레임 가져오기
    gray_frames = [calc_bgr2gray(color_frame) for color_frame in tqdm(color_frames)]    # 비디오의 모든 프레임을 회색으로 변환

    # 비디오 설정 변수들 정의
    fps = 30.0                                          # FPS, 초당 프레임 수
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')            # 인코딩 포맷 문자. MacOS의 경우, 인자에 *'mp4v' 를 대입!
    frame_h, frame_w = color_frames[0].shape[:2]

    video_out = cv2.VideoWriter(                        # 영상 저장을 위한 객체 생성
        VIDEO_OUTPUT_FILE_PATH,
        fourcc, fps, (frame_w, frame_h),
        isColor=False
    )

    # 영상 저장 및 시각화
    for color_frame, gray_frame in zip(color_frames, gray_frames):
        cv2.namedWindow('Gray', flags=cv2.WINDOW_NORMAL)
        cv2.imshow('Gray', gray_frame)

        cv2.namedWindow('Color', flags=cv2.WINDOW_NORMAL)
        cv2.imshow('Color', color_frame)

        video_out.write(gray_frame)
        if cv2.waitKey(int(1000 / fps)) != -1:  # 지연시간 = 1000/fps
            break

    video_out.release()     # 저장 객체 소멸

    cv2.destroyAllWindows() # 캡쳐 객체 소멸
