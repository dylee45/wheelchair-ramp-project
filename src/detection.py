from ultralytics import YOLO


def test_yolo():
    # 사전 학습된 YOLO 모델 불러오기
    model = YOLO("yolo11n.pt")

    # 모델 정보 출력
    model.info()

    print("YOLO 모델 로딩 성공!")


if __name__ == "__main__":
    test_yolo()
