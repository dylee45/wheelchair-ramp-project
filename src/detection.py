from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "runs" / "detect" / "ramp_test" / "weights" / "best.pt"
IMAGE_PATH = ROOT / "test_ramp.jpg"


def detect_ramp():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("학습된 모델 파일이 없습니다.")

    if not IMAGE_PATH.exists():
        raise FileNotFoundError("테스트 이미지가 없습니다.")

    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=str(IMAGE_PATH),
        conf=0.25,
        save=True,
        project=str(ROOT / "results"),
        name="prediction",
        exist_ok=True,
    )

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            print(f"탐지 대상: {class_name}")
            print(f"신뢰도: {confidence:.2%}")

    print("탐지 테스트 완료!")


if __name__ == "__main__":
    detect_ramp()
