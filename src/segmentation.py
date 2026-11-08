from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT / "runs" / "segment" / "ramp_seg_test"
    / "weights" / "best.pt"
)

IMAGE_PATH = ROOT / "test_ramp.jpg"


def segment_ramp():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"학습된 모델이 없습니다: {MODEL_PATH}"
        )

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"테스트 이미지가 없습니다: {IMAGE_PATH}"
        )

    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=str(IMAGE_PATH),
        conf=0.25,
        save=True,
        project=str(ROOT / "results"),
        name="segmentation",
        exist_ok=True,
    )

    for result in results:
        if result.masks is None:
            print("분할된 영역이 없습니다.")
            continue

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            print(
                f"분할 대상: {class_name}, "
                f"신뢰도: {confidence:.2%}"
            )

    print("Segmentation 테스트 완료!")


if __name__ == "__main__":
    segment_ramp()
