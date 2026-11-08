
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "dataset" / "ramp-detection" / "data.yaml"


def train_model():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"데이터셋 파일이 없습니다: {DATA_YAML}")

    model = YOLO("yolo11n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=10,
        imgsz=640,
        batch=4,
        device="cpu",
        project=str(ROOT / "runs" / "detect"),
        name="ramp_test",
        exist_ok=True,
    )

    print("경사로 탐지 모델 학습 완료!")


if __name__ == "__main__":
    train_model()
