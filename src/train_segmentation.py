from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

DATA_YAML = (
    ROOT / "dataset" / "ramp-segmentation" / "data.yaml"
)


def train_segmentation():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"데이터셋 파일을 찾을 수 없습니다: {DATA_YAML}"
        )

    model = YOLO("yolo11n-seg.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=10,
        imgsz=640,
        batch=4,
        device="cpu",
        project=str(ROOT / "runs" / "segment"),
        name="ramp_seg_test",
        exist_ok=True,
    )

    print("Segmentation 학습 완료!")


if __name__ == "__main__":
    train_segmentation()
