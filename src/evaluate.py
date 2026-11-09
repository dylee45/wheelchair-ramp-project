
from pathlib import Path
import argparse
import yaml

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "detection": {
        "weights": ROOT / "runs/detect/ramp_test/weights/best.pt",
        "dataset": ROOT / "dataset/ramp-detection",
        "task": "detect",
    },
    "segmentation": {
        "weights": ROOT / "runs/segment/ramp_seg_test/weights/best.pt",
        "dataset": ROOT / "dataset/ramp-segmentation",
        "task": "segment",
    },
}


def find_data_yaml(dataset_dir):
    candidates = [
        dataset_dir / "data.yaml",
        dataset_dir / "dataset.yaml",
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = list(dataset_dir.rglob("data.yaml"))

    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        f"데이터셋 YAML 파일을 확인하세요: {dataset_dir}"
    )


def evaluate(model_type):
    config = MODELS[model_type]

    weights = config["weights"]
    dataset_dir = config["dataset"]

    if not weights.exists():
        raise FileNotFoundError(
            f"학습된 모델이 없습니다: {weights}"
        )

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"데이터셋 폴더가 없습니다: {dataset_dir}"
        )

    data_yaml = find_data_yaml(dataset_dir)

    # 데이터셋 경로가 현재 PC에서 올바르게
    # 해석되도록 평가용 YAML을 생성
    with open(data_yaml, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    data_config["path"] = str(data_yaml.parent.resolve())

    evaluation_yaml = (
        data_yaml.parent / "evaluation_data.yaml"
    )

    with open(
        evaluation_yaml,
        "w",
        encoding="utf-8"
    ) as f:
        yaml.safe_dump(
            data_config,
            f,
            allow_unicode=True,
            sort_keys=False
        )

    # test가 없으면 val 사용
    split = "test" if data_config.get("test") else "val"

    print(f"\n평가 모델: {model_type}")
    print(f"모델 경로: {weights}")
    print(f"데이터셋: {evaluation_yaml}")
    print(f"평가 데이터: {split}\n")

    model = YOLO(str(weights))

    metrics = model.val(
        data=str(evaluation_yaml),
        split=split,
        imgsz=640,
        batch=1,
        device="cpu",
        plots=True,
        project=str(ROOT / "runs/evaluation"),
        name=model_type,
        exist_ok=True,
    )

    if model_type == "detection":
        scores = metrics.box
        print("\n[Object Detection 평가 결과]")

    else:
        scores = metrics.seg
        print("\n[Segmentation 마스크 평가 결과]")

    print(f"Precision: {scores.mp:.4f}")
    print(f"Recall:    {scores.mr:.4f}")
    print(f"mAP50:     {scores.map50:.4f}")
    print(f"mAP50-95:  {scores.map:.4f}")

    print(
        f"\n결과 저장 위치: "
        f"{ROOT / 'runs/evaluation' / model_type}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["detection", "segmentation"],
        required=True
    )

    args = parser.parse_args()

    evaluate(args.model)
