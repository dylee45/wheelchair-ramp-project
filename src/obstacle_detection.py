
from pathlib import Path
import argparse

from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "yolo11n.pt"

# 장애물 후보로 확인할 객체
# COCO 데이터셋 클래스 이름 사용
OBSTACLE_CLASSES = {
    "person",
    "bicycle",
    "motorcycle",
    "car",
    "chair",
    "bench",
    "backpack",
    "suitcase",
    "potted plant",
}

CONFIDENCE = 0.25


def detect_obstacles(image, model=None):
    """
    사진에서 장애물 후보 객체를 탐지한다.

    반환값:
        result: YOLO 추론 결과
        obstacles: 탐지된 장애물 후보 목록

    주의:
        장애물 후보가 발견됐다고 해서
        실제 경사로를 막고 있다는 뜻은 아니다.
    """

    if model is None:
        model = YOLO(str(MODEL_PATH))

    result = model.predict(
        source=image,
        conf=CONFIDENCE,
        verbose=False
    )[0]

    obstacles = []

    if result.boxes is None:
        return result, obstacles

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = result.names[class_id].lower()

        if class_name not in OBSTACLE_CLASSES:
            continue

        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        obstacles.append({
            "class_name": class_name,
            "confidence": confidence,
            "bbox": [
                float(x1),
                float(y1),
                float(x2),
                float(y2)
            ]
        })

    return result, obstacles


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        default="test_ramp.jpg",
        help="테스트 이미지 경로"
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.is_absolute():
        image_path = ROOT / image_path

    if not image_path.exists():
        print(f"이미지를 찾을 수 없습니다: {image_path}")
        return

    image = Image.open(image_path).convert("RGB")

    print("장애물 후보 탐지를 시작합니다.")

    result, obstacles = detect_obstacles(image)

    print(f"\n탐지된 장애물 후보: {len(obstacles)}개")

    for index, obstacle in enumerate(obstacles, start=1):
        print(
            f"{index}. {obstacle['class_name']} "
            f"(신뢰도: {obstacle['confidence']:.2f})"
        )

    if not obstacles:
        print("설정된 장애물 후보 클래스가 탐지되지 않았습니다.")

    output_dir = ROOT / "results" / "obstacle"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "obstacle_result.jpg"

    # 장애물 후보만 그리기
    import cv2

    output_image = cv2.cvtColor(
        __import__("numpy").array(image),
        cv2.COLOR_RGB2BGR
    )

    for obstacle in obstacles:
        x1, y1, x2, y2 = map(
            int,
            obstacle["bbox"]
        )

        cv2.rectangle(
            output_image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        label = (
            f"{obstacle['class_name']} "
            f"{obstacle['confidence']:.2f}"
        )

        cv2.putText(
            output_image,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    cv2.imwrite(
        str(output_path),
        output_image
    )

    print(f"\n결과 이미지 저장: {output_path}")
    print(
        "주의: 객체가 탐지되어도 "
        "경사로 통행을 방해하는지는 아직 판단하지 않습니다."
    )


if __name__ == "__main__":
    main()
