
from pathlib import Path
import argparse

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from src.obstacle_detection import detect_obstacles


ROOT = Path(__file__).resolve().parent.parent

SEGMENTATION_MODEL = (
    ROOT / "runs" / "segment"
    / "ramp_seg_test" / "weights" / "best.pt"
)

OVERLAP_THRESHOLD = 0.10


def get_ramp_mask(segmentation_result, image_shape):
    """
    Segmentation 결과에서 경사로 마스크를 추출한다.
    여러 경사로가 탐지되면 하나의 마스크로 합친다.
    """
    height, width = image_shape[:2]

    ramp_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    if segmentation_result.masks is None:
        return None

    if segmentation_result.boxes is None:
        return None

    masks = segmentation_result.masks.data.cpu().numpy()

    found = False

    for index, box in enumerate(segmentation_result.boxes):
        class_id = int(box.cls[0])
        class_name = segmentation_result.names[
            class_id
        ].strip().lower()

        if class_name not in {"ramp", "ramps"}:
            continue

        if index >= len(masks):
            continue

        mask = masks[index]

        mask = cv2.resize(
            mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )

        ramp_mask = np.maximum(
            ramp_mask,
            (mask > 0.5).astype(np.uint8)
        )

        found = True

    if not found:
        return None

    return ramp_mask


def analyze_obstacle_overlap(ramp_mask, obstacles):
    """
    장애물 Bounding Box와 경사로 마스크의
    겹침 비율을 계산한다.

    겹침 비율:
    장애물 박스 내부 픽셀 중
    경사로 마스크와 겹치는 픽셀 비율
    """
    if ramp_mask is None:
        return "판단 불가", []

    height, width = ramp_mask.shape

    results = []
    overlap_detected = False

    for obstacle in obstacles:
        x1, y1, x2, y2 = obstacle["bbox"]

        x1 = max(0, min(width, int(x1)))
        y1 = max(0, min(height, int(y1)))
        x2 = max(0, min(width, int(np.ceil(x2))))
        y2 = max(0, min(height, int(np.ceil(y2))))

        if x2 <= x1 or y2 <= y1:
            continue

        obstacle_area = (x2 - x1) * (y2 - y1)

        overlap_area = int(
            np.count_nonzero(
                ramp_mask[y1:y2, x1:x2]
            )
        )

        overlap_ratio = overlap_area / obstacle_area

        is_overlapping = (
            overlap_ratio >= OVERLAP_THRESHOLD
        )

        if is_overlapping:
            overlap_detected = True

        results.append({
            "class_name": obstacle["class_name"],
            "confidence": obstacle["confidence"],
            "bbox": obstacle["bbox"],
            "overlap_ratio": overlap_ratio,
            "is_overlapping": is_overlapping,
        })

    if overlap_detected:
        status = "겹침 감지"
    else:
        status = "겹침 미감지"

    return status, results


def create_visualization(image, ramp_mask, results):
    """
    경사로 영역과 장애물 후보를 시각화한다.
    """
    output = np.array(image).copy()

    if ramp_mask is not None:
        overlay = output.copy()
        overlay[ramp_mask > 0] = [0, 200, 0]

        output = cv2.addWeighted(
            output,
            0.65,
            overlay,
            0.35,
            0
        )

    for obstacle in results:
        x1, y1, x2, y2 = map(
            int,
            obstacle["bbox"]
        )

        overlapping = obstacle["is_overlapping"]

        color = (
            (255, 0, 0)
            if overlapping
            else (255, 180, 0)
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        label = (
            f"{obstacle['class_name']} "
            f"{obstacle['overlap_ratio']:.1%}"
        )

        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        default="test_ramp.jpg",
        help="분석할 이미지 경로"
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.is_absolute():
        image_path = ROOT / image_path

    if not image_path.exists():
        print(f"이미지가 없습니다: {image_path}")
        return

    if not SEGMENTATION_MODEL.exists():
        print(
            f"Segmentation 모델이 없습니다: "
            f"{SEGMENTATION_MODEL}"
        )
        return

    image = Image.open(image_path).convert("RGB")

    print("\n1. Segmentation 실행")

    segmentation_model = YOLO(
        str(SEGMENTATION_MODEL)
    )

    segmentation_result = segmentation_model.predict(
        source=image,
        conf=0.25,
        verbose=False
    )[0]

    image_array = np.array(image)

    ramp_mask = get_ramp_mask(
        segmentation_result,
        image_array.shape
    )

    print("2. 장애물 Detection 실행")

    _, obstacles = detect_obstacles(image)

    print("3. 경사로와 장애물 영역 비교")

    status, results = analyze_obstacle_overlap(
        ramp_mask,
        obstacles
    )

    print(f"\n최종 겹침 분석: {status}")

    if ramp_mask is None:
        print("경사로 마스크를 찾지 못했습니다.")

    if not obstacles:
        print("장애물 후보가 탐지되지 않았습니다.")

    for obstacle in results:
        print(
            f"{obstacle['class_name']} | "
            f"신뢰도: {obstacle['confidence']:.2f} | "
            f"겹침 비율: "
            f"{obstacle['overlap_ratio']:.1%}"
        )

    output = create_visualization(
        image,
        ramp_mask,
        results
    )

    output_dir = ROOT / "results" / "obstacle_analysis"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = output_dir / "overlap_result.jpg"

    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(
            output,
            cv2.COLOR_RGB2BGR
        )
    )

    print(f"\n결과 이미지: {output_path}")
    print(
        "주의: 겹침은 이미지상 위치 관계만 나타내며 "
        "실제 통행 차단을 확정하지 않습니다."
    )


if __name__ == "__main__":
    main()
