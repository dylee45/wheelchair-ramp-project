from pathlib import Path

from ultralytics import YOLO
from src.analysis import analyze_accessibility


ROOT = Path(__file__).resolve().parent

IMAGE_PATH = ROOT / "test_ramp.jpg"

DETECTION_MODEL = (
    ROOT / "runs" / "detect" / "ramp_test"
    / "weights" / "best.pt"
)

SEGMENTATION_MODEL = (
    ROOT / "runs" / "segment" / "ramp_seg_test"
    / "weights" / "best.pt"
)


def main():
    # 1. 파일 확인
    for path in [
        IMAGE_PATH,
        DETECTION_MODEL,
        SEGMENTATION_MODEL,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"파일을 찾을 수 없습니다: {path}"
            )

    # 2. 모델 불러오기
    detection_model = YOLO(str(DETECTION_MODEL))
    segmentation_model = YOLO(str(SEGMENTATION_MODEL))

    # 3. Object Detection 실행
    detection_results = detection_model.predict(
        source=str(IMAGE_PATH),
        conf=0.25,
        verbose=False,
    )

    ramp_detected = False

    for result in detection_results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = detection_model.names[class_id]

            if class_name.strip().lower() in {"ramp", "ramps"}:
                ramp_detected = True

    # 4. Segmentation 실행
    segmentation_results = segmentation_model.predict(
        source=str(IMAGE_PATH),
        conf=0.25,
        verbose=False,
        save=True,
        project=str(ROOT / "results"),
        name="integration",
        exist_ok=True,
    )

    ramp_segmented = False
    stairs_detected = False

    for result in segmentation_results:
        if result.masks is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = segmentation_model.names[class_id]
            class_name = class_name.strip().lower()

            if class_name in {"ramp", "ramps"}:
                ramp_segmented = True

            elif class_name in {"stair", "stairs"}:
                stairs_detected = True

    # 5. 분석 결과 전달
    # 실제 측정값이 아직 없으므로 None으로 설정
    analysis_result = analyze_accessibility(
        ramp_detected=ramp_detected,
        ramp_segmented=ramp_segmented,
        width_cm=None,
        slope_deg=None,
        obstacle_detected=None,
    )

    # 6. 결과 출력
    print("\n===== YOLO 통합 분석 결과 =====")
    print(f"Detection 경사로 탐지: {ramp_detected}")
    print(f"Segmentation 경사로 분할: {ramp_segmented}")
    print(f"Segmentation 계단 탐지: {stairs_detected}")

    print(f"\n최종 판단: {analysis_result.status}")

    print("\n판단 사유:")
    for reason in analysis_result.reasons:
        print(f"- {reason}")

    print("\n결과 이미지 저장 위치:")
    print(ROOT / "results" / "integration")


if __name__ == "__main__":
    main()
