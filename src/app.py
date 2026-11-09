
from pathlib import Path
import sys
import math

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from PIL import Image
from ultralytics import YOLO

from src.analysis import analyze_accessibility
from src.obstacle_detection import detect_obstacles
from src.obstacle_analysis import (
    get_ramp_mask,
    analyze_obstacle_overlap,
    create_visualization,
)
from src.map_view import show_ramp_map
from src.history import (
    save_history,
    load_history,
    delete_history,
    get_latest_records,
)


DETECTION_MODEL = (
    ROOT / "runs" / "detect"
    / "ramp_test" / "weights" / "best.pt"
)

SEGMENTATION_MODEL = (
    ROOT / "runs" / "segment"
    / "ramp_seg_test" / "weights" / "best.pt"
)

OBSTACLE_MODEL = ROOT / "yolo11n.pt"


@st.cache_resource
def load_models():
    if not DETECTION_MODEL.exists():
        raise FileNotFoundError(DETECTION_MODEL)

    if not SEGMENTATION_MODEL.exists():
        raise FileNotFoundError(SEGMENTATION_MODEL)

    return (
        YOLO(str(DETECTION_MODEL)),
        YOLO(str(SEGMENTATION_MODEL)),
        YOLO(str(OBSTACLE_MODEL)),
    )


def has_class(result, names, targets):
    if result.boxes is None:
        return False

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = str(names[class_id]).strip().lower()

        if class_name in targets:
            return True

    return False


def parse_measurement(value, name):
    if not value.strip():
        return None

    try:
        number = float(value)
    except ValueError:
        raise ValueError(
            f"{name}에는 숫자만 입력하세요."
        )

    if not math.isfinite(number):
        raise ValueError(
            f"{name}에는 유한한 숫자를 입력하세요."
        )

    return number


def make_final_decision(
    ramp_detected,
    ramp_segmented,
    width_cm,
    slope_deg,
    overlap_status,
    obstacles,
):
    """
    자동 장애물 분석을 반영한 시범 판정.

    장애물이 탐지되지 않았다는 이유만으로
    장애물이 없다고 확정하지 않는다.
    """

    # 기존 폭/경사각 및 모델 확인 로직 재사용
    base_result = analyze_accessibility(
        ramp_detected=ramp_detected,
        ramp_segmented=ramp_segmented,
        width_cm=width_cm,
        slope_deg=slope_deg,
        obstacle_detected=None,
    )

    reasons = [
        reason
        for reason in base_result.reasons
        if reason != "장애물 여부를 확인하지 못했습니다."
    ]

    if overlap_status == "겹침 감지":
        reasons.append(
            "YOLO가 경사로 영역과 겹치는 "
            "장애물 후보를 감지했습니다."
        )

    elif overlap_status == "겹침 미감지":
        if obstacles:
            reasons.append(
                "장애물 후보는 탐지되었지만 "
                "설정된 겹침 기준에는 해당하지 않았습니다."
            )
        else:
            reasons.append(
                "YOLO에서 설정된 장애물 후보가 "
                "탐지되지 않았습니다."
            )

        reasons.append(
            "장애물이 없다고 확정할 수는 없습니다."
        )

    else:
        reasons.append(
            "경사로 영역을 확인하지 못해 "
            "장애물 겹침을 판단할 수 없습니다."
        )

    # 명확한 입력 조건 위반은 우선 반영
    if base_result.status == "통행 불가":
        status = "통행 불가"

    # 자동 장애물 탐지만으로 안전을 확정하지 않음
    else:
        status = "판단 보류"

    return status, reasons


st.set_page_config(
    page_title="휠체어 경사로 분석",
    page_icon="♿",
    layout="wide",
)

st.title("♿ 휠체어 경사로 통행 분석")

st.write(
    "YOLO Detection, Segmentation, 장애물 탐지를 "
    "통합하여 경사로의 상태를 분석합니다."
)

st.caption(
    "현재는 시범 프로그램입니다. "
    "자동 분석 결과는 실제 통행 안전성을 보장하지 않습니다."
)


# --------------------------------------------------
# 1. 사진 업로드
# --------------------------------------------------

st.subheader("1. 경사로 사진 업로드")

uploaded_file = st.file_uploader(
    "분석할 사진을 선택하세요.",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="업로드한 사진",
        width="stretch",
    )

    # --------------------------------------------------
    # 2. 측정 정보
    # --------------------------------------------------

    st.subheader("2. 경사로 측정 정보")

    width_text = st.text_input(
        "경사로 폭 (cm)",
        placeholder="예: 130",
    )

    slope_text = st.text_input(
        "경사각 (도)",
        placeholder="예: 3.5",
    )

    st.caption(
        "폭과 경사각은 직접 측정한 값을 입력합니다. "
        "현재 테스트 기준은 폭 120cm 이상, "
        "경사각 5도 이하입니다."
    )

    # --------------------------------------------------
    # 3. 장애물 자동 분석 안내
    # --------------------------------------------------

    st.subheader("3. 장애물 자동 분석")

    st.info(
        "장애물 후보는 YOLO가 자동으로 탐지합니다. "
        "경사로 마스크와 장애물 박스의 겹침도 "
        "자동으로 계산합니다."
    )

    # --------------------------------------------------
    # 4. 위치 정보
    # --------------------------------------------------

    st.subheader("4. 경사로 위치 정보")

    latitude = st.number_input(
        "위도",
        min_value=-90.0,
        max_value=90.0,
        value=36.480000,
        format="%.6f",
    )

    longitude = st.number_input(
        "경도",
        min_value=-180.0,
        max_value=180.0,
        value=127.289000,
        format="%.6f",
    )

    st.caption(
        "기본 좌표는 테스트용 예시값입니다. "
        "실제 경사로 위치를 나타내지 않습니다."
    )

    # --------------------------------------------------
    # 5. 분석 실행
    # --------------------------------------------------

    if st.button(
        "전체 YOLO 분석 시작",
        type="primary",
    ):
        try:
            width_cm = parse_measurement(
                width_text, "경사로 폭"
            )

            slope_deg = parse_measurement(
                slope_text, "경사각"
            )

            if width_cm is not None and width_cm <= 0:
                st.error(
                    "경사로 폭은 0보다 커야 합니다."
                )
                st.stop()

            if slope_deg is not None and not (
                0 <= slope_deg < 90
            ):
                st.error(
                    "경사각은 0도 이상 90도 미만이어야 합니다."
                )
                st.stop()

            with st.spinner("YOLO 자동 분석 중..."):
                (
                    detection_model,
                    segmentation_model,
                    obstacle_model,
                ) = load_models()

                # A. 경사로 Detection
                detection_result = (
                    detection_model.predict(
                        source=image,
                        conf=0.25,
                        verbose=False,
                    )[0]
                )

                ramp_detected = has_class(
                    detection_result,
                    detection_model.names,
                    {"ramp", "ramps"},
                )

                # B. 경사로 Segmentation
                segmentation_result = (
                    segmentation_model.predict(
                        source=image,
                        conf=0.25,
                        verbose=False,
                    )[0]
                )

                ramp_segmented = (
                    segmentation_result.masks is not None
                    and has_class(
                        segmentation_result,
                        segmentation_model.names,
                        {"ramp", "ramps"},
                    )
                )

                stairs_detected = has_class(
                    segmentation_result,
                    segmentation_model.names,
                    {"stair", "stairs"},
                )

                # C. 장애물 Detection
                _, obstacles = detect_obstacles(
                    image,
                    model=obstacle_model,
                )

                # D. 경사로 마스크 추출
                ramp_mask = get_ramp_mask(
                    segmentation_result,
                    (image.height, image.width, 3),
                )

                # E. 겹침 분석
                overlap_status, overlap_results = (
                    analyze_obstacle_overlap(
                        ramp_mask,
                        obstacles,
                    )
                )

                overlap_image = create_visualization(
                    image,
                    ramp_mask,
                    overlap_results,
                )

                # F. 자동 분석 기반 판정
                status, reasons = make_final_decision(
                    ramp_detected=ramp_detected,
                    ramp_segmented=ramp_segmented,
                    width_cm=width_cm,
                    slope_deg=slope_deg,
                    overlap_status=overlap_status,
                    obstacles=obstacles,
                )

            # G. 기록 저장
            save_history(
                latitude,
                longitude,
                status,
                reasons,
            )

            # H. 결과 화면 유지
            st.session_state["last_result"] = {
                "detection_image": (
                    detection_result.plot()[:, :, ::-1].copy()
                ),
                "segmentation_image": (
                    segmentation_result.plot()[:, :, ::-1].copy()
                ),
                "overlap_image": overlap_image,
                "ramp_detected": ramp_detected,
                "ramp_segmented": ramp_segmented,
                "stairs_detected": stairs_detected,
                "obstacles": obstacles,
                "overlap_status": overlap_status,
                "overlap_results": overlap_results,
                "status": status,
                "reasons": reasons,
            }

            st.success(
                "자동 분석 결과가 저장되었습니다."
            )

        except ValueError as error:
            st.error(str(error))

        except FileNotFoundError as error:
            st.error(f"모델 파일 오류: {error}")

        except Exception as error:
            st.error(f"분석 오류: {error}")


# --------------------------------------------------
# 6. YOLO 분석 결과
# --------------------------------------------------

if "last_result" in st.session_state:
    result = st.session_state["last_result"]

    st.subheader("5. YOLO 분석 결과")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Object Detection**")

        st.image(
            result["detection_image"],
            channels="RGB",
            width="stretch",
        )

        st.write(
            f"경사로 탐지: "
            f"{result['ramp_detected']}"
        )

    with col2:
        st.write("**Segmentation**")

        st.image(
            result["segmentation_image"],
            channels="RGB",
            width="stretch",
        )

        st.write(
            f"경사로 분할: "
            f"{result['ramp_segmented']}"
        )

        st.write(
            f"계단 탐지: "
            f"{result['stairs_detected']}"
        )

    # --------------------------------------------------
    # 7. 장애물 분석 결과
    # --------------------------------------------------

    st.subheader("6. 장애물 자동 분석 결과")

    st.image(
        result["overlap_image"],
        channels="RGB",
        caption="경사로 및 장애물 겹침 분석",
        width="stretch",
    )

    st.write(
        f"장애물 후보: "
        f"{len(result['obstacles'])}개"
    )

    st.write(
        f"겹침 분석: "
        f"**{result['overlap_status']}**"
    )

    if result["overlap_results"]:
        for obstacle in result["overlap_results"]:
            st.write(
                f"- {obstacle['class_name']} "
                f"| 신뢰도 "
                f"{obstacle['confidence']:.2f} "
                f"| 겹침 "
                f"{obstacle['overlap_ratio']:.1%}"
            )
    else:
        st.info(
            "표시할 장애물 겹침 상세 결과가 없습니다."
        )

    # --------------------------------------------------
    # 8. 최종 판정
    # --------------------------------------------------

    st.subheader("7. 최종 분석 결과")

    if result["status"] == "통행 불가":
        st.error(
            "❌ 입력된 테스트 조건 미충족"
        )

    elif result["status"] == "조건 충족":
        st.success(
            "✅ 입력된 테스트 조건 충족"
        )

    else:
        st.warning(
            "❓ 판단 보류 — 추가 확인 필요"
        )

    for reason in result["reasons"]:
        st.write(f"- {reason}")

    st.caption(
        "장애물 자동 탐지 결과만으로 실제 통행 가능 "
        "또는 불가능을 확정하지 않습니다."
    )


# --------------------------------------------------
# 9. 지도
# --------------------------------------------------

st.subheader("8. 저장된 경사로 지도")

records = load_history()
latest_records = get_latest_records(records)

if latest_records:
    show_ramp_map(records)

    st.write(
        f"전체 분석 기록: {len(records)}개 / "
        f"지도에 표시된 위치: "
        f"{len(latest_records)}개"
    )
else:
    st.info(
        "저장된 경사로 기록이 없습니다."
    )


# --------------------------------------------------
# 10. 기록 관리
# --------------------------------------------------

st.subheader("9. 분석 기록 관리")

if records:
    st.dataframe(
        records,
        width="stretch",
        hide_index=True,
    )

    record_options = {
        (
            f"{record['date']} | "
            f"{record['latitude']}, "
            f"{record['longitude']} | "
            f"{record['status']} | "
            f"{record['id'][:8]}"
        ): record["id"]
        for record in records
    }

    selected = st.selectbox(
        "삭제할 분석 기록을 선택하세요.",
        options=list(record_options.keys()),
    )

    if st.button("선택한 기록 삭제"):
        record_id = record_options[selected]

        if delete_history(record_id):
            st.success("기록을 삭제했습니다.")
            st.rerun()
        else:
            st.error(
                "삭제할 기록을 찾지 못했습니다."
            )
else:
    st.info(
        "삭제할 분석 기록이 없습니다."
    )
