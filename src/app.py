from pathlib import Path
import sys
import hashlib

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


def has_class(result, targets):
    if result.boxes is None:
        return False

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = str(
            result.names[class_id]
        ).strip().lower()

        if class_name in targets:
            return True

    return False


st.set_page_config(
    page_title="경사로 자동 분석",
    page_icon="♿",
    layout="wide",
)

st.title("♿ 경사로 자동 분석 시스템")

st.write(
    "사진을 업로드하면 YOLO를 이용하여 "
    "경사로와 장애물 후보를 자동으로 분석합니다."
)

st.warning(
    "현재 프로그램은 시연용입니다. "
    "사진만으로 실제 경사로 폭, 경사각 및 "
    "휠체어 통행 안전성을 검증할 수 없습니다."
)


# ---------------------------------------------
# 1. 사진 업로드
# ---------------------------------------------

st.subheader("1. 사진 업로드")

uploaded_file = st.file_uploader(
    "경사로 사진을 업로드하세요.",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    image_id = hashlib.sha256(
        file_bytes
    ).hexdigest()

    if st.session_state.get("image_id") != image_id:
        st.session_state["image_id"] = image_id
        st.session_state.pop("last_result", None)

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="업로드한 사진",
        width="stretch",
    )

    # -----------------------------------------
    # 2. 지도 위치
    # -----------------------------------------

    st.subheader("2. 위치 정보")

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
        "기본 좌표는 테스트용 예시입니다. "
        "실제 경사로 위치가 아닙니다."
    )

    # -----------------------------------------
    # 3. 자동 분석
    # -----------------------------------------

    if st.button(
        "자동 분석 시작",
        type="primary",
    ):
        try:
            with st.spinner("YOLO 분석 중..."):
                (
                    detection_model,
                    segmentation_model,
                    obstacle_model,
                ) = load_models()

                detection_result = (
                    detection_model.predict(
                        source=image,
                        conf=0.25,
                        verbose=False,
                    )[0]
                )

                segmentation_result = (
                    segmentation_model.predict(
                        source=image,
                        conf=0.25,
                        verbose=False,
                    )[0]
                )

                ramp_detected = has_class(
                    detection_result,
                    {"ramp", "ramps"},
                )

                ramp_mask = get_ramp_mask(
                    segmentation_result,
                    (image.height, image.width, 3),
                )

                ramp_segmented = (
                    ramp_mask is not None
                )

                stairs_detected = has_class(
                    segmentation_result,
                    {"stair", "stairs"},
                )

                _, obstacles = detect_obstacles(
                    image,
                    model=obstacle_model,
                )

                overlap_status, overlap_results = (
                    analyze_obstacle_overlap(
                        ramp_mask,
                        obstacles,
                    )
                )

                visualization = create_visualization(
                    image,
                    ramp_mask,
                    overlap_results,
                )

                analysis = analyze_accessibility(
                    ramp_detected=ramp_detected,
                    ramp_segmented=ramp_segmented,
                    overlap_status=overlap_status,
                    stairs_detected=stairs_detected,
                )

                detection_image = (
                    detection_result.plot()[
                        :, :, ::-1
                    ].copy()
                )

                segmentation_image = (
                    segmentation_result.plot()[
                        :, :, ::-1
                    ].copy()
                )

            save_history(
                latitude,
                longitude,
                analysis.status,
                analysis.reasons,
            )

            st.session_state["last_result"] = {
                "image_id": image_id,
                "detection_image": detection_image,
                "segmentation_image": segmentation_image,
                "visualization": visualization,
                "ramp_detected": ramp_detected,
                "ramp_segmented": ramp_segmented,
                "stairs_detected": stairs_detected,
                "obstacles": obstacles,
                "overlap_status": overlap_status,
                "overlap_results": overlap_results,
                "status": analysis.status,
                "reasons": analysis.reasons,
            }

            st.success("자동 분석이 완료되었습니다.")

        except FileNotFoundError as error:
            st.error(f"모델 파일이 없습니다: {error}")

        except Exception as error:
            st.error(f"분석 오류: {error}")


# ---------------------------------------------
# 4. 분석 결과 출력
# ---------------------------------------------

result = st.session_state.get("last_result")

if result is not None:
    st.subheader("3. YOLO 분석 결과")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Object Detection**")
        st.image(
            result["detection_image"],
            channels="RGB",
            width="stretch",
        )

    with col2:
        st.write("**Segmentation**")
        st.image(
            result["segmentation_image"],
            channels="RGB",
            width="stretch",
        )

    st.write(
        "경사로 Detection:",
        result["ramp_detected"],
    )

    st.write(
        "경사로 Segmentation:",
        result["ramp_segmented"],
    )

    st.write(
        "계단 후보:",
        result["stairs_detected"],
    )

    st.subheader("4. 장애물 분석")

    st.image(
        result["visualization"],
        channels="RGB",
        width="stretch",
    )

    st.write(
        f"장애물 후보 수: "
        f"{len(result['obstacles'])}개"
    )

    st.write(
        f"겹침 결과: "
        f"{result['overlap_status']}"
    )

    for obstacle in result["overlap_results"]:
        st.write(
            f"- {obstacle['class_name']}: "
            f"겹침 비율 "
            f"{obstacle['overlap_ratio']:.1%}"
        )

    st.subheader("5. 최종 판정")

    st.warning(
        f"❓ {result['status']} "
        "— 실제 통행 조건 추가 확인 필요"
    )

    for reason in result["reasons"]:
        st.write(f"- {reason}")


# ---------------------------------------------
# 5. 지도 및 기록
# ---------------------------------------------

st.subheader("6. 경사로 지도")

records = load_history()
latest_records = get_latest_records(records)

if latest_records:
    show_ramp_map(records)
else:
    st.info("저장된 분석 기록이 없습니다.")


st.subheader("7. 분석 기록")

if records:
    st.dataframe(
        records,
        hide_index=True,
        width="stretch",
    )

    options = {
        (
            f"{r['date']} | "
            f"{r['latitude']}, {r['longitude']} "
            f"| {r['id'][:8]}"
        ): r["id"]
        for r in records
    }

    selected = st.selectbox(
        "삭제할 기록",
        list(options.keys()),
    )

    if st.button("기록 삭제"):
        if delete_history(options[selected]):
            st.rerun()
        else:
            st.error("기록 삭제에 실패했습니다.")
else:
    st.info("분석 기록이 없습니다.")
