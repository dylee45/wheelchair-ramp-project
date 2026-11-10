
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
    "사진을 업로드하면 YOLO가 경사로, "
    "계단 및 장애물 후보를 자동 분석합니다."
)

st.caption(
    "V는 시범 분석 조건 충족, "
    "X는 장애물 겹침 의심, "
    "?는 판단 보류를 의미합니다. "
    "실제 통행 안전성을 보증하지 않습니다."
)


# ------------------------------------
# 1. 사진 업로드
# ------------------------------------

st.subheader("1. 사진 업로드")

uploaded_file = st.file_uploader(
    "분석할 사진을 선택하세요.",
    type=["jpg", "jpeg", "png"],
)

image = None
image_id = None

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
        "기본 좌표는 테스트용 예시이며 "
        "실제 경사로 위치가 아닙니다."
    )

    # ------------------------------------
    # 3. YOLO 자동 분석
    # ------------------------------------

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

                stairs_detected = has_class(
                    segmentation_result,
                    {"stair", "stairs"},
                )

                ramp_mask = get_ramp_mask(
                    segmentation_result,
                    (image.height, image.width, 3),
                )

                ramp_segmented = (
                    ramp_mask is not None
                )

                _, obstacles = detect_obstacles(
                    image,
                    model=obstacle_model,
                )

                (
                    overlap_status,
                    overlap_results,
                ) = analyze_obstacle_overlap(
                    ramp_mask,
                    obstacles,
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

            st.success(
                "분석 결과가 지도에 저장되었습니다."
            )

        except FileNotFoundError as error:
            st.error(f"모델 파일 오류: {error}")

        except Exception as error:
            st.error(f"분석 오류: {error}")


# ------------------------------------
# 4. YOLO 결과
# ------------------------------------

result = st.session_state.get("last_result")

if result is not None and image_id == result.get("image_id"):
    st.subheader("3. YOLO 분석 결과")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Object Detection**")
        st.image(
            result["detection_image"],
            channels="RGB",
            width="stretch",
        )
        st.write(
            "경사로 탐지:",
            result["ramp_detected"],
        )

    with col2:
        st.write("**Segmentation**")
        st.image(
            result["segmentation_image"],
            channels="RGB",
            width="stretch",
        )
        st.write(
            "경사로 영역:",
            result["ramp_segmented"],
        )
        st.write(
            "계단 후보:",
            result["stairs_detected"],
        )

    st.subheader("4. 장애물 자동 분석")

    st.image(
        result["visualization"],
        channels="RGB",
        width="stretch",
    )

    st.write(
        f"장애물 후보: "
        f"{len(result['obstacles'])}개"
    )

    st.write(
        f"겹침 분석: "
        f"{result['overlap_status']}"
    )

    for obstacle in result["overlap_results"]:
        st.write(
            f"- {obstacle['class_name']} "
            f"| 신뢰도: "
            f"{obstacle['confidence']:.2f} "
            f"| 겹침 비율: "
            f"{obstacle['overlap_ratio']:.1%}"
        )

    # ------------------------------------
    # 5. 최종 판정
    # ------------------------------------

    st.subheader("5. 최종 판정")

    status = result["status"]

    if status == "조건 충족":
        st.success(
            "✅ V — 시범 분석 조건 충족"
        )

    elif status == "장애물 겹침 의심":
        st.error(
            "❌ X — 장애물 겹침 의심"
        )

    else:
        st.warning(
            "❓ 판단 보류 — 추가 확인 필요"
        )

    for reason in result["reasons"]:
        st.write(f"- {reason}")


# ------------------------------------
# 6. 지도
# ------------------------------------

st.subheader("6. 경사로 지도")

records = load_history()
latest_records = get_latest_records(records)

if latest_records:
    show_ramp_map(records)
else:
    st.info("저장된 경사로 기록이 없습니다.")


# ------------------------------------
# 7. 기록 관리
# ------------------------------------

st.subheader("7. 분석 기록 관리")

if records:
    st.dataframe(
        records,
        hide_index=True,
        width="stretch",
    )

    options = {
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
        "삭제할 분석 기록",
        options=list(options.keys()),
    )

    if st.button("선택한 기록 삭제"):
        if delete_history(options[selected]):
            st.success("기록이 삭제되었습니다.")
            st.rerun()
        else:
            st.error("기록 삭제에 실패했습니다.")

else:
    st.info("분석 기록이 없습니다.")
