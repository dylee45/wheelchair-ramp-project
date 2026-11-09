from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from PIL import Image
from ultralytics import YOLO

from src.analysis import analyze_accessibility
from src.map_view import show_ramp_map
from src.history import (
    save_history,
    load_history,
    delete_history,
    get_latest_records,
)


DETECTION_MODEL = (
    ROOT / "runs" / "detect" / "ramp_test"
    / "weights" / "best.pt"
)

SEGMENTATION_MODEL = (
    ROOT / "runs" / "segment" / "ramp_seg_test"
    / "weights" / "best.pt"
)


@st.cache_resource
def load_models():
    if not DETECTION_MODEL.exists():
        raise FileNotFoundError(DETECTION_MODEL)

    if not SEGMENTATION_MODEL.exists():
        raise FileNotFoundError(SEGMENTATION_MODEL)

    return (
        YOLO(str(DETECTION_MODEL)),
        YOLO(str(SEGMENTATION_MODEL))
    )


def has_class(result, names, targets):
    if result.boxes is None:
        return False

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = names[class_id].strip().lower()

        if class_name in targets:
            return True

    return False


st.set_page_config(
    page_title="휠체어 경사로 분석",
    page_icon="♿",
    layout="wide"
)

st.title("♿ 휠체어 경사로 통행 분석")

st.write(
    "사진을 업로드하면 YOLO로 경사로를 탐지하고, "
    "입력한 조건을 바탕으로 분석 결과를 지도에 표시합니다."
)

st.caption(
    "현재는 시범 프로그램입니다. 폭, 경사각, 장애물 여부는 "
    "사용자가 입력하며 실제 통행 안전성을 보장하지 않습니다."
)


# 1. 사진 업로드
st.subheader("1. 경사로 사진 업로드")

uploaded_file = st.file_uploader(
    "사진을 선택하세요.",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="업로드한 사진",
        width="stretch"
    )

    # 2. 측정 정보
    st.subheader("2. 경사로 측정 정보")

    width_text = st.text_input(
        "경사로 폭 (cm)",
        placeholder="예: 130"
    )

    slope_text = st.text_input(
        "경사각 (도)",
        placeholder="예: 3.5"
    )

    # 3. 장애물 정보
    st.subheader("3. 장애물 여부")

    obstacle_choice = st.radio(
        "경사로의 통행 영역에 장애물이 있나요?",
        options=[
            "확인하지 못함",
            "장애물 없음",
            "장애물 있음",
        ],
        horizontal=True
    )

    if obstacle_choice == "장애물 있음":
        obstacle_detected = True
    elif obstacle_choice == "장애물 없음":
        obstacle_detected = False
    else:
        obstacle_detected = None

    st.caption(
        "현재 장애물 여부는 자동 탐지가 아니라 "
        "사용자 입력으로 판단합니다."
    )

    # 4. 위치 정보
    st.subheader("4. 경사로 위치 정보")

    latitude = st.number_input(
        "위도",
        min_value=-90.0,
        max_value=90.0,
        value=36.4800,
        format="%.6f"
    )

    longitude = st.number_input(
        "경도",
        min_value=-180.0,
        max_value=180.0,
        value=127.2890,
        format="%.6f"
    )

    st.caption(
        "기본 좌표는 테스트용 예시값입니다. "
        "실제 경사로 위치는 직접 확인해야 합니다."
    )

    # 5. 분석 실행
    if st.button("경사로 분석 시작"):
        try:
            width_cm = (
                float(width_text)
                if width_text.strip()
                else None
            )

            slope_deg = (
                float(slope_text)
                if slope_text.strip()
                else None
            )

            if width_cm is not None and width_cm <= 0:
                st.error("경사로 폭은 0보다 커야 합니다.")
                st.stop()

            if slope_deg is not None and not (
                0 <= slope_deg < 90
            ):
                st.error(
                    "경사각은 0도 이상 90도 미만이어야 합니다."
                )
                st.stop()

            with st.spinner("YOLO 분석 중..."):
                detection_model, segmentation_model = (
                    load_models()
                )

                detection_result = detection_model.predict(
                    source=image,
                    conf=0.25,
                    verbose=False
                )[0]

                segmentation_result = (
                    segmentation_model.predict(
                        source=image,
                        conf=0.25,
                        verbose=False
                    )[0]
                )

                ramp_detected = has_class(
                    detection_result,
                    detection_model.names,
                    {"ramp", "ramps"}
                )

                ramp_segmented = (
                    segmentation_result.masks is not None
                    and has_class(
                        segmentation_result,
                        segmentation_model.names,
                        {"ramp", "ramps"}
                    )
                )

                stairs_detected = has_class(
                    segmentation_result,
                    segmentation_model.names,
                    {"stair", "stairs"}
                )

                analysis_result = analyze_accessibility(
                    ramp_detected=ramp_detected,
                    ramp_segmented=ramp_segmented,
                    width_cm=width_cm,
                    slope_deg=slope_deg,
                    obstacle_detected=obstacle_detected
                )

            save_history(
                latitude,
                longitude,
                analysis_result.status,
                analysis_result.reasons
            )

            st.session_state["last_result"] = {
                "detection_image": (
                    detection_result.plot()[:, :, ::-1].copy()
                ),
                "segmentation_image": (
                    segmentation_result.plot()[:, :, ::-1].copy()
                ),
                "ramp_detected": ramp_detected,
                "ramp_segmented": ramp_segmented,
                "stairs_detected": stairs_detected,
                "status": analysis_result.status,
                "reasons": analysis_result.reasons,
                "obstacle_choice": obstacle_choice,
            }

            st.success("분석 결과가 저장되었습니다.")

        except ValueError:
            st.error(
                "폭과 경사각에는 숫자만 입력하세요."
            )
        except FileNotFoundError as error:
            st.error(f"모델 파일 오류: {error}")
        except Exception as error:
            st.error(f"분석 오류: {error}")


# 6. 최근 분석 결과
if "last_result" in st.session_state:
    result = st.session_state["last_result"]

    st.subheader("5. YOLO 분석 결과")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Object Detection**")

        st.image(
            result["detection_image"],
            channels="RGB",
            width="stretch"
        )

        st.write(
            f"경사로 탐지: {result['ramp_detected']}"
        )

    with col2:
        st.write("**Segmentation**")

        st.image(
            result["segmentation_image"],
            channels="RGB",
            width="stretch"
        )

        st.write(
            f"경사로 분할: {result['ramp_segmented']}"
        )

        st.write(
            f"계단 탐지: {result['stairs_detected']}"
        )

    st.subheader("6. 최종 분석 결과")

    if result["status"] == "조건 충족":
        st.success("✅ 입력된 테스트 조건 충족")
    elif result["status"] == "통행 불가":
        st.error("❌ 설정 조건 미충족")
    else:
        st.warning("❓ 판단 보류")

    st.write(
        f"장애물 입력: {result['obstacle_choice']}"
    )

    for reason in result["reasons"]:
        st.write(f"- {reason}")

    st.caption(
        "본 결과는 시범 분석이며 실제 휠체어 "
        "통행 안전성을 보장하지 않습니다."
    )


# 7. 저장된 경사로 지도
st.subheader("7. 저장된 경사로 지도")

records = load_history()
latest_records = get_latest_records(records)

if latest_records:
    show_ramp_map(latest_records)

    st.write(
        f"전체 분석 기록: {len(records)}개 / "
        f"지도에 표시된 위치: {len(latest_records)}개"
    )
else:
    st.info("저장된 경사로 기록이 없습니다.")


# 8. 분석 기록 관리
st.subheader("8. 분석 기록 관리")

if records:
    st.dataframe(
        records,
        width="stretch",
        hide_index=True
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
        options=list(record_options.keys())
    )

    if st.button("선택한 기록 삭제"):
        record_id = record_options[selected]

        if delete_history(record_id):
            st.success("선택한 기록을 삭제했습니다.")
            st.rerun()
        else:
            st.error("삭제할 기록을 찾지 못했습니다.")
else:
    st.info("삭제할 분석 기록이 없습니다.")
