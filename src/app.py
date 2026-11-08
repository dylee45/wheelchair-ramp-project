from pathlib import Path
import sys
from src.map_view import show_ramp_map

import streamlit as st
from PIL import Image
from ultralytics import YOLO

# 프로젝트 최상위 폴더
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis import analyze_accessibility


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
    detection_model = YOLO(str(DETECTION_MODEL))
    segmentation_model = YOLO(str(SEGMENTATION_MODEL))
    return detection_model, segmentation_model


def has_class(result, names, target_names):
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = names[class_id].strip().lower()

        if class_name in target_names:
            return True

    return False


st.set_page_config(
    page_title="휠체어 경사로 분석",
    page_icon="♿",
    layout="wide",
)

st.title("♿ 휠체어 경사로 통행 분석")
st.write(
    "사진을 업로드하면 YOLO가 경사로와 계단을 분석합니다."
)

uploaded_file = st.file_uploader(
    "경사로 사진을 업로드하세요.",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("1. 입력 이미지")
    st.image(image, width="stretch")

    # 아직 측정값이 없는 경우 비워 둠
    st.subheader("2. 경사로 측정 정보 (선택)")

    width_text = st.text_input(
        "경사로 폭 (cm)",
        placeholder="예: 130",
    )

    slope_text = st.text_input(
        "경사각 (도)",
        placeholder="예: 3.5",
    )


    
    st.subheader("3. 경사로 위치 정보")

    latitude = st.number_input(
        "위도 (Latitude)",
        min_value=-90.0,
        max_value=90.0,
        value=36.4800,
        format="%.6f"
    )

    longitude = st.number_input(
        "경도 (Longitude)",
        min_value=-180.0,
        max_value=180.0,
        value=127.2890,
        format="%.6f"
    )


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

            with st.spinner("YOLO 모델 분석 중..."):
                detection_model, segmentation_model = (
                    load_models()
                )

                detection_result = detection_model.predict(
                    source=image,
                    conf=0.25,
                    verbose=False,
                )[0]

                segmentation_result = (
                    segmentation_model.predict(
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

                ramp_segmented = (
                    segmentation_result.masks is not None
                    and has_class(
                        segmentation_result,
                        segmentation_model.names,
                        {"ramp", "ramps"},
                    )
                )

                stairs_detected = (
                    segmentation_result.masks is not None
                    and has_class(
                        segmentation_result,
                        segmentation_model.names,
                        {"stair", "stairs"},
                    )
                )

                analysis_result = analyze_accessibility(
                    ramp_detected=ramp_detected,
                    ramp_segmented=ramp_segmented,
                    width_cm=width_cm,
                    slope_deg=slope_deg,
                    obstacle_detected=None,
                )

            st.subheader("3. YOLO 분석 결과")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Object Detection**")
                st.image(
                    detection_result.plot()[:, :, ::-1],
                    channels="RGB",
                    width="stretch",
                )
                st.write(
                    f"경사로 탐지: {ramp_detected}"
                )

            with col2:
                st.write("**Segmentation**")
                st.image(
                    segmentation_result.plot()[:, :, ::-1],
                    channels="RGB",
                    width="stretch",
                )
                st.write(
                    f"경사로 분할: {ramp_segmented}"
                )
                st.write(
                    f"계단 탐지: {stairs_detected}"
                )

            st.subheader("4. 최종 분석 결과")

            if analysis_result.status == "통행 불가":
                st.error("❌ 설정 조건 미충족")
            elif analysis_result.status == "조건 충족":
                st.success("✅ 입력된 테스트 조건 충족")
            else:
                st.warning("❓ 판단 보류")

            for reason in analysis_result.reasons:
                st.write(f"- {reason}")
                
                st.subheader("5. 경사로 위치 및 분석 결과")

                show_ramp_map(
                    latitude=latitude,
                    longitude=longitude,
                    status=analysis_result.status
                    )


            st.caption(
                "본 결과는 시범 분석이며 실제 휠체어 "
                "통행 안전성을 보장하지 않습니다."
            )

        except ValueError:
            st.error(
                "폭과 경사각에는 숫자만 입력해 주세요."
            )
        except FileNotFoundError as error:
            st.error(f"모델 파일 오류: {error}")
        except Exception as error:
            st.error(f"분석 중 오류가 발생했습니다: {error}")
