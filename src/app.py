
from pathlib import Path
import sys

# 프로젝트 경로 설정
ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from PIL import Image
from ultralytics import YOLO

from src.analysis import analyze_accessibility
from src.map_view import show_ramp_map
from src.history import save_history, load_history


# 학습된 모델 경로
DETECTION_MODEL = (
    ROOT / "runs" / "detect" / "ramp_test"
    / "weights" / "best.pt"
)

SEGMENTATION_MODEL = (
    ROOT / "runs" / "segment" / "ramp_seg_test"
    / "weights" / "best.pt"
)


# 모델 로딩
@st.cache_resource
def load_models():
    if not DETECTION_MODEL.exists():
        raise FileNotFoundError(DETECTION_MODEL)

    if not SEGMENTATION_MODEL.exists():
        raise FileNotFoundError(SEGMENTATION_MODEL)

    detection_model = YOLO(str(DETECTION_MODEL))
    segmentation_model = YOLO(str(SEGMENTATION_MODEL))

    return detection_model, segmentation_model


# 특정 클래스 탐지 여부 확인
def has_class(result, names, target_names):
    if result.boxes is None:
        return False

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = names[class_id].strip().lower()

        if class_name in target_names:
            return True

    return False


# Streamlit 기본 설정
st.set_page_config(
    page_title="휠체어 경사로 분석",
    page_icon="♿",
    layout="wide"
)

st.title("♿ 휠체어 경사로 통행 분석")

st.write(
    "경사로 사진을 업로드하면 YOLO Detection과 "
    "Segmentation으로 분석하고 지도에 표시합니다."
)


# 1. 이미지 업로드
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

    # 2. 경사로 정보 입력
    st.subheader("2. 경사로 측정 정보")

    width_text = st.text_input(
        "경사로 폭 (cm)",
        placeholder="예: 130"
    )

    slope_text = st.text_input(
        "경사각 (도)",
        placeholder="예: 3.5"
    )

    # 3. 위치 입력
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

    st.caption(
        "위도와 경도 기본값은 테스트용 예시 좌표입니다."
    )

    # 4. 분석 실행
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

                # Object Detection 실행
                detection_result = detection_model.predict(
                    source=image,
                    conf=0.25,
                    verbose=False
                )[0]

                ramp_detected = has_class(
                    detection_result,
                    detection_model.names,
                    {"ramp", "ramps"}
                )

                # Segmentation 실행
                segmentation_result = (
                    segmentation_model.predict(
                        source=image,
                        conf=0.25,
                        verbose=False
                    )[0]
                )

                ramp_segmented = (
                    segmentation_result.masks is not None
                    and has_class(
                        segmentation_result,
                        segmentation_model.names,
                        {"ramp", "ramps"}
                    )
                )

                stairs_detected = (
                    segmentation_result.masks is not None
                    and has_class(
                        segmentation_result,
                        segmentation_model.names,
                        {"stair", "stairs"}
                    )
                )

                # 통행 조건 분석
                analysis_result = analyze_accessibility(
                    ramp_detected=ramp_detected,
                    ramp_segmented=ramp_segmented,
                    width_cm=width_cm,
                    slope_deg=slope_deg,
                    obstacle_detected=None
                )

            # 분석 결과 저장
            save_history(
                latitude=latitude,
                longitude=longitude,
                status=analysis_result.status,
                reasons=analysis_result.reasons
            )

            # 결과를 세션에 보관
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
            }

            st.success("분석 결과가 저장되었습니다.")

        except ValueError:
            st.error(
                "폭과 경사각에는 숫자만 입력해 주세요."
            )

        except FileNotFoundError as error:
            st.error(f"파일을 찾을 수 없습니다: {error}")

        except Exception as error:
            st.error(f"오류가 발생했습니다: {error}")


# 5. 최근 YOLO 분석 결과
if "last_result" in st.session_state:

    result = st.session_state["last_result"]

    st.subheader("4. YOLO 분석 결과")

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

    # 6. 최종 분석 결과
    st.subheader("5. 최종 분석 결과")

    if result["status"] == "조건 충족":
        st.success("✅ 입력된 테스트 조건 충족")

    elif result["status"] == "통행 불가":
        st.error("❌ 설정 조건 미충족")

    else:
        st.warning("❓ 판단 보류")

    for reason in result["reasons"]:
        st.write(f"- {reason}")

    st.caption(
        "본 결과는 시범 분석이며 실제 휠체어 "
        "통행 안전성을 보장하지 않습니다."
    )


# 7. 저장된 경사로 지도
st.subheader("6. 저장된 경사로 지도")

records = load_history()

if records:
    show_ramp_map(records)

    st.write(
        f"저장된 분석 기록: {len(records)}개"
    )

else:
    st.info("아직 저장된 경사로 기록이 없습니다.")


# 8. 저장된 분석 기록 목록
st.subheader("7. 분석 기록 목록")

if records:
    st.dataframe(
        records,
        width="stretch",
        hide_index=True
    )
else:
    st.info("표시할 분석 기록이 없습니다.")
