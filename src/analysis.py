
from dataclasses import dataclass


# 프로그램 시연을 위한 임시 기준
# 실제 법적 접근성 기준이 아님
MIN_WIDTH_CM = 120.0
MAX_SLOPE_DEG = 5.0


@dataclass
class AnalysisResult:
    status: str
    reasons: list[str]


def analyze_accessibility(
    ramp_detected: bool,
    ramp_segmented: bool,
    width_cm: float | None,
    slope_deg: float | None,
    obstacle_detected: bool | None = None,
) -> AnalysisResult:

    reasons = []
    blocked = False
    unknown = False

    # 1. Object Detection 결과
    if ramp_detected:
        reasons.append(
            "Object Detection에서 경사로가 탐지되었습니다."
        )
    else:
        reasons.append(
            "Object Detection에서 경사로를 확인하지 못했습니다."
        )
        unknown = True

    # 2. Segmentation 결과
    if ramp_segmented:
        reasons.append(
            "Segmentation에서 경사로 영역이 확인되었습니다."
        )
    else:
        reasons.append(
            "Segmentation에서 경사로 영역을 확인하지 못했습니다."
        )
        unknown = True

    # 3. 경사로 폭 확인
    if width_cm is None:
        reasons.append(
            "경사로 폭 정보가 입력되지 않았습니다."
        )
        unknown = True

    elif width_cm < MIN_WIDTH_CM:
        reasons.append(
            f"입력된 폭 {width_cm:g}cm는 "
            f"테스트 기준 {MIN_WIDTH_CM:g}cm보다 좁습니다."
        )
        blocked = True

    else:
        reasons.append(
            f"입력된 폭 {width_cm:g}cm는 "
            "테스트 기준을 충족합니다."
        )

    # 4. 경사각 확인
    if slope_deg is None:
        reasons.append(
            "경사각 정보가 입력되지 않았습니다."
        )
        unknown = True

    elif slope_deg > MAX_SLOPE_DEG:
        reasons.append(
            f"입력된 경사각 {slope_deg:g}도는 "
            f"테스트 기준 {MAX_SLOPE_DEG:g}도를 초과합니다."
        )
        blocked = True

    else:
        reasons.append(
            f"입력된 경사각 {slope_deg:g}도는 "
            "테스트 기준을 충족합니다."
        )

    # 5. 장애물 여부 확인
    if obstacle_detected is None:
        reasons.append(
            "장애물 여부가 확인되지 않았습니다."
        )
        unknown = True

    elif obstacle_detected:
        reasons.append(
            "사용자가 통행 영역에 장애물이 있다고 입력했습니다."
        )
        blocked = True

    else:
        reasons.append(
            "사용자가 통행 영역에 장애물이 없다고 입력했습니다."
        )

    # 6. 최종 판정
    # 명확히 입력된 조건 위반이 있으면 통행 불가
    if blocked:
        status = "통행 불가"

    # 정보가 부족하거나 모델이 확인하지 못하면 판단 보류
    elif unknown:
        status = "판단 보류"

    # 모든 조건을 충족한 경우
    else:
        status = "조건 충족"
        reasons.append(
            "입력된 모든 시범 테스트 조건을 충족했습니다."
        )

    return AnalysisResult(
        status=status,
        reasons=reasons
    )
