from dataclasses import dataclass


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

    # 프로젝트 초기 테스트용 가정값
    # 실제 접근성 기준은 추후 확인하여 적용
    MIN_WIDTH_CM = 120
    MAX_SLOPE_DEG = 5.0

    # 1. 경사로 탐지 여부
    if not ramp_detected:
        reasons.append("경사로가 탐지되지 않았습니다.")
        unknown = True

    # 2. 경사로 영역 분할 여부
    if not ramp_segmented:
        reasons.append("경사로 영역이 확인되지 않았습니다.")
        unknown = True

    # 3. 경사로 폭 분석
    if width_cm is None:
        reasons.append("경사로 폭 정보가 없습니다.")
        unknown = True
    elif width_cm < MIN_WIDTH_CM:
        reasons.append("설정한 최소 폭 조건을 충족하지 않습니다.")
        blocked = True

    # 4. 경사각 분석
    if slope_deg is None:
        reasons.append("경사각 정보가 없습니다.")
        unknown = True
    elif slope_deg > MAX_SLOPE_DEG:
        reasons.append("설정한 최대 경사각 조건을 초과합니다.")
        blocked = True

    # 5. 장애물 여부
    if obstacle_detected is None:
        reasons.append("장애물 여부를 확인하지 못했습니다.")
        unknown = True
    elif obstacle_detected:
        reasons.append("통행을 방해하는 장애물이 확인되었습니다.")
        blocked = True

    # 6. 최종 결과
    if blocked:
        status = "통행 불가"
    elif unknown:
        status = "판단 보류"
    else:
        status = "조건 충족"

    if status == "조건 충족":
        reasons.append("입력된 모든 테스트 조건을 충족합니다.")

    return AnalysisResult(status=status, reasons=reasons)


if __name__ == "__main__":
    result = analyze_accessibility(
        ramp_detected=True,
        ramp_segmented=True,
        width_cm=130,
        slope_deg=3.0,
        obstacle_detected=False,
    )

    print("판단 결과:", result.status)

    for reason in result.reasons:
        print("-", reason)
