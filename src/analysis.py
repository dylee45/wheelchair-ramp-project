from dataclasses import dataclass


@dataclass
class AnalysisResult:
    status: str
    reasons: list[str]


def analyze_accessibility(
    ramp_detected: bool,
    ramp_segmented: bool,
    overlap_status: str,
    stairs_detected: bool = False,
) -> AnalysisResult:
    """
    사진 기반 자동 분석 결과를 종합한다.

    주의:
    - 실제 폭(cm)과 경사각(도)은 측정하지 않는다.
    - 장애물 겹침은 통행 차단의 확정 증거가 아니다.
    - 실제 통행 가능 여부는 별도 검증이 필요하다.
    """

    reasons = []

    if ramp_detected:
        reasons.append(
            "Object Detection에서 경사로 후보를 탐지했습니다."
        )
    else:
        reasons.append(
            "Object Detection에서 경사로를 확인하지 못했습니다."
        )

    if ramp_segmented:
        reasons.append(
            "Segmentation에서 경사로 영역을 추출했습니다."
        )
    else:
        reasons.append(
            "Segmentation에서 경사로 영역을 확인하지 못했습니다."
        )

    if stairs_detected:
        reasons.append(
            "계단 후보가 탐지되었습니다. "
            "경사로와의 위치 관계를 추가로 확인해야 합니다."
        )

    if overlap_status == "겹침 감지":
        reasons.append(
            "경사로 영역과 장애물 후보의 겹침이 감지되었습니다."
        )
        reasons.append(
            "실제 통행을 방해하는지는 추가 확인이 필요합니다."
        )

    elif overlap_status == "겹침 미감지":
        reasons.append(
            "설정된 기준에서 장애물 후보의 겹침이 감지되지 않았습니다."
        )
        reasons.append(
            "탐지되지 않은 장애물이 존재할 수 있습니다."
        )

    else:
        reasons.append(
            "경사로와 장애물의 겹침을 판단할 수 없습니다."
        )

    reasons.append(
        "사진만으로 실제 경사로 폭과 경사각을 "
        "검증하지 못했습니다."
    )

    return AnalysisResult(
        status="판단 보류",
        reasons=reasons,
    )
