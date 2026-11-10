
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

    reasons = []

    if ramp_detected:
        reasons.append(
            "Object Detection에서 경사로 후보를 탐지했습니다."
        )
    else:
        reasons.append(
            "Object Detection에서 경사로를 탐지하지 못했습니다."
        )

    if ramp_segmented:
        reasons.append(
            "Segmentation에서 경사로 영역을 확인했습니다."
        )
    else:
        reasons.append(
            "Segmentation에서 경사로 영역을 확인하지 못했습니다."
        )

    if stairs_detected:
        reasons.append(
            "계단 후보가 탐지되어 추가 확인이 필요합니다."
        )

    if overlap_status == "겹침 감지":
        reasons.append(
            "경사로 영역과 장애물 후보의 겹침이 감지되었습니다."
        )

    elif overlap_status == "겹침 미감지":
        reasons.append(
            "설정된 기준에서 장애물 겹침이 감지되지 않았습니다."
        )

    else:
        reasons.append(
            "장애물과 경사로 영역의 겹침을 판단할 수 없습니다."
        )

    # 경사로 영역이 확인된 경우에만
    # 겹침 결과를 유효한 것으로 취급한다.
    if (
        ramp_detected
        and ramp_segmented
        and overlap_status == "겹침 감지"
    ):
        status = "장애물 겹침 의심"

    elif (
        ramp_detected
        and ramp_segmented
        and overlap_status == "겹침 미감지"
        and not stairs_detected
    ):
        status = "조건 충족"
        reasons.append(
            "시범 영상 분석 조건을 충족했습니다."
        )

    else:
        status = "판단 보류"
        reasons.append(
            "자동 분석 정보가 부족하거나 "
            "추가 확인이 필요합니다."
        )

    reasons.append(
        "실제 경사로 폭과 경사각은 측정되지 않았으며, "
        "이 결과는 실제 통행 안전성을 보증하지 않습니다."
    )

    return AnalysisResult(
        status=status,
        reasons=reasons,
    )
