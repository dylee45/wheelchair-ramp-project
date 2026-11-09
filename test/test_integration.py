
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.analysis import analyze_accessibility
from src.obstacle_analysis import analyze_obstacle_overlap
import src.history as history


class TestAccessibility(unittest.TestCase):

    def test_all_manual_conditions_pass(self):
        result = analyze_accessibility(
            ramp_detected=True,
            ramp_segmented=True,
            width_cm=130,
            slope_deg=3,
            obstacle_detected=False,
        )

        self.assertEqual(
            result.status,
            "조건 충족"
        )

    def test_narrow_ramp(self):
        result = analyze_accessibility(
            ramp_detected=True,
            ramp_segmented=True,
            width_cm=90,
            slope_deg=3,
            obstacle_detected=False,
        )

        self.assertEqual(
            result.status,
            "통행 불가"
        )

    def test_steep_ramp(self):
        result = analyze_accessibility(
            ramp_detected=True,
            ramp_segmented=True,
            width_cm=130,
            slope_deg=8,
            obstacle_detected=False,
        )

        self.assertEqual(
            result.status,
            "통행 불가"
        )

    def test_missing_segmentation(self):
        result = analyze_accessibility(
            ramp_detected=True,
            ramp_segmented=False,
            width_cm=130,
            slope_deg=3,
            obstacle_detected=False,
        )

        self.assertEqual(
            result.status,
            "판단 보류"
        )


class TestObstacleOverlap(unittest.TestCase):

    def test_overlap_detected(self):
        ramp_mask = np.zeros(
            (100, 100),
            dtype=np.uint8
        )

        ramp_mask[20:80, 20:80] = 1

        obstacles = [{
            "class_name": "bicycle",
            "confidence": 0.9,
            "bbox": [30, 30, 60, 60],
        }]

        status, results = analyze_obstacle_overlap(
            ramp_mask,
            obstacles
        )

        self.assertEqual(
            status,
            "겹침 감지"
        )

        self.assertAlmostEqual(
            results[0]["overlap_ratio"],
            1.0
        )

    def test_no_overlap(self):
        ramp_mask = np.zeros(
            (100, 100),
            dtype=np.uint8
        )

        ramp_mask[20:80, 20:80] = 1

        obstacles = [{
            "class_name": "chair",
            "confidence": 0.9,
            "bbox": [0, 0, 10, 10],
        }]

        status, results = analyze_obstacle_overlap(
            ramp_mask,
            obstacles
        )

        self.assertEqual(
            status,
            "겹침 미감지"
        )

        self.assertEqual(
            results[0]["overlap_ratio"],
            0.0
        )

    def test_missing_mask(self):
        status, results = analyze_obstacle_overlap(
            None,
            []
        )

        self.assertEqual(
            status,
            "판단 불가"
        )

        self.assertEqual(results, [])


class TestHistory(unittest.TestCase):

    def test_save_load_delete(self):
        # 실제 분석 기록을 변경하지 않도록
        # 임시 CSV 파일을 사용한다.
        with tempfile.TemporaryDirectory() as folder:
            test_file = (
                Path(folder) / "test_history.csv"
            )

            with patch.object(
                history,
                "HISTORY_FILE",
                test_file
            ):
                history.save_history(
                    36.480000,
                    127.289000,
                    "판단 보류",
                    ["테스트 기록"],
                )

                records = history.load_history()

                self.assertEqual(len(records), 1)
                self.assertEqual(
                    records[0]["status"],
                    "판단 보류"
                )

                record_id = records[0]["id"]

                deleted = history.delete_history(
                    record_id
                )

                self.assertTrue(deleted)
                self.assertEqual(
                    history.load_history(),
                    []
                )

    def test_latest_record_per_location(self):
        records = [
            {
                "id": "1",
                "date": "2026-11-09 10:00:00",
                "latitude": "36.480000",
                "longitude": "127.289000",
                "status": "판단 보류",
            },
            {
                "id": "2",
                "date": "2026-11-09 11:00:00",
                "latitude": "36.480000",
                "longitude": "127.289000",
                "status": "통행 불가",
            },
            {
                "id": "3",
                "date": "2026-11-09 12:00:00",
                "latitude": "36.485000",
                "longitude": "127.295000",
                "status": "판단 보류",
            },
        ]

        latest = history.get_latest_records(records)

        self.assertEqual(len(latest), 2)

        same_location = [
            record
            for record in latest
            if float(record["latitude"]) == 36.48
        ]

        self.assertEqual(
            same_location[0]["id"],
            "2"
        )


if __name__ == "__main__":
    unittest.main()
