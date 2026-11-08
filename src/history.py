import csv
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HISTORY_FILE = ROOT / "results" / "ramp_history.csv"

FIELDS = [
    "date",
    "latitude",
    "longitude",
    "status",
    "reason",
]


def save_history(latitude, longitude, status, reasons):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    file_exists = HISTORY_FILE.exists()

    with open(
        HISTORY_FILE,
        "a",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)

        if not file_exists or HISTORY_FILE.stat().st_size == 0:
            writer.writeheader()

        writer.writerow({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "latitude": latitude,
            "longitude": longitude,
            "status": status,
            "reason": " / ".join(reasons),
        })


def load_history():
    if not HISTORY_FILE.exists():
        return []

    with open(
        HISTORY_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        return list(csv.DictReader(file))
