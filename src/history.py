
import csv
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HISTORY_FILE = ROOT / "results" / "ramp_history.csv"

FIELDS = [
    "id",
    "date",
    "latitude",
    "longitude",
    "status",
    "reason",
]


def save_history(latitude, longitude, status, reasons):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 기존 기록을 읽고 새 기록을 추가한다.
    records = load_history()

    records.append({
        "id": str(uuid.uuid4()),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "latitude": str(latitude),
        "longitude": str(longitude),
        "status": status,
        "reason": " / ".join(reasons),
    })

    _write_history(records)


def load_history():
    if not HISTORY_FILE.exists():
        return []

    with open(
        HISTORY_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.DictReader(file)
        records = list(reader)

    # 기존 CSV에 id가 없더라도 처리한다.
    # 동일한 옛 기록에는 항상 동일한 임시 ID를 부여한다.
    for index, record in enumerate(records):
        if not record.get("id"):
            key = (
                f"{index}|{record.get('date', '')}|"
                f"{record.get('latitude', '')}|"
                f"{record.get('longitude', '')}|"
                f"{record.get('status', '')}|"
                f"{record.get('reason', '')}"
            )
            record["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, key))

    return records


def delete_history(record_id):
    records = load_history()

    remaining = [
        record
        for record in records
        if record["id"] != record_id
    ]

    if len(remaining) == len(records):
        return False

    _write_history(remaining)
    return True


def get_latest_records(records):
    latest = {}

    for record in records:
        # 같은 좌표는 같은 위치로 취급
        key = (
            float(record["latitude"]),
            float(record["longitude"]),
        )

        # 기록 날짜를 기준으로 최신 항목 선택
        previous = latest.get(key)

        if (
            previous is None
            or record["date"] >= previous["date"]
        ):
            latest[key] = record

    return list(latest.values())


def _write_history(records):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(
        HISTORY_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDS,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(records)
