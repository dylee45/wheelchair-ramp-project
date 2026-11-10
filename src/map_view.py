
import html

import folium
import streamlit as st
from streamlit_folium import st_folium

from src.history import get_latest_records


def get_marker_style(status):
    if status == "조건 충족":
        return {
            "color": "green",
            "symbol": "V",
        }

    if status in ("장애물 겹침 의심", "통행 불가"):
        return {
            "color": "red",
            "symbol": "X",
        }

    return {
        "color": "orange",
        "symbol": "?",
    }


def show_ramp_map(records):
    latest_records = get_latest_records(records)

    if not latest_records:
        st.info("지도에 표시할 분석 기록이 없습니다.")
        return

    valid_records = []

    for record in latest_records:
        try:
            lat = float(record["latitude"])
            lon = float(record["longitude"])

            if not (-90 <= lat <= 90):
                continue

            if not (-180 <= lon <= 180):
                continue

            valid_records.append((lat, lon, record))

        except (ValueError, KeyError, TypeError):
            continue

    if not valid_records:
        st.warning("유효한 위치 정보가 없습니다.")
        return

    # 첫 번째 유효한 좌표를 지도 중심으로 설정
    first_lat, first_lon, _ = valid_records[0]

    ramp_map = folium.Map(
        location=[first_lat, first_lon],
        zoom_start=15,
        tiles="OpenStreetMap",
    )

    for lat, lon, record in valid_records:
        status = record.get("status", "판단 보류")
        style = get_marker_style(status)

        reason = html.escape(
            str(record.get("reason", ""))
        )
        safe_status = html.escape(str(status))
        safe_date = html.escape(
            str(record.get("date", ""))
        )

        popup_html = f"""
        <div style="width:260px">
            <b>분석 결과: {safe_status}</b>
            <hr>
            <b>분석 일시</b><br>
            {safe_date}<br><br>
            <b>판정 사유</b><br>
            {reason}
        </div>
        """

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(
                popup_html,
                max_width=320,
            ),
            tooltip=f"{style['symbol']} | {status}",
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    background:{style['color']};
                    color:white;
                    border:2px solid white;
                    border-radius:50%;
                    width:30px;
                    height:30px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    font-size:18px;
                    font-weight:bold;
                    box-shadow:0 2px 6px #555;
                ">
                    {style['symbol']}
                </div>
                """
            ),
        ).add_to(ramp_map)

    # 여러 마커가 가까이 있는 경우 모두 보이게 조정
    if len(valid_records) > 1:
        coordinates = [
            [lat, lon]
            for lat, lon, _ in valid_records
        ]

        ramp_map.fit_bounds(
            coordinates,
            padding=(30, 30),
            max_zoom=16,
        )

    st_folium(
        ramp_map,
        height=480,
        width=None,
        returned_objects=[],
    )
