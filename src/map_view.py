
import folium
from html import escape
from streamlit_folium import st_folium

from src.history import get_latest_records


def show_ramp_map(records):
    latest_records = get_latest_records(records)

    if not latest_records:
        return

    locations = []

    for record in latest_records:
        locations.append([
            float(record["latitude"]),
            float(record["longitude"])
        ])

    ramp_map = folium.Map(
        location=locations[0],
        zoom_start=15
    )

    for record in latest_records:
        latitude = float(record["latitude"])
        longitude = float(record["longitude"])
        status = record["status"]

        if status == "조건 충족":
            color = "green"
            icon = "ok-sign"
        elif status == "통행 불가":
            color = "red"
            icon = "remove-sign"
        else:
            color = "orange"
            icon = "question-sign"

        popup_html = f"""
        <b>분석 날짜:</b>
        {escape(str(record.get("date", "")))}<br>

        <b>분석 결과:</b>
        {escape(str(status))}<br>

        <b>판단 사유:</b>
        {escape(str(record.get("reason", "")))}
        """

        folium.Marker(
            location=[latitude, longitude],
            popup=folium.Popup(
                popup_html,
                max_width=300
            ),
            tooltip=f"분석 결과: {status}",
            icon=folium.Icon(
                color=color,
                icon=icon,
                prefix="glyphicon"
            )
        ).add_to(ramp_map)

    # 서로 다른 좌표가 2개 이상이면 모두 보이도록 확대 범위 조절
    if len(locations) > 1:
        ramp_map.fit_bounds(
            locations,
            padding=(30, 30),
            max_zoom=17
        )

    st_folium(
        ramp_map,
        height=500,
        width=None,
        returned_objects=[],
        key="ramp_history_map"
    )
