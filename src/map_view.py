
import folium
from streamlit_folium import st_folium


def show_ramp_map(records):
    if not records:
        return

    first = records[0]

    ramp_map = folium.Map(
        location=[
            float(first["latitude"]),
            float(first["longitude"])
        ],
        zoom_start=15
    )

    for record in records:
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

        folium.Marker(
            location=[
                float(record["latitude"]),
                float(record["longitude"])
            ],
            popup=folium.Popup(
                f"분석 결과: {status}",
                max_width=250
            ),
            tooltip=status,
            icon=folium.Icon(
                color=color,
                icon=icon,
                prefix="glyphicon"
            )
        ).add_to(ramp_map)

    st_folium(
        ramp_map,
        height=450,
        width=None,
        returned_objects=[]
    )
