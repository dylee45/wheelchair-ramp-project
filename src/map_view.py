import folium
from streamlit_folium import st_folium


def show_ramp_map(latitude, longitude, status):
    # 분석 결과에 따라 마커 색상 설정
    if status == "조건 충족":
        marker_color = "green"
        marker_icon = "ok-sign"
        message = "설정한 테스트 조건 충족"

    elif status == "통행 불가":
        marker_color = "red"
        marker_icon = "remove-sign"
        message = "설정한 통행 조건 미충족"

    else:
        marker_color = "orange"
        marker_icon = "question-sign"
        message = "추가 정보가 필요합니다"

    # 지도 생성
    ramp_map = folium.Map(
        location=[latitude, longitude],
        zoom_start=17
    )

    # 경사로 위치 표시
    folium.Marker(
        location=[latitude, longitude],
        popup=folium.Popup(message, max_width=250),
        tooltip=f"경사로 분석 결과: {status}",
        icon=folium.Icon(
            color=marker_color,
            icon=marker_icon,
            prefix="glyphicon"
        )
    ).add_to(ramp_map)

    # Streamlit 화면에 지도 출력
    st_folium(
        ramp_map,
        width=None,
        height=450,
        returned_objects=[]
    )
