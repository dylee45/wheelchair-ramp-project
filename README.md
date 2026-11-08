# YOLO 기반 휠체어 통행 가능 경사로 판별 시스템

## 1. 프로젝트 소개

본 프로젝트는 YOLO Object Detection과 Segmentation 기술을 활용하여 사진 속 경사로 및 주변 환경을 분석하고, 휠체어 통행 가능 여부를 지도에 표시하는 시스템을 개발하는 것을 목표로 한다.

사용자가 건물 입구나 통행로의 사진을 업로드하면 경사로를 탐지하고, 보행 영역과 장애물의 관계를 분석하여 통행 가능 여부를 판단하는 기능을 구현할 예정이다.

## 2. 주요 기능

- **Object Detection:** 사진 속 경사로 및 장애물 탐지
- **Segmentation:** 경사로와 보행 가능 영역 분할
- **통행 조건 분석:** 장애물과 보행 영역의 관계 및 통행 조건 분석
- **사용자 인터페이스:** 사진 업로드 및 분석 결과 확인
- **지도 시각화:** 위치별 통행 가능 여부 표시

※ 실제 경사각과 폭의 정확한 측정에는 별도 측정 정보가 필요할 수 있으며, 초기 구현에서는 측정값 입력 또는 판별 보류 기능을 고려한다.

## 3. 사용 기술

| 구분 | 기술 |
|---|---|
| 개발 언어 | Python |
| 개발 환경 | VS Code |
| 객체 탐지 | YOLO Object Detection |
| 영역 분할 | YOLO Segmentation |
| 데이터 관리 | Roboflow |
| 이미지 처리 | OpenCV |
| 사용자 인터페이스 | Streamlit |
| 지도 시각화 | Folium |
| 버전 관리 | Git, GitHub, SourceTree |

## 4. 프로젝트 구조

```text
wheelchair-ramp-project/
├── dataset/
├── models/
├── results/
├── src/
│   ├── __init__.py
│   ├── detection.py
│   ├── segmentation.py
│   ├── analysis.py
│   └── app.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 5. 프로그램 동작 과정

1. 사용자가 사진 업로드
2. Object Detection으로 경사로 및 장애물 탐지
3. Segmentation으로 경사로와 보행 영역 구분
4. 통행 조건 분석 및 판별
5. 통행 가능 / 불가 / 판별 보류 결과 출력
6. 지도에 결과 표시

## 6. GitHub 브랜치 운영 계획

| 브랜치 | 개발 내용 |
|---|---|
| main | 검증된 코드 관리 |
| feat/yolo | Object Detection 개발 |
| feat/segmentation | Segmentation 개발 |
| feat/analysis | 통행 조건 분석 |
| feat/ui-map | UI 및 지도 연동 |

각 기능은 별도 브랜치에서 개발하고 Pull Request를 통해 main에 병합한다.

## 7. 개발 진행 상황

- [x] GitHub 저장소 생성 및 SourceTree 연결
- [ ] 프로젝트 폴더 구조 구성
- [ ] Python 가상환경 구축
- [ ] YOLO Object Detection 학습
- [ ] YOLO Segmentation 학습
- [ ] 통행 조건 분석 기능 구현
- [ ] UI 및 지도 연동
- [ ] 전체 기능 통합 및 테스트

## 8. 기대 결과

사용자가 사진을 업로드하면 경사로와 주변 통행 환경을 분석하고, 해당 위치의 통행 가능 여부를 지도에 표시하는 프로그램을 구현한다.

이를 통해 휠체어 이용자가 건물 접근 환경에 대한 정보를 쉽게 확인할 수 있도록 지원한다.
