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
- [x] 프로젝트 폴더 구조 구성
- [x] Python 가상환경 구축
- [x] YOLO 라이브러리 설치 및 실행 테스트
- [x] 공개 라벨링 데이터셋 확보
- [x] YOLO Object Detection 학습 및 테스트
- [x] YOLO Segmentation 학습
- [ ] 통행 조건 분석 기능 구현
- [ ] UI 및 지도 연동
- [ ] 전체 기능 통합 및 테스트

## 8. 기대 결과

사용자가 사진을 업로드하면 경사로와 주변 통행 환경을 분석하고, 해당 위치의 통행 가능 여부를 지도에 표시하는 프로그램을 구현한다.

이를 통해 휠체어 이용자가 건물 접근 환경에 대한 정보를 쉽게 확인할 수 있도록 지원한다.



## 9. 개발 환경 설정

Python 가상환경 생성:

```powershell
python -m venv .venv
```

가상환경 활성화 (Windows PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

필요 라이브러리 설치:

```powershell
python -m pip install -r requirements.txt
```

YOLO 모델 로딩 테스트:

```powershell
python src/detection.py
```

## 10. 학습 데이터셋

본 프로젝트는 공개된 경사로 라벨링 데이터셋을
활용하여 YOLO Object Detection 모델을 학습한다.

- 데이터셋 출처: 추후 입력
- 데이터셋 이름: 추후 입력
- 데이터셋 라이선스: 추후 확인
- 데이터 형식: YOLO Object Detection
- 탐지 클래스: 데이터셋 확인 후 입력

## 11. Object Detection 실행 방법

학습:

```powershell
python src/train.py
```

추론:

```powershell
python src/detection.py
```

## 통행 조건 분석 기능

### 목적

Object Detection과 Segmentation 결과 및 별도로 확인한 경사로 폭과 경사각 정보를 활용하여 통행 조건을 분석한다.

### 주요 기능

- 경사로 탐지 여부 확인
- 경사로 영역 분할 여부 확인
- 경사로 폭 및 경사각 조건 비교
- 통행 경로 장애물 여부 확인
- 조건 충족 / 통행 불가 / 판단 보류 결과 반환
- 판단 사유 출력

### 실행 방법

`python src/analysis.py`

### 현재 구현 범위

현재는 분석 조건문의 정상 동작을 확인하는 초기 테스트 단계이다. 경사로 폭과 경사각은 직접 입력하며, 임시 기준값은 실제 접근성 기준으로 검증되지 않았다.

추후 YOLO 모델의 실제 추론 결과와 연동하고, 검증된 접근성 기준을 적용할 예정이다.

## YOLO 통합 분석

### 개발 목적

Object Detection, Segmentation 및 통행 조건 분석 기능을 하나의 프로그램으로 통합한다.

### 처리 과정

1. 사용자가 입력한 경사로 이미지 불러오기
2. YOLO Object Detection으로 경사로 위치 탐지
3. YOLO Segmentation으로 경사로 및 계단 영역 분할
4. 탐지 결과를 통행 조건 분석 모듈에 전달
5. 분석 결과와 판단 사유 출력

### 실행 방법

`python main.py`

### 결과 저장 위치

`results/integration/`

### 현재 구현 범위

- 경사로 존재 여부 자동 탐지
- 경사로 및 계단 영역 분할
- 모델 예측 결과를 분석 모듈에 전달
- 통행 조건 판단 결과 출력

경사로 폭과 경사각은 현재 자동 측정되지 않으므로, 측정 정보가 없는 경우 판단 보류로 처리한다.

## Streamlit 사용자 인터페이스

### 개발 목적

사용자가 웹 화면에서 경사로 사진을 업로드하고 YOLO 분석 결과를 확인할 수 있도록 구현한다.

### 주요 기능

- JPG, JPEG, PNG 이미지 업로드
- YOLO Object Detection 결과 시각화
- YOLO Segmentation 결과 시각화
- 경사로 폭 및 경사각 측정값 선택 입력
- 통행 조건 분석 결과와 판단 사유 출력

### 실행 방법

`python -m streamlit run src/app.py`

실행 후 브라우저에서 `http://localhost:8501`에 접속한다.

### 현재 제한 사항

- 경사로 폭과 경사각은 자동 계산되지 않음
- 장애물의 통행 방해 여부는 아직 판단하지 않음
- 판정 기준은 초기 테스트용 임시값이며 실제 접근성 기준 검증이 필요함
- 지도 표시 기능은 추후 구현 예정