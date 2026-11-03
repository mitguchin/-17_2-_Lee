# MNIST 손글씨 숫자 인식 서비스 (Streamlit + ONNX Runtime)

마우스로 그린 숫자(0~9)를 ONNX Model Zoo 의 `mnist-12.onnx` 모델로 실시간 인식하는 웹 서비스입니다.

## 화면 구성
| 영역 | 내용 |
|---|---|
| ① 입력 캔버스 | `streamlit-drawable-canvas` 로 마우스 드로잉 (검은 배경 + 흰 펜) |
| ② 전처리 이미지 | 모델 입력(28×28, 0.0~1.0)으로 변환된 이미지를 확대 표시 |
| ③ 모델 추론 결과 | 0~9 레이블별 확률 막대 차트, 예측 숫자 · 확률 |
| ④ 이미지 저장소 | 저장한 그림과 예측 레이블 · 확률(2순위 포함) 갤러리 |

## 파일 구조
```
.
├── app.py               # Streamlit UI (4개 화면 영역, 세션 상태, 모델 캐싱)
├── model_utils.py       # 모델 다운로드/디스크 캐시, 전처리, 추론
├── requirements.txt     # 고정 버전 의존성
├── Dockerfile           # 컨테이너 이미지 정의 (빌드 시 모델 포함)
├── .dockerignore
├── .streamlit/config.toml
├── deploy.py            # 빌드 → 테스트 → push → 보고서 PDF · 코드 zip 자동 생성
└── report/              # 보고서 템플릿(template.html) + 실행 화면 이미지
```

## 로컬 실행
```bash
pip install -r requirements.txt
streamlit run app.py          # http://localhost:8501
```
첫 실행 시 `models/mnist-12.onnx` 를 자동으로 내려받고, 이후에는 디스크에 저장된 파일을 재사용합니다.

## 한 번에 배포하고 보고서 만들기 (권장)
Docker Desktop 을 켠 상태에서 프로젝트 폴더에서 실행합니다. (추가 패키지 설치 불필요)
```bash
python deploy.py --id <dockerhub-id> --team <팀명> --name <이름>
# Apple Silicon 맥이라면 --multiarch 를 붙이면 x86 서버용 이미지도 함께 올라갑니다
```
진행 순서: 이미지 빌드 → 컨테이너 헬스체크 → `docker login` → `docker push` → Docker Hub 반영 확인
→ `미션17/{팀명}_{이름}/` 에 **보고서 PDF**(digest · 크기 · 테스트 결과 자동 기록)와 **코드 zip** 생성.
중간 단계가 실패하면 멈추므로, 배포되지 않은 이미지로 보고서가 만들어지지 않습니다.
PDF 변환에는 PC 의 Chrome 또는 Edge 를 사용하며, 둘 다 없으면 HTML 보고서를 저장합니다
(브라우저로 열어 ‘PDF로 저장’). 보고서 내용은 `report/template.html` 에서 고칠 수 있습니다.

## Docker 빌드 · 실행 · Docker Hub 배포 (수동)
```bash
# 1) 빌드
docker build -t <dockerhub-id>/mnist-streamlit:latest .

# 2) 로컬 실행 확인
docker run --rm -p 8501:8501 <dockerhub-id>/mnist-streamlit:latest

# 3) Docker Hub 업로드
docker login
docker push <dockerhub-id>/mnist-streamlit:latest

# (Apple Silicon 등에서 빌드해 x86 서버에도 배포하려면)
docker buildx build --platform linux/amd64,linux/arm64 \
  -t <dockerhub-id>/mnist-streamlit:latest --push .
```
배포 후 이미지 주소: `https://hub.docker.com/r/<dockerhub-id>/mnist-streamlit`

## 환경 변수
| 이름 | 기본값 | 설명 |
|---|---|---|
| `MNIST_MODEL_URL` | ONNX Model Zoo mnist-12 (LFS 주소) | 모델 다운로드 URL |
| `MNIST_MODEL_DIR` | `./models` (컨테이너: `/app/models`) | 모델 저장 경로 |
