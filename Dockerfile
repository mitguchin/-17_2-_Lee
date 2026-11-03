# ---------------------------------------------------------------
# MNIST 손글씨 숫자 인식 Streamlit 서비스 Docker 이미지
#   빌드:  docker build -t <dockerhub-id>/mnist-streamlit:latest .
#   실행:  docker run -p 8501:8501 <dockerhub-id>/mnist-streamlit:latest
# ---------------------------------------------------------------
FROM python:3.12-slim

# 파이썬 로그 즉시 출력, .pyc 생성 안 함, pip 캐시 미사용 (이미지 용량 절감)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MNIST_MODEL_DIR=/app/models

WORKDIR /app

# 1) 의존성 먼저 설치 → 소스만 바뀌면 이 레이어는 캐시를 재사용해 빌드가 빨라짐
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 2) 소스 복사
COPY model_utils.py app.py ./
COPY .streamlit/ .streamlit/

# 3) 빌드 시점에 모델을 미리 다운로드해 이미지에 포함
#    → 컨테이너 실행 시 네트워크 없이도 바로 추론 가능, 첫 요청 지연 없음
RUN python -c "import model_utils as mu; print('model saved:', mu.download_model())"

# 4) 보안을 위해 root 가 아닌 일반 사용자로 실행
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501

# 컨테이너 상태 확인 (Streamlit 내장 헬스체크 엔드포인트)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
