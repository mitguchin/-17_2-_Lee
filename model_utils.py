"""
model_utils.py
--------------
MNIST ONNX 모델의 다운로드 · 캐싱 · 전처리 · 추론을 담당하는 모듈.

Streamlit UI(app.py)와 분리해 두었기 때문에 이 모듈만 단독으로 import 해서
테스트하거나 다른 서비스에서 재사용할 수 있습니다.

모델 사양 (onnx/models · validated/vision/classification/mnist/model/mnist-12.onnx)
  - 입력  : 이름 "Input3",           shape (1, 1, 28, 28), float32
            검은 배경(0.0)에 흰 글씨(1.0)인 28x28 흑백 이미지 (값 범위 0.0~1.0)
  - 출력  : 이름 "Plus214_Output_0", shape (1, 10), float32
            0~9 각 클래스에 대한 logit(정규화되지 않은 점수) → softmax로 확률 변환
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy import ndimage

# ---------------------------------------------------------------------------
# 1. 모델 관리 (다운로드 + 디스크 캐시)
# ---------------------------------------------------------------------------

# GitHub 저장소의 모델 파일은 Git LFS로 관리되므로 'raw' 주소가 아닌
# media.githubusercontent.com 주소를 사용해야 실제 바이너리를 받을 수 있습니다.
MODEL_URL = os.getenv(
    "MNIST_MODEL_URL",
    "https://media.githubusercontent.com/media/onnx/models/main/"
    "validated/vision/classification/mnist/model/mnist-12.onnx",
)

# 모델 저장 위치 (환경변수로 변경 가능). Docker 이미지에서는 빌드 시 미리 받아 둡니다.
MODEL_DIR = Path(os.getenv("MNIST_MODEL_DIR", Path(__file__).parent / "models"))
MODEL_PATH = MODEL_DIR / "mnist-12.onnx"

# 정상적인 mnist-12.onnx 는 약 26KB 입니다. LFS 포인터(수백 바이트)를
# 잘못 받는 경우를 걸러내기 위한 최소 크기 기준입니다.
_MIN_MODEL_BYTES = 10_000


def download_model(url: str = MODEL_URL, path: Path = MODEL_PATH) -> Path:
    """모델 파일이 없거나 손상되었을 때만 다운로드합니다 (디스크 캐시).

    이미 정상 파일이 있으면 네트워크를 사용하지 않고 바로 경로를 반환하므로,
    컨테이너 재시작이나 새 세션에서도 재다운로드가 일어나지 않습니다.
    """
    path = Path(path)
    if path.exists() and path.stat().st_size > _MIN_MODEL_BYTES:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".part")  # 다운로드 중단 시 반쪽 파일이 남지 않도록 임시 파일 사용
    urllib.request.urlretrieve(url, tmp_path)

    if tmp_path.stat().st_size <= _MIN_MODEL_BYTES:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"다운로드한 파일이 너무 작습니다. Git LFS 포인터일 수 있습니다: {url}"
        )
    tmp_path.replace(path)
    return path


def create_session(model_path: Path = MODEL_PATH) -> ort.InferenceSession:
    """ONNX Runtime 추론 세션을 생성합니다.

    세션 생성은 그래프 로딩·최적화가 들어가는 비교적 무거운 작업이므로,
    app.py 에서는 이 함수를 st.cache_resource 로 감싸 한 번만 실행되게 합니다.
    (모든 사용자 세션·모든 rerun 이 같은 세션 객체를 공유)
    """
    model_path = download_model(path=Path(model_path))
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


# ---------------------------------------------------------------------------
# 2. 이미지 전처리
# ---------------------------------------------------------------------------

def preprocess(rgba: np.ndarray | None) -> np.ndarray | None:
    """캔버스 RGBA 이미지를 MNIST 형식의 28x28 float32 배열로 변환합니다.

    MNIST 원본 데이터셋을 만들 때 사용한 방식과 같게 처리해야 정확도가 높습니다.
      1) 흑백 변환       : 검은 배경 + 흰 글씨 (MNIST와 동일한 극성)
      2) 바운딩 박스 크롭 : 글씨가 있는 영역만 잘라냄 → 캔버스 어디에 그려도 동일하게 인식
      3) 20x20 리사이즈  : 가로세로 비율을 유지한 채 긴 변을 20px로 맞춤
      4) 28x28 패딩       : 무게중심(center of mass)이 이미지 중앙에 오도록 배치
      5) 정규화           : 0~255 픽셀값을 255로 나눠 0.0~1.0 float32 로 변환
                             (ONNX Model Zoo 의 모델 설명서에 명시된 입력 범위)

    Returns:
        (28, 28) float32 배열(0.0~1.0). 아무것도 그리지 않았으면 None.
    """
    if rgba is None:
        return None

    # 1) 흑백 변환. 펜은 흰색이므로 RGB 평균을 밝기로 사용하고,
    #    알파(투명도)를 곱해 투명한 배경 픽셀은 검정(0)이 되도록 합니다.
    arr = np.asarray(rgba, dtype=np.float32)
    gray = arr[..., :3].mean(axis=2)
    if arr.shape[-1] == 4:
        gray *= arr[..., 3] / 255.0

    # 아주 옅은 픽셀(안티앨리어싱 잔여물)은 무시하고 실제 획이 있는지 확인
    ys, xs = np.nonzero(gray > 10)
    if len(ys) == 0:
        return None

    # 2) 글씨가 있는 영역만 크롭
    cropped = gray[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]

    # 3) 비율 유지하며 긴 변을 20px로 리사이즈 (LANCZOS: 축소 시 계단 현상 최소화)
    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    small = Image.fromarray(cropped.astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)

    # 4) 28x28 캔버스 가운데에 붙인 뒤, 무게중심을 (14, 14)로 이동
    canvas = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[top: top + new_h, left: left + new_w] = np.asarray(small, dtype=np.float32)

    cy, cx = ndimage.center_of_mass(canvas)
    canvas = ndimage.shift(canvas, (14 - cy, 14 - cx), order=1, mode="constant", cval=0.0)

    # 5) 0.0~1.0 으로 정규화
    return (np.clip(canvas, 0, 255) / 255.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. 추론
# ---------------------------------------------------------------------------

def softmax(logits: np.ndarray) -> np.ndarray:
    """수치적으로 안정적인 softmax (최댓값을 빼서 overflow 방지)."""
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def predict(session: ort.InferenceSession, image28: np.ndarray) -> np.ndarray:
    """전처리된 28x28 이미지로 추론해 0~9 각 레이블의 확률(합=1)을 반환합니다."""
    # 모델 입력 shape (N, C, H, W) = (1, 1, 28, 28) 로 차원 확장
    x = image28.reshape(1, 1, 28, 28).astype(np.float32)

    input_name = session.get_inputs()[0].name    # "Input3"
    output_name = session.get_outputs()[0].name  # "Plus214_Output_0"
    logits = session.run([output_name], {input_name: x})[0][0]  # shape (10,)
    return softmax(logits)
