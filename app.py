"""
app.py
------
손글씨 숫자 인식 Streamlit 서비스 (MNIST ONNX 모델).

화면 구성 (4개 영역)
  ① 입력 캔버스      : streamlit-drawable-canvas 로 마우스 드로잉
  ② 전처리 이미지     : 모델 입력(28x28)으로 변환된 결과를 확대해 표시
  ③ 모델 추론 결과    : 0~9 레이블별 확률 막대 차트
  ④ 이미지 저장소     : 저장한 그림 + 예측 레이블 + 확률을 갤러리로 표시

실행:  streamlit run app.py
"""

from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

import model_utils as mu

# ---------------------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------------------
st.set_page_config(page_title="MNIST 손글씨 숫자 인식", page_icon="✍️", layout="wide")

CANVAS_SIZE = 280      # 캔버스 한 변 길이(px). 28의 10배로 잡아 비율 계산을 단순화
PREVIEW_SIZE = 196     # 28x28 전처리 이미지를 화면에 확대해 보여줄 크기(px)
THUMB_SIZE = 112       # 저장소 썸네일 크기(px)
GALLERY_COLS = 5       # 저장소 한 줄에 보여줄 이미지 수


# ---------------------------------------------------------------------------
# 모델 캐싱
#   st.cache_resource 는 서버 프로세스 전체에서 단 한 번만 함수를 실행하고,
#   반환된 객체(InferenceSession)를 모든 사용자 세션과 rerun 에서 공유합니다.
#   → 캔버스에 선을 그을 때마다 앱이 rerun 되어도 모델을 다시 로드하지 않습니다.
#   모델 파일 자체는 model_utils.download_model() 이 디스크에 캐시합니다.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="MNIST ONNX 모델을 불러오는 중입니다...")
def load_model():
    return mu.create_session()


# ---------------------------------------------------------------------------
# 세션 상태 초기화
#   - gallery    : 이미지 저장소 (사용자 세션별로 독립)
#   - canvas_key : 캔버스 위젯의 key. 값을 바꾸면 새 캔버스가 만들어져 그림이 지워짐
# ---------------------------------------------------------------------------
st.session_state.setdefault("gallery", [])
st.session_state.setdefault("canvas_key", 0)


def clear_canvas():
    st.session_state.canvas_key += 1


def to_display_image(arr28: np.ndarray, size: int) -> Image.Image:
    """28x28 배열(0.0~1.0)을 화면용으로 확대. NEAREST 보간으로 픽셀 경계를 그대로 보여줍니다."""
    img = Image.fromarray((np.clip(arr28, 0, 1) * 255).astype(np.uint8))
    return img.resize((size, size), Image.NEAREST)


def to_thumbnail(rgba: np.ndarray, size: int) -> Image.Image:
    """캔버스 원본(RGBA)을 검은 배경 위에 합성한 썸네일로 변환합니다."""
    img = Image.fromarray(rgba.astype(np.uint8))
    bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
    return Image.alpha_composite(bg, img).convert("L").resize((size, size), Image.LANCZOS)


def probability_chart(probs: np.ndarray) -> alt.Chart:
    """0~9 레이블별 확률 막대 차트. 예측(최댓값) 레이블만 강조색으로 표시합니다."""
    df = pd.DataFrame({"레이블": [str(i) for i in range(10)], "확률": probs})
    df["예측"] = df["확률"] == df["확률"].max()

    bars = alt.Chart(df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("레이블:N", sort=None, axis=alt.Axis(labelAngle=0, title=None)),
        y=alt.Y("확률:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title=None)),
        color=alt.condition("datum.예측", alt.value("#E4572E"), alt.value("#9AA5B1")),
        tooltip=["레이블", alt.Tooltip("확률:Q", format=".2%")],
    )
    # 1% 이상인 막대에만 수치 라벨 표시 (0% 라벨이 줄지어 보이는 것 방지)
    labels = bars.mark_text(dy=-6, fontSize=11).transform_filter("datum.확률 >= 0.01").encode(
        text=alt.Text("확률:Q", format=".0%"), color=alt.value("#6B7280")
    )
    # 아직 그린 것이 없으면(모든 확률 0) 숫자 라벨은 생략
    chart = bars + labels if probs.max() > 0 else bars
    return chart.properties(height=260)


# ---------------------------------------------------------------------------
# 사이드바: 그리기 옵션 + 모델 정보
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("설정")
    stroke_width = st.slider("펜 굵기", min_value=8, max_value=32, value=18)
    realtime = st.toggle("그리는 동안 실시간 추론", value=True,
                         help="끄면 마우스를 뗄 때(또는 왼쪽 아래 ⟳ 버튼)만 추론합니다.")
    st.divider()
    st.caption("모델 정보")
    st.markdown(
        "- 모델: `mnist-12.onnx` (ONNX Model Zoo)\n"
        "- 입력: `Input3` · (1, 1, 28, 28) float32\n"
        "- 출력: `Plus214_Output_0` · (1, 10) logits\n"
        "- 런타임: ONNX Runtime (CPU)"
    )

session = load_model()

st.title("✍️ MNIST 손글씨 숫자 인식")
st.caption("캔버스에 0~9 숫자를 하나 그리면, ONNX 모델이 실시간으로 어떤 숫자인지 예측합니다.")

col_canvas, col_pre, col_result = st.columns([1.1, 1.0, 1.4], gap="large")

# ---------------------------------------------------------------------------
# ① 입력 캔버스
# ---------------------------------------------------------------------------
with col_canvas:
    st.subheader("① 입력 캔버스")
    canvas = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",          # MNIST와 같은 '검은 배경 + 흰 글씨'
        background_color="#000000",
        update_streamlit=realtime,
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
    )
    b1, b2 = st.columns(2)
    save_clicked = b1.button("💾 저장", width="stretch", type="primary")
    b2.button("🧹 지우기", width="stretch", on_click=clear_canvas)

# 캔버스 → 전처리 → 추론
raw = canvas.image_data if canvas is not None else None
processed = mu.preprocess(raw)
probs = mu.predict(session, processed) if processed is not None else None

# ---------------------------------------------------------------------------
# ② 전처리 이미지
# ---------------------------------------------------------------------------
with col_pre:
    st.subheader("② 전처리 이미지")
    if processed is None:
        st.image(to_display_image(np.zeros((28, 28)), PREVIEW_SIZE))
        st.caption("숫자를 그리면 모델 입력(28×28)이 여기에 표시됩니다.")
    else:
        st.image(to_display_image(processed, PREVIEW_SIZE))
        st.caption("크롭 → 20px 리사이즈 → 28×28 무게중심 정렬 (값 범위 0.0~1.0)")

# ---------------------------------------------------------------------------
# ③ 모델 추론 결과
# ---------------------------------------------------------------------------
with col_result:
    st.subheader("③ 모델 추론 결과")
    if probs is None:
        st.info("아직 그린 숫자가 없습니다.")
        st.altair_chart(probability_chart(np.zeros(10)), width="stretch")
    else:
        pred = int(np.argmax(probs))
        m1, m2 = st.columns(2)
        m1.metric("예측 숫자", pred)
        m2.metric("확률", f"{probs[pred]:.1%}")
        st.altair_chart(probability_chart(probs), width="stretch")

# 저장 버튼 처리: 현재 그림 + 예측 결과를 세션 저장소에 추가
if save_clicked:
    if probs is None:
        st.toast("저장할 그림이 없습니다. 먼저 숫자를 그려 주세요.", icon="⚠️")
    else:
        pred = int(np.argmax(probs))
        st.session_state.gallery.insert(0, {   # 최신 항목이 앞에 오도록 맨 앞에 삽입
            "thumb": to_thumbnail(raw, THUMB_SIZE),
            "processed": processed.copy(),
            "pred": pred,
            "prob": float(probs[pred]),
            "probs": probs.tolist(),
            "time": datetime.now().strftime("%H:%M:%S"),
        })
        st.toast(f"저장했습니다: 예측 {pred} ({probs[pred]:.1%})", icon="✅")

# ---------------------------------------------------------------------------
# ④ 이미지 저장소
# ---------------------------------------------------------------------------
st.divider()
gallery = st.session_state.gallery
head_l, head_r = st.columns([4, 1])
head_l.subheader(f"④ 이미지 저장소 ({len(gallery)}개)")
if gallery and head_r.button("전체 삭제", width="stretch"):
    st.session_state.gallery = []
    st.rerun()

if not gallery:
    st.caption("‘💾 저장’ 버튼을 누르면 그린 이미지와 예측 결과가 여기에 쌓입니다.")
else:
    # GALLERY_COLS 개씩 끊어 그리드로 배치
    for row_start in range(0, len(gallery), GALLERY_COLS):
        cols = st.columns(GALLERY_COLS)
        for col, (idx, item) in zip(cols, enumerate(gallery[row_start: row_start + GALLERY_COLS], row_start)):
            with col, st.container(border=True):
                st.image(item["thumb"], width="stretch")
                st.markdown(f"**예측: {item['pred']}** · {item['prob']:.1%}")
                # 2순위 후보도 함께 보여주면 헷갈린 경우를 파악하기 쉬움
                second = int(np.argsort(item["probs"])[-2])
                st.caption(f"2순위 {second} ({item['probs'][second]:.1%}) · {item['time']}")
                if st.button("삭제", key=f"del_{idx}_{item['time']}", width="stretch"):
                    st.session_state.gallery.pop(idx)
                    st.rerun()
