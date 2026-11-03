#!/usr/bin/env python3
"""
deploy.py
---------
빌드 → 컨테이너 실행 테스트 → Docker Hub 업로드 → 배포 확인 → 보고서 PDF · 코드 zip 생성
을 한 번에 실행하는 배포 자동화 스크립트입니다. (Windows / macOS / Linux 공통, 표준 라이브러리만 사용)

사용법
    python deploy.py --id <DockerHubID> --team <팀명> --name <이름>
    python deploy.py                       # 값을 생략하면 실행 중에 물어봅니다

결과물 (기본 위치: ./미션17/{팀명}_{이름}/)
    MNIST_숫자인식_보고서.pdf   ← 실제 배포 결과(digest, 크기, 테스트 결과 등)가 채워진 보고서
    MNIST_숫자인식_코드.zip     ← 소스 코드 (models/, 결과물 폴더 등은 제외)

준비물
    - Docker (Docker Desktop 실행 중이어야 함)
    - PDF 변환용 Chrome 또는 Edge (Windows 에는 Edge 가 기본 설치되어 있음)
      없으면 HTML 보고서를 대신 저장하니, 브라우저에서 열어 'PDF로 저장' 하면 됩니다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # 프로젝트 폴더 (Dockerfile 위치)
REPORT_DIR = ROOT / "report"                     # 보고서 템플릿 + 실행 화면 이미지
TEMPLATE = REPORT_DIR / "template.html"
PDF_NAME = "MNIST_숫자인식_보고서.pdf"
ZIP_NAME = "MNIST_숫자인식_코드.zip"

# 코드 zip 에서 뺄 폴더 / 확장자 (용량만 차지하거나 자동 생성되는 것들)
ZIP_EXCLUDE_DIRS = {"models", "__pycache__", ".git", ".venv", "venv", ".idea", ".vscode", "미션17"}
ZIP_EXCLUDE_SUFFIXES = {".pyc", ".zip", ".pdf", ".part"}


# ---------------------------------------------------------------------------
# 공통 도우미
# ---------------------------------------------------------------------------

def step(n: int, total: int, title: str) -> None:
    print(f"\n[{n}/{total}] {title}", flush=True)


def fail(msg: str) -> None:
    print(f"\n✗ {msg}", file=sys.stderr)
    sys.exit(1)


def docker(*args: str, capture: bool = False, check: bool = True) -> str:
    """docker 명령 실행. capture=False 면 진행 상황이 터미널에 그대로 보입니다."""
    cmd = ["docker", *args]
    if not capture:
        print("  $ " + " ".join(cmd), flush=True)
    result = subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if check and result.returncode != 0:
        detail = (result.stderr or "").strip() if capture else ""
        fail(f"명령이 실패했습니다: {' '.join(cmd)}\n{detail}")
    return (result.stdout or "").strip() if capture else ""


def ask(value: str | None, prompt: str) -> str:
    """인자로 받지 못한 값은 실행 중에 입력받습니다."""
    while not value:
        value = input(f"{prompt}: ").strip()
    return value


# ---------------------------------------------------------------------------
# 1~4. 빌드 · 테스트 · 업로드 · 확인
# ---------------------------------------------------------------------------

def check_docker() -> str:
    """docker 명령과 데몬(Docker Desktop)이 살아 있는지 확인하고 엔진 버전을 반환."""
    if shutil.which("docker") is None:
        fail("docker 명령을 찾을 수 없습니다. Docker Desktop 을 설치해 주세요.")
    version = docker("version", "--format", "{{.Server.Version}}", capture=True, check=False)
    if not version:
        fail("Docker 데몬에 연결할 수 없습니다. Docker Desktop 이 실행 중인지 확인해 주세요.")
    return version


def build_image(image: str) -> None:
    docker("build", "-t", image, str(ROOT))


def smoke_test(image: str, timeout: int = 90) -> str:
    """컨테이너를 임시 포트로 띄워 헬스체크가 'ok' 를 돌려주는지 확인합니다.

    호스트 포트를 비워 두면(-p 127.0.0.1::8501) Docker 가 빈 포트를 골라 주므로
    8501 을 이미 쓰고 있어도 충돌하지 않습니다.
    """
    cid = docker("run", "-d", "--rm", "-p", "127.0.0.1::8501", image, capture=True)
    try:
        mapping = docker("port", cid, "8501", capture=True).splitlines()[0]  # 예: 127.0.0.1:55012
        port = mapping.rsplit(":", 1)[1]
        url = f"http://127.0.0.1:{port}/_stcore/health"
        print(f"  컨테이너 {cid[:12]} 실행, 헬스체크 대기: {url}", flush=True)

        start = time.time()
        while time.time() - start < timeout:
            try:
                with urllib.request.urlopen(url, timeout=3) as r:
                    body = r.read().decode().strip()
                if r.status == 200 and body == "ok":
                    secs = time.time() - start
                    return f"정상 — 컨테이너 기동 후 {secs:.1f}초 만에 헬스체크(/_stcore/health) 'ok' 응답"
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                pass
            time.sleep(1.5)

        logs = docker("logs", cid, capture=True, check=False)
        fail(f"{timeout}초 안에 컨테이너가 응답하지 않았습니다. 컨테이너 로그:\n{logs[-2000:]}")
    finally:
        docker("stop", cid, capture=True, check=False)
    return ""  # (도달하지 않음)


def push_image(image: str, multiarch: bool) -> None:
    # 이미 로그인되어 있으면 저장된 자격 증명을 그대로 사용하고, 아니면 아이디/비밀번호(또는 토큰)를 묻습니다.
    docker("login")
    docker("push", image)
    if multiarch:
        # Apple Silicon(arm64) 에서 빌드해도 x86 서버에서 돌 수 있도록 두 아키텍처를 함께 올림
        docker("buildx", "build", "--platform", "linux/amd64,linux/arm64", "-t", image, "--push", str(ROOT))


def image_info(image: str, multiarch: bool) -> dict:
    """push 된 이미지의 digest · 플랫폼 · 크기를 조회합니다."""
    fmt = "{{json .}}"
    data = json.loads(docker("image", "inspect", "--format", fmt, image, capture=True))
    repo_digests = data.get("RepoDigests") or []
    digest = repo_digests[0].split("@", 1)[-1] if repo_digests else "(조회 실패)"
    platform_ = f"{data.get('Os', '?')}/{data.get('Architecture', '?')}"

    if multiarch:
        # 멀티 아키텍처로 다시 올렸다면 태그가 가리키는 것은 manifest list 이므로 그 digest 를 기록
        out = docker("buildx", "imagetools", "inspect", image, capture=True, check=False)
        for line in out.splitlines():
            if line.strip().startswith("Digest:"):
                digest = line.split(":", 1)[1].strip()
                break
        platform_ = "linux/amd64, linux/arm64"

    size_mb = int(data.get("Size", 0)) / 1_000_000
    return {"digest": digest, "platform": platform_, "size": f"{size_mb:.0f} MB (압축 전)"}


def check_docker_hub(hub_id: str, repo: str, tag: str) -> str:
    """Docker Hub 공개 API 로 태그가 실제로 올라갔는지 확인합니다."""
    url = f"https://hub.docker.com/v2/repositories/{hub_id}/{repo}/tags/{tag}"
    for attempt in range(5):  # push 직후에는 반영까지 몇 초 걸릴 수 있어 몇 번 재시도
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                info = json.load(r)
            updated = info.get("last_updated", "")[:19].replace("T", " ")
            size = info.get("full_size")
            size_txt = f", 압축 크기 {size / 1_000_000:.0f} MB" if size else ""
            return f"확인됨 — 태그 '{tag}' 최종 업데이트 {updated} (UTC){size_txt}"
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                return "비공개 저장소라 API로 조회할 수 없음 — Docker Hub 에서 Public 으로 바꾸면 누구나 pull 가능"
            time.sleep(3)
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(3)
    return "API 조회 실패 (push 는 성공함) — 잠시 후 Docker Hub 페이지에서 직접 확인해 주세요"


# ---------------------------------------------------------------------------
# 5. 보고서 PDF · 코드 zip
# ---------------------------------------------------------------------------

def find_browser(explicit: str | None) -> str | None:
    """PDF 변환에 쓸 Chrome/Edge 실행 파일을 찾습니다."""
    if explicit:
        return explicit
    candidates: list[str] = []
    system = platform.system()
    if system == "Windows":
        for base in (os.environ.get("PROGRAMFILES(X86)"), os.environ.get("PROGRAMFILES"),
                     os.environ.get("LOCALAPPDATA")):
            if base:
                candidates += [
                    rf"{base}\Microsoft\Edge\Application\msedge.exe",
                    rf"{base}\Google\Chrome\Application\chrome.exe",
                ]
    elif system == "Darwin":
        candidates += [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
                 "microsoft-edge", "msedge", "chrome"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    return next((c for c in candidates if c and Path(c).exists()), None)


def render_report(values: dict, out_dir: Path, browser: str | None) -> Path:
    """템플릿의 {{키}} 를 채운 뒤 헤드리스 브라우저로 PDF 를 만듭니다."""
    page = TEMPLATE.read_text(encoding="utf-8")
    for key, val in values.items():
        page = page.replace("{{" + key + "}}", html.escape(str(val)))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # 이미지가 상대경로로 연결되어 있으므로 템플릿과 같은 폴더 구성으로 복사
        for img in REPORT_DIR.glob("*.png"):
            shutil.copy(img, tmp / img.name)
        html_path = tmp / "report.html"
        html_path.write_text(page, encoding="utf-8")

        pdf_path = out_dir / PDF_NAME
        if browser:
            cmd = [
                browser, "--headless", "--disable-gpu", "--no-sandbox",
                "--no-first-run", "--no-default-browser-check", "--disable-extensions",
                "--disable-background-networking", "--disable-sync",
                "--no-pdf-header-footer", "--print-to-pdf-no-header",
                f"--print-to-pdf={pdf_path}",
                html_path.as_uri(),
            ]
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
            except subprocess.TimeoutExpired:
                pass  # 아래에서 HTML 로 대신 저장
            if pdf_path.exists() and pdf_path.stat().st_size > 10_000:
                return pdf_path
            print("  ! 브라우저로 PDF 변환에 실패해 HTML 보고서로 저장합니다.")

        # 브라우저가 없거나 변환에 실패한 경우: 이미지까지 함께 HTML 로 저장
        html_dir = out_dir / "보고서_HTML"
        html_dir.mkdir(exist_ok=True)
        for f in tmp.iterdir():
            if f.is_file():
                shutil.copy(f, html_dir / f.name)
        print(f"  → {html_dir / 'report.html'} 을 브라우저로 열어 '인쇄 → PDF로 저장' 해 주세요.")
        return html_dir / "report.html"


def make_code_zip(out_dir: Path) -> Path:
    """소스 코드를 'mnist-app/...' 구조로 압축합니다 (모델 파일 · 결과물 폴더 제외)."""
    zip_path = out_dir / ZIP_NAME
    out_resolved = out_dir.resolve()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(ROOT.rglob("*")):
            rel = path.relative_to(ROOT)
            if (path.is_dir()
                    or set(rel.parts) & ZIP_EXCLUDE_DIRS
                    or path.suffix in ZIP_EXCLUDE_SUFFIXES
                    or out_resolved in path.resolve().parents):
                continue
            zf.write(path, Path("mnist-app") / rel)
    return zip_path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST 서비스 빌드 · 배포 · 보고서 자동 생성")
    parser.add_argument("--id", help="Docker Hub 아이디")
    parser.add_argument("--team", help="팀명")
    parser.add_argument("--name", help="이름")
    parser.add_argument("--repo", default="mnist-streamlit", help="Docker Hub 저장소 이름 (기본: mnist-streamlit)")
    parser.add_argument("--tag", default="latest", help="이미지 태그 (기본: latest)")
    parser.add_argument("--out", help="결과물 폴더 (기본: ./미션17/{팀명}_{이름})")
    parser.add_argument("--multiarch", action="store_true",
                        help="amd64 + arm64 이미지를 함께 push (Apple Silicon 맥에서 권장)")
    parser.add_argument("--browser", help="PDF 변환에 쓸 Chrome/Edge 실행 파일 경로 (보통 자동 탐색)")
    args = parser.parse_args()

    hub_id = ask(args.id, "Docker Hub 아이디").lower()  # Docker 이미지 이름은 소문자만 허용
    team = ask(args.team, "팀명")
    name = ask(args.name, "이름")
    image = f"{hub_id}/{args.repo}:{args.tag}"
    out_dir = Path(args.out) if args.out else ROOT / "미션17" / f"{team}_{name}"

    total = 6
    print(f"이미지: {image}\n결과물 폴더: {out_dir}")

    step(1, total, "Docker 확인 및 이미지 빌드")
    docker_version = check_docker()
    build_image(image)

    step(2, total, "컨테이너 실행 테스트")
    health = smoke_test(image)
    print(f"  ✓ {health}")

    step(3, total, "Docker Hub 로그인 및 업로드")
    push_image(image, args.multiarch)
    pushed_at = dt.datetime.now().astimezone()

    step(4, total, "배포 확인")
    info = image_info(image, args.multiarch)
    hub_check = check_docker_hub(hub_id, args.repo, args.tag)
    print(f"  digest: {info['digest']}\n  {hub_check}")

    step(5, total, "보고서 PDF 생성")
    out_dir.mkdir(parents=True, exist_ok=True)
    values = {
        "TEAM": team,
        "NAME": name,
        "DOCKERHUB_ID": hub_id,
        "IMAGE": image,
        "HUB_URL": f"https://hub.docker.com/r/{hub_id}/{args.repo}",
        "DIGEST": info["digest"],
        "PLATFORM": info["platform"],
        "SIZE": info["size"],
        "PUSHED_AT": pushed_at.strftime("%Y-%m-%d %H:%M:%S (UTC%z)"),
        "DOCKER_VERSION": docker_version,
        "HEALTH": health,
        "HUB_CHECK": hub_check,
        "DATE": pushed_at.strftime("%Y년 %m월 %d일"),
        "GENERATED_AT": pushed_at.strftime("%Y-%m-%d %H:%M"),
    }
    report = render_report(values, out_dir, find_browser(args.browser))
    print(f"  ✓ {report}")

    step(6, total, "코드 zip 생성")
    code_zip = make_code_zip(out_dir)
    print(f"  ✓ {code_zip}")

    print(f"\n완료! 제출 폴더: {out_dir}\nDocker Hub: {values['HUB_URL']}")


if __name__ == "__main__":
    main()
