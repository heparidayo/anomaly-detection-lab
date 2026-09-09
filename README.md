# 자동차 부품 이미지 이상탐지 입문

## 1. Python 가상환경 만들기

Python 3.10~3.12를 설치하고 이 폴더에서 실행합니다.

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux
python3 -m venv .venv
source .venv/bin/activate
```

PowerShell에서 활성화가 차단되면 실행 정책을 바꾸지 않아도 됩니다.
이후 명령의 `python` 대신 `.\.venv\Scripts\python.exe`를 사용하세요.

## 2. 의존성 설치

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. 전체 실행

```bash
python run_pipeline.py
```

## 4. 생성되는 결과

`data/`에 합성 이미지, `outputs/models/`에 모델,
`outputs/metrics/`에 점수와 평가 CSV/JSON, `outputs/figures/`에 그래프를 저장합니다.

## 5. 코드 공부 권장 순서

`docs/ARCHITECTURE.md` → `src/generate_dataset.py` → `src/dataset.py` →
`src/model.py` → `src/train.py` → `src/inference.py` → `src/threshold.py` →
`src/evaluate.py` → `src/visualize.py` → `run_pipeline.py` → `docs/LEARNING_GUIDE.md`.

이 프로젝트는 정상 이미지만으로 작은 Convolutional AutoEncoder를 학습하는
교육용 baseline입니다. 산업용 최고 성능이나 현장 적용 가능성을 보장하지 않습니다.
