# 자동차 부품 이미지 이상탐지 입문

## 1. Python 가상환경 만들기

Python 3.10~3.12를 설치하고 이 프로젝트 폴더에서 실행합니다.

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
이 작업 환경에는 이미 `.venv`를 만들어 두었으므로 바로 아래 실행 명령을 사용할 수 있습니다.

## 2. 의존성 설치

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch와 torchvision은 서로 맞는 버전인 2.7.1 / 0.22.1을 사용합니다.
CPU 전용 설치가 필요하면 requirements 설치 전에 다음을 실행할 수 있습니다.

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
```

NVIDIA GPU를 쓰려면 GPU 드라이버에 맞는 PyTorch CUDA 빌드가 필요합니다.
이 프로젝트의 실제 검증 환경은 CUDA 12.8 빌드였습니다.

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

환경별 설치 선택은 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)를 참고하세요.
코드는 `torch.cuda.is_available()`로 GPU 사용 가능 여부를 확인하며 없으면 CPU를 사용합니다.
패키지 설치 후에는 데이터 다운로드나 모델 다운로드 없이 동작합니다.

## 3. 전체 실행 명령

```bash
python run_pipeline.py
```

Windows에서 가상환경을 활성화하지 않고 바로 실행하려면:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py
```

기본값: 128×128 RGB, 학습 정상 500장, validation 정상 100장,
test 정상 100장/불량 100장, 15 epoch, batch size 32, Adam lr=0.001, seed=42.
15 epoch 중 validation loss가 가장 낮은 모델을 최종 평가합니다.
CPU에서는 시간이 더 걸릴 수 있습니다.

빠른 동작 확인:

```bash
python run_pipeline.py --smoke-test
```

정상 학습 80장, validation 20장, test 정상/불량 각 20장, 3 epoch로 실행합니다.
smoke 데이터와 결과는 `outputs/smoke/`에 분리하여 기본 데이터와 결과를 덮어쓰지 않습니다.

## 4. 생성되는 결과

- `data/train/normal/`, `data/val/normal/`, `data/test/normal/`, `data/test/anomaly/`: 합성 부품 PNG.
- `data/dataset_info.json`: 데이터 생성 조건.
- `outputs/models/autoencoder.pt`: 가장 좋은 validation epoch의 가중치와 threshold.
- `outputs/metrics/training_history.csv`: epoch별 train/validation loss.
- `outputs/metrics/validation_scores.csv`: threshold 계산에 사용한 정상 점수.
- `outputs/metrics/test_scores.csv`: 모든 test 이미지의 점수·정답·판정·TP/TN/FP/FN.
- `outputs/metrics/threshold_comparison.csv`: 90/95/99 percentile 및 고정값 0.01 예시 비교.
- `outputs/metrics/defect_breakdown.csv`: 결함 종류별 Recall과 FN.
- `outputs/metrics/normal_variation.csv`: 편차 단계별 평균 점수와 FPR.
- `outputs/metrics/variation_scores.csv`: 각 변형 이미지의 원본 경로·변화량·점수.
- `outputs/metrics/final_metrics.json`: 최종 지표·실행 조건·환경 버전.
- `outputs/metrics/inference_time.json`: 장치와 batch=1 추론 시간·측정 범위.
- `outputs/figures/`: training_loss, score_histogram, confusion_matrix, roc_curve,
  sample_reconstruction, error_heatmap, normal_variation PNG 7개.

PNG는 matplotlib으로 저장하며 별도 웹 UI나 서버를 사용하지 않습니다.
그림의 라벨은 OS별 한글 폰트 문제를 피하기 위해 영어로 표기합니다.
생성 데이터·모델·결과와 가상환경은 Git에서 제외하고 소스와 학습 문서를 커밋합니다.

## 5. 코드 공부 권장 순서

1. `docs/ARCHITECTURE.md`로 전체 흐름을 봅니다.
2. `src/generate_dataset.py` → `src/dataset.py`에서 이미지가 Tensor가 되는 과정을 읽습니다.
3. `src/model.py` → `src/train.py`에서 모델과 학습 루프를 읽습니다.
4. `src/inference.py` → `src/threshold.py`에서 점수와 판정을 읽습니다.
5. `src/evaluate.py` → `src/visualize.py`에서 결과 해석을 읽습니다.
6. `run_pipeline.py`로 전체 실행을 연결합니다.
7. `docs/LEARNING_GUIDE.md`의 20개 질문과 6개 실습, `docs/EXPERIMENTS.md`의 기록 양식을 사용합니다.

## 이 baseline이 하는 일

정상 이미지만 복원하도록 작은 CNN AutoEncoder를 학습합니다.
이미지별 `mean((x - x_hat)**2)`가 validation normal의 95 percentile보다 크면 불량으로 판정합니다.
이는 산업용 최신 최고 성능 모델이 아니라 전체 학습 사이클을 이해하기 위한 교육용 baseline입니다.
불량도 잘 복원하거나 정상의 촬영 편차를 불량으로 오판할 수 있습니다.
합성 데이터의 점수로 실제 자동차 부품 검사 성능을 주장할 수 없습니다.

## 실행 옵션과 새 이미지 추론

```bash
python run_pipeline.py --help
python run_pipeline.py --epochs 5 --output-dir outputs/epoch5
python run_pipeline.py --device cpu --smoke-test --output-dir outputs/cpu_smoke
python run_pipeline.py --percentile 90 --output-dir outputs/p90
python run_pipeline.py --quiet-scores
python -m src.inference data/test/anomaly/scratch_0000.png
```

기본 실행은 validation/test의 모든 이미지 점수를 콘솔에 출력합니다.
`--quiet-scores`는 콘솔만 줄이며 CSV는 그대로 저장합니다.
단일 추론은 모델에 저장된 해상도·threshold를 사용하고 가중치를 다시 학습하지 않습니다.
다른 실행 결과를 쓸 때는 `--model outputs/epoch5/models/autoencoder.pt`를 지정하세요.

데이터가 네 폴더 모두에 있으면 재생성하지 않습니다.
일부 폴더만 채워져 있으면 오류를 내어 기존 이미지에 합성 데이터를 섞지 않습니다.
데이터 생성 코드를 바꾸거나 생성 seed를 바꿀 때는 새 빈 경로를 지정하세요.

```bash
python run_pipeline.py --seed 123 --data-dir data/seed123 --output-dir outputs/seed123
```

생성되는 형상은 128×128 좌표를 기준으로 하며 다른 해상도에서는 리사이즈합니다.
`--image-size 64`처럼 16의 배수인 해상도를 지원합니다.
기존 데이터도 입력 단계에서 지정 해상도로 맞춥니다.
`--latent-channels`는 압축 표현의 채널 수를 바꿉니다.
여러 실험은 서로 다른 `--output-dir`에 저장하세요. 같은 출력 경로로 재실행하면 그 경로의 결과를 갱신합니다.

기업 데이터로 바꿀 때도 같은 네 폴더 구성을 사용하고 `--data-dir`를 지정하면 됩니다.
정상/불량 기준은 사람이 정해야 하며, 학습 정상에 불량이 섞이지 않았는지 확인해야 합니다.
결함 종류별 표는 합성 파일명 규칙 `종류_번호.png`를 해석하므로 다른 파일명은 직접 분류 체계를 맞춰야 합니다.
실제 이미지를 쓴 정상 편차 실험의 배경 채움은 왼쪽 위 5×5 영역의 평균값을 사용하므로
그 영역이 배경이 아닌 사진에서는 실험 변환 코드를 조정해야 합니다.

## 실제 검증 기록

기본 실행과 smoke test의 실제 수치는 `docs/EXPERIMENTS.md`에 기록합니다.
추론 시간은 warm-up 10회 후 50회 반복한 **모델 forward + MSE** 평균입니다.
이미지 읽기, 전처리, GPU 전송, 카메라 촬영, PLC 통신, 배출기 동작 시간은 포함하지 않습니다.
따라서 이 시간만으로 제조 라인의 전체 cycle time을 충족한다고 판단하면 안 됩니다.
