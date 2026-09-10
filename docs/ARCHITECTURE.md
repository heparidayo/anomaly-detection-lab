# 전체 구조와 코드 읽는 순서

[README](../README.md) · [전체 구조](ARCHITECTURE.md) · [학습 가이드](LEARNING_GUIDE.md) · [실험 기록](EXPERIMENTS.md) · [설명 가이드](TEACHING_NOTES.md)

이 프로젝트의 목표는 작은 모델로 **데이터 준비 → 학습 → 추론 → 평가**를 직접 경험하는 것입니다.
네트워크 접속, 기업 데이터, 사전 학습 가중치 없이 설치 후 전체 데모를 실행합니다.

[데이터 흐름](#flow) · [파일 역할](#files) · [실행·데이터](#running) · [결과 파일](#outputs)

<a id="flow"></a>

## 데이터 흐름

```text
Image File (128 x 128, RGB, 0~255)
    |
    v
Dataset: 이미지 한 장 읽기 / 크기 통일 / 0~1 Tensor 변환
    |
    v
DataLoader: 여러 장을 하나의 Batch로 묶기
    |
    v
Tensor [B, C, H, W] = [B, 3, 128, 128]
    |
    v
AutoEncoder
    | Encoder: [B,16,64,64] -> [B,32,32,32]
    |          -> [B,64,16,16] -> [B,64,8,8]
    | Decoder: [B,64,16,16] -> [B,32,32,32]
    |          -> [B,16,64,64] -> [B,3,128,128]
    v
Reconstruction (0~1)
    |
    v
Reconstruction Error: (input - reconstruction)^2
    |
    v
Anomaly Score: mean over C,H,W -> [B]
    |
    v
Threshold: validation normal score의 95 percentile
    |
    +---- score > threshold ----> Anomaly (1)
    |
    +---- score <= threshold ---> Normal (0)
    |
    v
Evaluation: 실제 label과 비교 -> TP / TN / FP / FN
    |
    v
CSV / JSON / PNG
```

B는 Batch Size, C는 Channel, H는 Height, W는 Width입니다.
MSE를 계산할 때 B까지 평균내면 배치 전체 점수 하나가 됩니다.
추론에서는 각 이미지를 판정해야 하므로 C, H, W만 평균냅니다.

## 학습과 평가에서 사용하는 데이터

```text
train/normal (500) ----> MSE + backward + Adam ----> 가중치 업데이트
val/normal   (100) ----> validation loss ---------> 저장할 epoch 선택
                  ----> 정상 score 95 percentile -> threshold 계산
test/normal  (100) --+
test/anomaly (100) --+-> 가중치/threshold 고정 ----> 최종 성능 평가
test/normal         ---> 편차만 추가 ------------> score / FP rate 비교
```

학습 시 Dataset이 주는 label은 전부 0입니다. Loss의 정답은 이 숫자 0이 아니라 입력 이미지입니다.
test label은 평가할 때 사용하며 optimizer나 threshold 계산 함수에 전달하지 않습니다.
이 작은 실습에서는 val을 checkpoint 선택과 threshold 보정에 함께 사용합니다.
엄밀한 후속 연구에서는 별도 calibration set을 두어 이 두 용도를 나누는 것도 검토하세요.

<a id="files"></a>

## 폴더와 파일

```text
anomaly-detection-lab/
├── README.md
├── requirements.txt
├── run_pipeline.py
├── src/                  # 단계별 Python 코드
├── data/
│   ├── train/normal/
│   ├── val/normal/
│   └── test/
│       ├── normal/
│       └── anomaly/
├── outputs/
│   ├── models/
│   ├── metrics/
│   └── figures/
└── docs/
    ├── ARCHITECTURE.md
    ├── LEARNING_GUIDE.md
    ├── EXPERIMENTS.md
    ├── TEACHING_NOTES.md
    └── assets/           # 공개 문서에 포함된 기준 결과
```

`data/`의 이미지와 `outputs/`의 실행 결과는 로컬에서 생성합니다.
`docs/assets/`는 2026-09-10 기준 실행의 설명용 사본이며 재실행해도 자동 갱신되지 않습니다.

### Python 파일의 역할

1. [generate_dataset.py](../src/generate_dataset.py) / `draw_part(), generate_dataset()`
   Pillow로 부품을 그리고 정상 촬영 편차와 네 종류의 결함을 넣습니다.
   split별 난수열을 분리해 train 이미지 수를 바꿔도 val/test를 유지합니다.
2. [dataset.py](../src/dataset.py) / `PartDataset, make_loaders()`
   파일을 RGB Tensor로 변환합니다. train만 shuffle하며 val/test에는 랜덤 증강을 하지 않습니다.
3. [model.py](../src/model.py) / `AutoEncoder.forward()`
   네 번의 stride=2 convolution으로 압축하고 네 번의 transposed convolution으로 복원합니다.
   기본 latent shape는 [B,64,8,8]이며 전체 학습 파라미터는 214,819개입니다.
4. [train.py](../src/train.py) / `train_model(), set_seed()`
   정상 이미지로만 Adam을 실행합니다. 배치 크기를 고려해 epoch 평균 loss를 구하고
   validation loss가 가장 작은 checkpoint를 저장하고 다시 불러옵니다.
5. [inference.py](../src/inference.py) / `anomaly_scores(), score_loader(), predict_image(), benchmark_inference()`
   이미지마다 MSE를 계산합니다. 저장한 모델로 한 장 추론도 지원하고 warm-up 후 속도를 측정합니다.
6. [threshold.py](../src/threshold.py) / `choose_threshold(), classify()`
   validation normal의 percentile로 경계를 구합니다. 경계보다 엄격히 클 때만 불량으로 판정합니다.
7. [evaluate.py](../src/evaluate.py) / `compute_metrics(), compare_thresholds(), run_variation_experiments()`
   성능 지표, threshold 비교, 정상 편차 실험과 CSV 저장을 담당합니다.
8. [visualize.py](../src/visualize.py) / `plot_...()`
   학습 곡선, 점수 분포, confusion matrix, ROC, 복원·heatmap, 편차 그래프를 PNG로 저장합니다.
9. [run_pipeline.py](../run_pipeline.py) / `main()`
   위 함수를 실험 순서대로 연결하고 경로·seed·버전·최종 결과를 기록합니다.

소스 파일은 `src/`라는 Python package에 있습니다.
개별 실행은 프로젝트 폴더에서 `python -m src.inference ...` 형식을 사용하세요.

## 정상 편차 실험을 읽는 법

같은 test normal 이미지에 추가 변화 하나만 가합니다. 모델과 threshold는 바꾸지 않습니다.

- 밝기: 0, ±5%, ±10%, ±20%, ±30%. 원래 이미지 값에 1±비율을 곱합니다.
- 위치: 0, 좌우 ±2, ±5, ±10, ±20 pixel. 수평 방향만 검사합니다.
- 회전: 0, ±3°, ±5°, ±10°, ±20°.
- 변화 0에서는 정상 N장을 한 번씩, 나머지는 +와 - 방향을 각각 적용해 2N장을 평가합니다.
- 이미 있던 정상 편차에 **추가**하는 양입니다. 원래 위치를 추정하거나 보정하지 않습니다.
- 모든 실험 이미지의 실제 label은 0으로 유지합니다. 따라서 불량 판정 비율이 FPR입니다.

같은 원본의 변형본들은 독립적인 새 표본이 아닙니다.
보간, 밝기 clipping, 화면 밖으로 잘린 부품도 점수에 영향을 줄 수 있습니다.
그래프가 반드시 단조 증가할 필요는 없으며, 변화가 심한 상태까지 정상이라고 보는 것은
이 합성 스트레스 실험의 가정입니다. 현장에서는 실제 허용 공차를 별도로 정의해야 합니다.

## 결과 재현

`dataset_info.json`에는 생성 seed와 정상 편차 범위를 기록합니다.
`final_metrics.json`에는 실행 인자, 실제 이미지 개수, 선택 epoch, 라이브러리 버전과 장치를 기록합니다.
모델 파일에는 state_dict, 해상도, latent channel 수, 선택 epoch와 보정한 threshold를 함께 저장합니다.
같은 환경에서 seed를 고정하지만 OS, 장치, 라이브러리가 달라지면 수치가 조금 달라질 수 있습니다.
이미 존재하는 완전한 데이터셋은 재사용하므로, 새 seed로 생성하려면 새 `--data-dir`를 지정해야 합니다.


<a id="running"></a>

## 실행 조건과 데이터 바꾸기

프로젝트 루트에서 실행합니다. 기본 설정은 128×128 RGB, batch size 32,
Adam lr=0.001, seed 42, 15 epoch, validation 정상 95 percentile입니다.
사용 가능한 옵션은 다음 명령으로 확인합니다.

```bash
python run_pipeline.py --help
python run_pipeline.py --epochs 5 --output-dir outputs/epoch5
python run_pipeline.py --smoke-test --device cpu --output-dir outputs/cpu_smoke
```

해상도는 `--image-size 64`처럼 16의 배수로 지정하고,
압축 표현의 채널 수는 `--latent-channels`로 바꿉니다.
형상은 128×128 좌표로 그린 뒤 지정 해상도로 리사이즈합니다.
기존 이미지도 Dataset에서 요청한 크기로 맞춥니다.

### 데이터 재사용과 새 실험

네 데이터 폴더 모두에 이미지가 있으면 기존 파일을 재사용합니다.
일부 폴더만 채워져 있으면 오류를 내어 합성 데이터를 섞지 않습니다.
생성 코드를 수정하거나 seed를 바꿔 새 데이터를 만들 때는 **새 빈 데이터 경로**를 지정하세요.

```bash
python run_pipeline.py --seed 123 --data-dir data/seed123 --output-dir outputs/seed123
```

같은 `--output-dir`로 실행하면 그 폴더의 결과를 갱신합니다.
비교할 실험마다 출력 경로를 나누고, 데이터가 그대로였는지 새로 생성됐는지도 기록하세요.
smoke의 기본 데이터 경로는 `outputs/smoke/data/`입니다.
smoke에 `--output-dir`만 바꾸면 데이터도 그 출력 폴더의 `data/`에 생성됩니다.

### 실제 이미지로 바꾸기

`train/normal`, `val/normal`, `test/normal`, `test/anomaly`를 갖춘 폴더를
`--data-dir`로 지정합니다. PNG/JPG/JPEG/BMP를 읽고 RGB로 변환합니다.

정상/불량 기준은 사람이 정합니다. 학습 정상에 불량이 섞이지 않도록 검토하고,
같은 부품의 비슷한 사진이 서로 다른 split에 들어가지 않게 분리하세요.
결함별 결과는 합성 파일명 `종류_번호.png`를 해석하므로 실제 파일명에 맞게 분류 기준을 조정해야 합니다.
정상 편차 실험의 배경 채움은 왼쪽 위 5×5 픽셀 평균입니다.
그 부분이 배경이 아닌 사진이라면 [편차 변환 코드](../src/evaluate.py)를 조정하세요.

<a id="single-image"></a>

### 저장한 모델로 이미지 한 장 판정하기

기본 실행을 끝낸 뒤:

```bash
python -m src.inference data/test/anomaly/scratch_0000.png
```

smoke test만 끝낸 뒤:

```bash
python -m src.inference outputs/smoke/data/test/anomaly/scratch_0000.png --model outputs/smoke/models/autoencoder.pt
```

단일 추론은 모델에 저장된 해상도와 threshold를 사용합니다. 가중치를 다시 학습하지 않습니다.
직접 지정한 출력 경로를 쓴다면 `--model`도 그 경로의 모델 파일로 맞추세요.

<a id="outputs"></a>

## 결과 파일별 읽는 법

아래 파일은 **직접 실행한 후** 생성됩니다. 기본 경로는 `outputs/`이고 smoke는 `outputs/smoke/`입니다.

1. `models/autoencoder.pt`: validation loss가 가장 낮은 모델과 그 모델로 보정한 threshold.
2. `metrics/training_history.csv`: epoch별 train/validation loss. 학습이 진행되는지 먼저 확인합니다.
3. `metrics/validation_scores.csv`: 정상 validation의 점수. threshold의 근거입니다.
4. `metrics/test_scores.csv`: 원본 경로, label, score, threshold, prediction, outcome.
   `outcome`에서 FP/FN을 찾아 실제 사진과 복원 그림을 확인합니다.
5. `metrics/threshold_comparison.csv`: 90/95/99 percentile과 고정값 0.01의 비교.
   같은 점수를 쓰므로 각 행의 AUROC는 같습니다.
6. `metrics/defect_breakdown.csv`: 결함 종류별 TP/FN과 Recall. 전체 지표가 가리는 실패를 찾습니다.
7. `metrics/normal_variation.csv`: 각 편차 크기의 평균 score와 FPR.
   `variation_scores.csv`에는 원본별 부호 있는 변화량과 점수가 있습니다.
8. `metrics/final_metrics.json`: 실제 데이터 수, 실행 인자, 선택 epoch, 지표, 환경 버전.
   `inference_time.json`에는 warm-up 횟수, 장치와 시간 측정 범위를 저장합니다.

### 그림 7개

- `training_loss.png`: 정상 복원 학습의 진행.
- `score_histogram.png`: 정상/불량 점수가 겹치는 영역과 판정 경계.
- `confusion_matrix.png`: 실제 label과 판정의 조합.
- `roc_curve.png`: 연속 score의 순위 구분 능력.
- `sample_reconstruction.png`: Original / Reconstruction / Absolute Error 비교. FP/FN이 있으면 그 사례도 포함합니다.
- `error_heatmap.png`: 같은 색 범위를 사용한 RGB 평균 절대 오차.
- `normal_variation.png`: 같은 정상 이미지의 추가 편차에 따른 score와 FPR.

그림은 별도 창 없이 PNG로 저장하며, OS별 폰트 차이를 줄이기 위해 라벨을 영어로 표기합니다.
Heatmap의 절대 오차와 판정에 사용하는 제곱 오차 평균은 서로 다른 표현입니다.

## 실행이 막혔을 때

- `python` 명령을 찾지 못하면 Python 설치와 터미널 경로를 확인하세요. Windows에서는 `py -3.12`도 확인할 수 있습니다.
- `No module named torch`라면 실행 중인 가상환경과 패키지를 설치한 환경이 같은지 확인하세요.
- GPU가 있어도 CUDA가 감지되지 않으면 [README의 설치 안내](../README.md)를 확인하거나 `--device cpu`로 진행하세요.
- 데이터 폴더가 불완전하다는 오류가 나면 네 폴더를 모두 채우거나 새 빈 `--data-dir`를 지정하세요.
- `No module named src`라면 프로젝트 루트에서 `python -m src.inference ...` 형식으로 실행하세요.
- 점수가 예상과 다르면 [기준 실험 조건](EXPERIMENTS.md#reference-run)과 자신의 epoch·데이터·threshold를 먼저 비교하세요.
