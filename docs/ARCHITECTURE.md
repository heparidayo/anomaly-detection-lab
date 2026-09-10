# 전체 구조와 코드 읽는 순서

이 프로젝트의 목표는 작은 모델로 **데이터 준비 → 학습 → 추론 → 평가**를 직접 경험하는 것입니다.
네트워크 접속, 기업 데이터, 사전 학습 가중치 없이 설치 후 전체 데모를 실행합니다.

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

## Python 파일의 역할

1. `generate_dataset.py / draw_part(), generate_dataset()`
   Pillow로 부품을 그리고 정상 촬영 편차와 네 종류의 결함을 넣습니다.
   split별 난수열을 분리해 train 이미지 수를 바꿔도 val/test를 유지합니다.
2. `dataset.py / PartDataset, make_loaders()`
   파일을 RGB Tensor로 변환합니다. train만 shuffle하며 val/test에는 랜덤 증강을 하지 않습니다.
3. `model.py / AutoEncoder.forward()`
   네 번의 stride=2 convolution으로 압축하고 네 번의 transposed convolution으로 복원합니다.
   기본 latent shape는 [B,64,8,8]이며 전체 학습 파라미터는 214,819개입니다.
4. `train.py / train_model(), set_seed()`
   정상 이미지로만 Adam을 실행합니다. 배치 크기를 고려해 epoch 평균 loss를 구하고
   validation loss가 가장 작은 checkpoint를 저장하고 다시 불러옵니다.
5. `inference.py / anomaly_scores(), score_loader(), predict_image(), benchmark_inference()`
   이미지마다 MSE를 계산합니다. 저장한 모델로 한 장 추론도 지원하고 warm-up 후 속도를 측정합니다.
6. `threshold.py / choose_threshold(), classify()`
   validation normal의 percentile로 경계를 구합니다. 경계보다 엄격히 클 때만 불량으로 판정합니다.
7. `evaluate.py / compute_metrics(), compare_thresholds(), run_variation_experiments()`
   성능 지표, threshold 비교, 정상 편차 실험과 CSV 저장을 담당합니다.
8. `visualize.py / plot_...()`
   학습 곡선, 점수 분포, confusion matrix, ROC, 복원·heatmap, 편차 그래프를 PNG로 저장합니다.
9. `run_pipeline.py / main()`
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
