# 다른 사람에게 설명하기

[README](../README.md) · [전체 구조](ARCHITECTURE.md) · [학습 가이드](LEARNING_GUIDE.md) · [실험 기록](EXPERIMENTS.md)

코드를 한 줄씩 모두 읽기보다, **입력 → 학습 → 점수 → 판정 → 실패 분석**의 연결을 설명하는 진행안입니다.
아래 15분은 발표 분량의 예시이며 설치·학습에 걸리는 시간은 포함하지 않습니다.

## 설명을 마쳤을 때 전달할 세 가지

1. 학습은 가중치를 바꾸는 과정이고, 추론은 그 가중치로 계산하는 과정입니다.
2. 복원 오차는 연속적인 점수이며, 임계값을 적용해야 정상/불량 판정이 됩니다.
3. 정상 편차와 실제 결함 모두 점수를 올릴 수 있고, 불량도 낮은 점수를 받을 수 있습니다.

## 발표 전에 준비하기

[README](../README.md)의 설치를 끝내고 기본 실행을 한 번 마칩니다.

```bash
python run_pipeline.py --quiet-scores
```

발표 중에는 이미 생성한 결과를 열고, 학습 루프의 동작만 짧게 보여줄 때 smoke test를 사용합니다.
청중의 컴퓨터에 PyTorch가 없어도 [기준 복원 그림](assets/sample_reconstruction.png)과
[실험 결과](EXPERIMENTS.md#reference-run)를 GitHub에서 열어 설명할 수 있습니다.
기준 결과를 보여줄 때는 미리 실행해 둔 한 실험의 결과라고 밝혀주세요.

## 15분 진행안

### 0~3분 · 무엇이 정상이고 무엇이 불량인가?

열어볼 자료: [부품 생성 코드](../src/generate_dataset.py), [복원 그림의 Original 열](assets/sample_reconstruction.png).

밝기·위치·회전이 조금 달라도 정상이라는 데이터 생성 기준을 먼저 설명합니다.
스크래치·구멍 누락 같은 결함과 촬영 조건의 차이를 구분하세요.

청중에게 던질 질문: “사진이 달라졌다는 사실만으로 불량이라고 할 수 있을까요?”

### 3~6분 · 모델은 무엇을 학습하는가?

열어볼 코드: [Dataset](../src/dataset.py) → [모델](../src/model.py) → [학습](../src/train.py).

`[B,3,128,128]`에서 각 축의 의미를 짚고,
입력 이미지 자체가 복원 학습의 정답이라는 점을 설명합니다.
`zero_grad → forward → loss → backward → step`의 역할을 연결하세요.

청중에게 던질 질문: “다섯 단계 중 실제 가중치가 바뀌는 곳은 어디인가요?”

### 6~9분 · 복원 결과는 어떻게 정상/불량 판정이 되는가?

열어볼 코드: [anomaly_scores](../src/inference.py), [choose_threshold](../src/threshold.py).
열어볼 그림: [점수 히스토그램](assets/score_histogram.png).

배치 축은 남기고 채널·높이·너비만 평균내서 이미지당 점수 하나를 만든다고 설명합니다.
임계값은 불량 test 정답을 보지 않고 정상 validation 점수로 정합니다.
점수가 임계값과 정확히 같으면 정상으로 처리합니다.

청중에게 던질 질문: “95 percentile이면 앞으로 들어오는 정상도 정확히 5%씩 오검할까요?”

### 9~12분 · 높은 성적표만으로 충분한가?

열어볼 자료: [기준 실험](EXPERIMENTS.md#reference-run), [복원 비교](assets/sample_reconstruction.png).

불량을 positive로 놓고 FP와 FN의 의미를 설명합니다.
기준 실험의 Accuracy는 79%지만 구멍 누락 Recall은 0%였다는 사실을 함께 보여주세요.
복원 그림에서 모델이 결함을 그대로 복원하는 FN 사례를 찾습니다.

청중에게 던질 질문: “정상을 잘 복원하는 모델이라면 모든 불량도 찾아낼까요?”

### 12~15분 · 정상인데 왜 불량이라고 할까?

열어볼 그림: [정상 편차 실험](assets/normal_variation.png).

모델과 임계값은 그대로 두고 같은 정상 이미지에 추가 변화만 줬다고 설명합니다.
밝기 ±20%에서 FPR이 98%가 된 결과를 살펴봅니다.
작은 회전에서는 점수가 낮아졌다는 관찰도 보여주고, 보간의 영향은 아직 가설임을 구분하세요.

청중에게 던질 질문: “다음 실험에서는 변수 하나를 무엇으로 바꾸면 좋을까요?”

## 터미널에서 이미지 한 장 판정하기

기본 실행을 완료했다면:

```bash
python -m src.inference data/test/anomaly/scratch_0000.png
```

smoke test만 실행했다면 데이터와 모델 경로를 둘 다 smoke 결과로 지정합니다.

```bash
python -m src.inference outputs/smoke/data/test/anomaly/scratch_0000.png --model outputs/smoke/models/autoencoder.pt
```

출력의 `score`, `threshold`, `prediction`을 같이 읽습니다.
한 장을 맞혔다고 전체 성능이 입증된 것은 아니므로, 앞서 본 test 결과와 연결하세요.

## 설명할 때 구분할 표현

- “정상으로 학습했다”와 “불량을 전혀 놓치지 않는다”는 다른 주장입니다.
- anomaly score는 불량 확률이 아닙니다.
- heatmap은 RGB 평균 절대 오차이고, 판정 점수는 MSE입니다.
- AUROC 0.82는 현재 임계값에서 정답률이 82%라는 뜻이 아닙니다.
- 추론 시간은 모델과 MSE 계산 범위입니다. 제조 라인의 전체 cycle time은 따로 측정해야 합니다.
- 같은 원본을 변형한 200장은 서로 독립적인 부품 200개가 아닙니다.

설명 뒤 직접 실습할 사람에게는 [과제 6개](LEARNING_GUIDE.md#exercises)와
[기록 템플릿](EXPERIMENTS.md#template)을 안내하세요.
