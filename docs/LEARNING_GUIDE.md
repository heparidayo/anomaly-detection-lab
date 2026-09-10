# 처음 읽는 PyTorch 이상탐지 학습 가이드

[README](../README.md) · [전체 구조](ARCHITECTURE.md) · [학습 가이드](LEARNING_GUIDE.md) · [실험 기록](EXPERIMENTS.md) · [설명 가이드](TEACHING_NOTES.md)

먼저 `python run_pipeline.py --smoke-test`를 실행해 파일이 생기는 것을 확인하세요.
이후 [run_pipeline.py](../run_pipeline.py) / `main()`의 단계 번호를 따라 각 함수를 찾아가면 됩니다.
줄 번호는 수정할 때 바뀌므로 아래에서는 **파일과 함수 이름**으로 연결합니다.

## 수업 중 찾아보기

- 데이터: [학습의 의미](#q1) · [Tensor](#q2) · [Dataset과 DataLoader](#q3) · [Batch](#q4)
- 학습 루프: [Forward](#q5) · [Loss](#q6) · [MSE](#q7) · [zero_grad](#q8) · [backward](#q9) · [step](#q10) · [epoch](#q11)
- 점수와 판정: [데이터 분리](#q12) · [정상만 학습하는 이유](#q13) · [복원 오차](#q14) · [Threshold](#q15)
- 평가와 현장 연결: [FP와 FN](#q16) · [오검·미검의 영향](#q17) · [성능 지표](#q18) · [정상 편차](#q19) · [추론 시간](#q20)
- 직접 해보기: [실습 6개](#exercises) · [실험 기록 양식](EXPERIMENTS.md#template)

처음에는 코드 실행과 1~11번을 연결하고, 두 번째로 읽을 때 12~20번으로 결과를 해석해 보세요.
아래 수치 예시는 [2026-09-10 기준 실험](EXPERIMENTS.md#reference-run)의 결과입니다.
직접 실행했을 때 맞춰야 하는 정답 수치는 아닙니다.

<a id="q1"></a>

## 1. 머신러닝에서 “학습한다”는 것은 실제 코드에서 무엇을 의미하나요?

[src/train.py](../src/train.py) / `train_model()`에서 입력 이미지의 복원 오차를 계산하고,
그 오차를 줄이는 방향으로 모델 안의 숫자, 즉 가중치와 bias를 여러 번 수정한다는 뜻입니다.
규칙을 사람이 하나씩 쓰는 대신 데이터에 맞게 이 숫자들을 조정합니다.
이 모델에는 학습 가능한 숫자가 214,819개 있습니다.

[src/model.py](../src/model.py)는 어떤 계산을 할지 정하고, `train.py`는 계산에 사용되는 숫자를 바꿉니다.
학습 뒤 저장하는 `state_dict`에는 이런 가중치가 들어갑니다.
불량 사진을 외우거나 “불량”이라는 단어를 이해하는 과정은 아닙니다.

<a id="q2"></a>

## 2. Tensor란 무엇인가요?

NumPy 배열처럼 숫자를 여러 축으로 담은 자료구조입니다.
PyTorch Tensor는 GPU로 옮길 수 있고 미분 계산에도 사용할 수 있습니다.
`PartDataset.__getitem__()`이 반환하는 이미지 한 장은
`float32` 자료형의 `[3,128,128]` Tensor입니다.
RGB 3개 채널마다 128×128개의 밝기 값이 있고 범위는 0~1입니다.

Pillow RGB 이미지를 NumPy 배열로 바꾸면 [높이, 너비, 채널] 순서지만
PyTorch convolution은 [채널, 높이, 너비] 순서를 사용합니다.
`ToTensor()`가 이 순서와 0~255 범위를 바꿉니다.
`images.to(device)`는 같은 숫자를 CPU 또는 GPU의 메모리로 옮깁니다.

<a id="q3"></a>

## 3. Dataset과 DataLoader는 왜 필요한가요?

[src/dataset.py](../src/dataset.py) / `PartDataset`은 “몇 장인가?”와 “i번째 이미지를 어떻게 읽나?”를 담당합니다.
각 항목은 이미지 Tensor, 정답 label, 파일 경로입니다.
`make_loaders()`가 만드는 DataLoader는 Dataset에서 여러 장을 꺼내 하나의 배치로 묶습니다.
파일 읽기와 배치 구성이 학습 루프 안에 섞이지 않아 각 단계를 이해하기 쉽습니다.

train에서는 순서를 섞어 특정 파일 순서에 치우치지 않게 합니다.
validation과 test는 같은 이미지를 같은 전처리로 반복 평가합니다.
학습·validation은 normal 폴더만 읽고 test만 두 클래스를 읽습니다.

<a id="q4"></a>

## 4. Batch는 무엇인가요?

한 번의 계산에 함께 넣는 이미지 묶음입니다.
기본 batch size는 32이므로 Tensor shape는 `[32,3,128,128]`입니다.
[**B**, **C**, **H**, **W**]에서 B=Batch, C=Channel, H=Height, W=Width입니다.

500장을 32장씩 묶으면 마지막 배치는 20장입니다.
`train_model()`에서 loss에 현재 배치 크기를 곱해 누적하는 이유는
이 작은 마지막 배치를 32장짜리 배치와 똑같은 비중으로 평균내지 않기 위해서입니다.

<a id="q5"></a>

## 5. Forward pass란 무엇인가요?

입력에서 출력까지 순서대로 계산하는 과정입니다.
`reconstructed = model(images)`를 실행하면 `AutoEncoder.forward()`가 호출됩니다.
Encoder가 이미지를 작은 표현으로 바꾸고 Decoder가 원래 해상도로 복원합니다.
forward 자체가 가중치를 수정하지는 않습니다.

기본 입력 [B,3,128,128]은 마지막 Encoder에서 [B,64,8,8]이 됩니다.
한 장의 숫자 개수는 49,152개에서 4,096개로 줄어듭니다.
latent dimension은 여기서 단일 벡터 길이가 아니라 채널×높이×너비로 된 공간 표현입니다.

<a id="q6"></a>

## 6. Loss는 무엇인가요?

모델 출력이 원하는 정답과 얼마나 다른지를 나타내는 숫자입니다.
작을수록 학습 목표에 가깝습니다.
`criterion(reconstructed, images)`에서 정답은 원본 이미지입니다.
이 실습은 분류 모델이 아니므로 label 0을 reconstruction과 비교하지 않습니다.

낮은 정상 복원 loss가 높은 불량 검출 성능을 보장하지는 않습니다.
모델이 불량까지 잘 복원하면 정상 loss는 낮아도 불량을 놓칠 수 있습니다.

<a id="q7"></a>

## 7. MSE는 왜 사용하나요?

MSE는 차이를 제곱한 뒤 평균낸 값입니다.

```python
((images - reconstructed) ** 2).mean()
```

큰 복원 오차를 더 크게 반영하고, 미분 가능하며, 코드로 이해하기 쉽습니다.
따라서 이번 교육용 복원 학습에 사용합니다.
픽셀 MSE는 사람이 중요하게 여기는 결함 크기나 위험도와 일치하지 않을 수 있고,
작은 국소 결함은 전체 이미지 평균에 희석됩니다.

`train_model()`의 loss는 배치까지 평균낸 숫자 하나입니다.
[src/inference.py](../src/inference.py) / `anomaly_scores()`는 `mean(dim=(1,2,3))`으로
채널·높이·너비만 평균내므로 배치 안 이미지마다 점수가 하나씩 나옵니다.

<a id="q8"></a>

## 8. optimizer.zero_grad()는 왜 하나요?

PyTorch는 backward를 호출하면 기존 gradient에 새 gradient를 누적합니다.
보통 이번 배치의 기울기만으로 한 번 업데이트하고 싶으므로
`train_model()`에서 forward 전에 이전 gradient를 비웁니다.
이것은 모델이 배운 가중치를 초기화하는 동작이 아닙니다.
의도적으로 gradient를 누적하는 학습 방식도 있지만 이 프로젝트에서는 쓰지 않습니다.

<a id="q9"></a>

## 9. loss.backward()에서는 실제로 무슨 일이 일어나나요?

forward 과정에서 기록한 연산 관계를 거꾸로 따라가며 미분의 연쇄법칙을 적용합니다.
각 가중치를 조금 바꿀 때 loss가 얼마나 변하는지, 즉 gradient를 계산해
파라미터의 `.grad`에 저장합니다.
일반적으로 양의 기울기 방향으로 값을 늘리면 loss가 커지는 방향이라는 뜻입니다.
복잡한 convolution의 미분식을 직접 구현할 필요가 없습니다.

이 호출만으로 가중치가 바뀌지는 않습니다. 실제 변경은 다음 `optimizer.step()`입니다.

<a id="q10"></a>

## 10. optimizer.step()은 무엇을 바꾸나요?

`train_model()`의 Adam optimizer가 모델의 weight와 bias를 수정합니다.
gradient, learning rate, 이전 gradient의 통계 등을 이용해 업데이트 크기를 정합니다.
이 프로젝트의 초기 learning rate는 0.001입니다.
learning rate 자체가 가중치인 것은 아닙니다.

학습의 핵심 순서는 다음과 같습니다.

```python
optimizer.zero_grad()
reconstructed = model(images)
loss = criterion(reconstructed, images)
loss.backward()
optimizer.step()
```

<a id="q11"></a>

## 11. epoch란 무엇인가요?

학습 데이터 전체를 한 번 사용한 단위입니다.
500장에 batch size 32면 epoch마다 16번의 업데이트를 합니다.
15 epoch는 각 정상 이미지를 대략 15번 보며 가중치를 조정한다는 뜻입니다.

epoch를 늘린다고 test 성능이 계속 좋아지지는 않습니다.
`train_model()`은 validation loss가 가장 낮은 epoch의 모델을 저장합니다.
train loss는 epoch 도중 바뀌는 모델로 측정하고, validation loss는 그 epoch 끝의 모델로
측정하므로 두 곡선이 항상 일정한 위아래 관계일 필요도 없습니다.

<a id="q12"></a>

## 12. train / validation / test 데이터는 왜 나누나요?

train은 가중치를 바꾸는 데 사용합니다.
validation은 학습에 쓰지 않은 정상 이미지에서 복원 상태를 점검하고 threshold를 정합니다.
test는 선택이 끝난 모델과 threshold의 성능을 마지막에 측정하는 데 사용합니다.

test의 불량 정답을 보고 threshold를 반복 조정하면 시험 답을 보고 기준을 맞추는 셈입니다.
그러면 보고한 성능이 새 현장 이미지의 성능보다 낙관적일 수 있습니다.
[run_pipeline.py](../run_pipeline.py)는 test inference 전에 threshold를 결정합니다.
90/95/99 비교도 교육용 관찰이며 test 결과로 기본 threshold를 자동 재선택하지 않습니다.

이 실습은 데이터 크기를 줄이려고 validation을 checkpoint 선택과 threshold 보정에 함께 사용합니다.
더 엄밀한 실험에서는 모델 선택용 validation과 별도 calibration 데이터로 나눌 수 있습니다.
실제 제조 데이터는 비슷한 연속 프레임이 train/test에 함께 들어가지 않도록
부품 개체·생산 lot·촬영 시점 등으로 나누는 것도 필요합니다.

<a id="q13"></a>

## 13. AutoEncoder는 왜 정상 이미지만 학습하나요?

정상 부품에서 반복되는 형상과 촬영 편차를 복원하도록 만들기 위해서입니다.
불량이 희귀하거나 앞으로 어떤 불량이 나올지 모르는 상황에서도
정상 데이터로 시작할 수 있다는 장점이 있습니다.

`PartDataset`은 train/val의 normal 폴더만 읽고,
`train_model()`도 label이 정상인지 검사합니다.
불량까지 학습하면 그 패턴도 낮은 오차로 복원하도록 배울 수 있습니다.

<a id="q14"></a>

## 14. Reconstruction Error가 왜 Anomaly Score가 될 수 있나요?

학습했던 정상 패턴은 잘 복원하고 낯선 패턴은 덜 복원한다는 가정이 있기 때문입니다.
하지만 이것은 항상 참인 법칙이 아닙니다.
AutoEncoder가 불량을 잘 복사하거나 밝기 변화만으로 큰 오차를 만들 수 있습니다.

기록된 기준 실험에서는 구멍 누락 25개를 모두 놓쳤습니다.
[복원 비교 그림](assets/sample_reconstruction.png)의 구멍 누락 예시에서도 모델은 누락된 상태를 복원합니다.
따라서 높은 점수는 “학습한 복원 기준에서 차이가 크다”는 의미이며,
불량 확률이나 결함의 물리적 위험도가 아닙니다.

그림의 heatmap은 RGB 평균 **절대 오차**이고 최종 scalar score는 **제곱 오차 평균**입니다.
위치를 보기 위한 표현과 판정에 쓰는 값을 구분하세요.
heatmap의 밝은 곳이 실제 결함 위치라고 보장되지도 않습니다.

<a id="q15"></a>

## 15. Threshold는 왜 필요한가요?

score는 연속적인 숫자이므로 최종 정상/불량 결정에는 경계가 필요합니다.
[src/threshold.py](../src/threshold.py) / `choose_threshold()`는 정상 validation score의 95 percentile을 구합니다.
점수를 작은 순서대로 놓았을 때 약 95% 지점에 해당하는 값이며,
NumPy는 필요한 경우 인접한 값 사이를 보간합니다.
이 값보다 크면 1(불량), 같거나 작으면 0(정상)으로 판정합니다.

정상 validation만 쓰면 불량 정답 없이 “정상 복원 오차가 보통 어디까지 나오나?”를 기준으로
경계를 정할 수 있습니다. 95 percentile은 validation에서 약 5%의 상위 점수를 허용하는 선택입니다.
미래 정상의 FP가 정확히 5%라는 보장은 없습니다. 유한한 표본 수, 동점,
학습/보정 선택과 촬영 조건 변화 때문에 달라집니다.

반면 0.01 같은 고정값은 현재 데이터와 모델의 점수 규모에 맞지 않을 수 있습니다.
`compare_thresholds()`에서 두 방식을 비교합니다.
기준 실험의 고정값 0.01은 모든 test 이미지를 정상으로 판정해 불량 100개를 놓쳤습니다.
이 숫자는 교육용 예시이며 새 프로젝트의 정답 threshold가 아닙니다.

<a id="q16"></a>

## 16. False Positive와 False Negative는 무엇인가요?

이 프로젝트에서는 **불량이 positive(1)**입니다.

- TP: 실제 불량을 불량으로 판정.
- TN: 실제 정상을 정상으로 판정.
- FP: 실제 정상을 불량으로 판정. 정상 오검.
- FN: 실제 불량을 정상으로 판정. 불량 미검.

`outputs/metrics/test_scores.csv`의 `outcome`을 FP나 FN으로 필터하면 해당 파일을 찾을 수 있습니다.
`plot_confusion()`에서 행은 실제 label, 열은 예측입니다.
기준 실험은 TP 64, TN 94, FP 6, FN 36입니다.
FPR은 FP/(FP+TN), 즉 정상 중 오검 비율이며 전체 이미지 중 FP 비율과 다릅니다.

<a id="q17"></a>

## 17. 산업 검사에서는 FP와 FN 중 어느 것이 더 위험할 수 있나요?

안전과 직결되는 부품이면 불량이 출하되는 FN의 영향이 매우 클 수 있습니다.
FP도 정상 제품의 재검사·폐기·라인 정지로 이어질 수 있습니다.
어느 쪽이 더 중요한지는 결함 종류, 후속 검사 유무, 허용 공차, 재검사 비용에 따라 달라집니다.
이 코드의 95 percentile은 학습을 위한 기본값이며 현장 승인 기준은 아닙니다.

threshold를 낮추면 같은 점수 집합에서 TP/FP는 줄어들지 않고,
threshold를 높이면 FP는 늘어나지 않지만 FN은 줄어들지 않습니다.
표본 사이에 점수가 없으면 threshold를 바꿔도 판정 개수가 그대로일 수 있습니다.
실제 기본 실행에서도 90과 95 percentile의 Recall은 둘 다 0.64였습니다.

<a id="q18"></a>

## 18. Precision / Recall / F1 / AUROC는 각각 무엇인가요?

[src/evaluate.py](../src/evaluate.py) / `compute_metrics()`에서 계산합니다.

- Precision = TP/(TP+FP). 불량이라고 보낸 것 중 실제 불량인 비율입니다.
  기준 실험은 64/70 ≈ 0.9143입니다.
- Recall = TP/(TP+FN). 실제 불량 중 잡아낸 비율입니다. 기준 실험은 64/100 = 0.64입니다.
- F1 = 2×Precision×Recall/(Precision+Recall). 둘 중 하나가 낮으면 함께 낮아집니다.
  기준 실험은 약 0.7529입니다.
- Accuracy = (TP+TN)/전체 수. 기준 실험은 158/200 = 0.79입니다.
- AUROC는 threshold를 여러 값으로 바꿀 때 FPR과 Recall 관계인 ROC 곡선 아래 면적입니다.
  불량의 점수를 정상보다 높게 순위 매기는 능력을 나타내며 동점은 절반으로 셉니다.
  0.5는 무작위 순위 수준, 1.0은 완전 분리입니다. 기준 실험은 0.8204입니다.

AUROC에는 이진 예측이 아니라 원래 score를 넣습니다.
그래서 같은 모델/테스트 집합이면 90/95/99 percentile 행의 AUROC는 같습니다.
AUROC 0.82가 현재 threshold의 Accuracy 82%라는 뜻은 아닙니다.

불량 예측이 전혀 없으면 Precision 분모가 0이므로 코드에서는 0으로 보고합니다.
test에 한 클래스만 있으면 AUROC를 계산하지 않고 `None`으로 둡니다.
실제 제조처럼 불량 비율이 매우 낮을 때 Accuracy만 보면 모두 정상이라고 예측해도
높게 나올 수 있고, Precision도 이 데모의 50:50 test 비율과 달라질 수 있습니다.

<a id="q19"></a>

## 19. 정상 편차가 왜 anomaly detection을 어렵게 만드나요?

카메라 조명이나 위치가 달라지면 불량이 없어도 픽셀 값과 경계 위치가 바뀝니다.
모델이 그 변화를 충분히 배우지 못하면 복원 오차가 커집니다.
결국 “새로운 정상 촬영 조건”과 “실제 결함”이 둘 다 큰 score로 나타날 수 있습니다.

[generate_dataset.py](../src/generate_dataset.py) / `draw_part()`는 원래 정상에도 촬영 편차를 넣습니다.
[evaluate.py](../src/evaluate.py) / `run_variation_experiments()`는 test normal에 추가 편차를 주고
같은 모델과 threshold로 score와 FPR을 측정합니다.
밝기·위치·회전 중 한 종류만 바꾸고 +와 - 방향을 모두 검사합니다.

모든 그래프가 깔끔한 증가 곡선일 필요는 없습니다.
작은 회전에서 보간으로 경계와 잡음이 부드러워지면 MSE가 오히려 작아질 수 있습니다.
이는 가능한 해석이며 이 실험만으로 원인이 입증된 것은 아닙니다.
추가 실험에서 보간 방식을 바꿔 확인할 수 있습니다.
서로 다른 점수 분포가 같은 평균을 가질 수 있으므로 평균 score와 FPR을 함께 보세요.

<a id="q20"></a>

## 20. inference time은 왜 제조현장에서 중요한가요?

다음 부품이 검사 위치에 도착하기 전에 판정을 마쳐야 하기 때문입니다.
[inference.py](../src/inference.py) / `benchmark_inference()`는 batch size 1로 warm-up 10회 후 50회 반복해
평균과 표준편차, CPU/GPU 장치를 저장합니다.
GPU 연산은 비동기라서 `torch.cuda.synchronize()`로 완료를 기다린 뒤 시간을 읽습니다.

측정 범위는 이미 장치에 올라간 Tensor의 모델 forward와 MSE입니다.
디스크 읽기·리사이즈·장치 전송은 빠져 있습니다.
실제 cycle time은 촬영, 데이터 전달, 전처리, 판정, 설비 통신과 배출 동작까지 포함해 확인해야 합니다.
이 수치의 역수를 취해 제조 라인의 보장 처리량이라고 해석하지 마세요.
다른 프로그램의 GPU 사용, 발열, 전원 설정에 따라서도 시간이 달라집니다.

<a id="exercises"></a>

## 직접 수정해볼 과제

아래에는 결과의 정답 대신 수정할 파일만 안내합니다.
실험마다 결과 폴더를 나누고 [실험 기록](EXPERIMENTS.md#template) 양식으로 예상과 실제를 비교하세요.

1. **Exercise 1:** Threshold를 95 percentile에서 90 percentile로 바꾸고 FP/FN 변화를 확인하세요.
   수정할 파일: [run_pipeline.py](../run_pipeline.py), [src/threshold.py](../src/threshold.py).
2. **Exercise 2:** 정상 학습 이미지의 brightness variation을 크게 만들고 성능을 확인하세요.
   수정할 파일: [src/generate_dataset.py](../src/generate_dataset.py).
3. **Exercise 3:** 새로운 scratch defect 형태를 추가하세요.
   수정할 파일: [src/generate_dataset.py](../src/generate_dataset.py).
4. **Exercise 4:** AutoEncoder의 latent dimension을 변경하고 결과를 비교하세요.
   수정할 파일: [src/model.py](../src/model.py), [run_pipeline.py](../run_pipeline.py).
5. **Exercise 5:** Epoch를 5 / 20 / 50으로 바꿔 결과를 비교하세요.
   수정할 파일: [run_pipeline.py](../run_pipeline.py).
6. **Exercise 6:** 이미지 해상도를 바꾸고 정확도와 추론 속도를 비교하세요.
   수정할 파일: [run_pipeline.py](../run_pipeline.py), [src/generate_dataset.py](../src/generate_dataset.py), [src/dataset.py](../src/dataset.py), [src/model.py](../src/model.py).

## 더 읽을 공식 자료

- [PyTorch 학습 루프 입문](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html):
  gradient 초기화, backward, optimizer step의 동작을 더 살펴볼 수 있습니다.
- [PyTorch 재현성 안내](https://docs.pytorch.org/docs/stable/notes/randomness.html):
  seed 고정의 범위와 장치·버전 간 차이를 이해할 때 참고하세요.
