"""정상 validation의 점수만으로 판정 경계를 고릅니다."""

import numpy as np


def choose_threshold(normal_val_scores, percentile=95):
    scores = np.asarray(normal_val_scores, dtype=float)
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("Validation scores must be a nonempty finite 1-D array.")
    if not 0 < percentile < 100:
        raise ValueError("percentile must be between 0 and 100, exclusive")
    # 임의의 고정값과 달리, 현재 모델/데이터의 정상 복원 오차에 맞춘 경계다.
    return float(np.percentile(scores, percentile))


def classify(scores, threshold):
    # 1=불량, 0=정상. 경계와 정확히 같으면 정상으로 정의한다.
    return (np.asarray(scores) > threshold).astype(np.int64)

