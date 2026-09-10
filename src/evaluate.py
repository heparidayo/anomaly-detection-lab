"""불량=1을 positive로 두고 평가하고, 정상 편차가 오검을 만드는지 실험합니다."""

import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from src.inference import anomaly_scores
from src.threshold import choose_threshold, classify


def save_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty result table.")
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compute_metrics(labels, scores, threshold):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or labels.size == 0:
        raise ValueError("Labels and scores must be matching nonempty 1-D arrays.")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Labels must be 0/1 and scores must be finite.")
    predicted = classify(scores, threshold)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {"threshold": float(threshold), "tp": int(tp), "tn": int(tn),
            "fp": int(fp), "fn": int(fn),
            # 불량 예측이 하나도 없을 때 precision=0으로 보고한다.
            "precision": float(precision_score(labels, predicted, zero_division=0)),
            "recall": float(recall_score(labels, predicted, zero_division=0)),
            "f1": float(f1_score(labels, predicted, zero_division=0)),
            "accuracy": float(accuracy_score(labels, predicted)),
            # AUROC에는 0/1 판정이 아니라 연속적인 점수를 넣는다.
            "auroc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
            "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None}


def compare_thresholds(val_scores, labels, test_scores):
    rows = []
    for percentile in (90, 95, 99):
        threshold = choose_threshold(val_scores, percentile)
        rows.append({"method": "validation_percentile", "percentile": percentile,
                     **compute_metrics(labels, test_scores, threshold)})
    # 임의의 상수가 현재 점수 크기와 안 맞을 수 있음을 비교하는 교육용 예시다.
    # 이 값이나 test 결과로 기본 threshold를 다시 고르지 않는다.
    rows.append({"method": "fixed_example", "percentile": "",
                 **compute_metrics(labels, test_scores, 0.01)})
    return rows


@torch.inference_mode()
def run_variation_experiments(model, test_dataset, device, threshold, batch_size=32):
    model.eval()
    normal_samples = [test_dataset[i] for i, (_, label) in enumerate(test_dataset.samples) if label == 0]
    # 새 정상 이미지를 뽑지 않고 같은 test normal에 추가 변화를 준다.
    # 각 크기에서 +방향/-방향 모두 검사해 한쪽 변화만 유리한 상황을 피한다.
    settings = {
        "brightness": ([0, 5, 10, 20, 30], "percent"),
        "position": ([0, 2, 5, 10, 20], "pixels"),
        "rotation": ([0, 3, 5, 10, 20], "degrees"),
    }
    summary_rows, individual_rows = [], []
    for kind, (magnitudes, unit) in settings.items():
        for magnitude in magnitudes:
            changed_images, metadata = [], []
            for image, _, path in normal_samples:
                for direction in ([0] if magnitude == 0 else [-1, 1]):
                    change = direction * magnitude
                    # 배경을 검정으로 채우면 새 검정 테두리가 오검 원인이 될 수 있다.
                    fill = image[:, :5, :5].mean(dim=(1, 2)).tolist()
                    if magnitude == 0:
                        changed = image
                    elif kind == "brightness":
                        changed = TF.adjust_brightness(image, 1 + change / 100)
                    elif kind == "position":
                        changed = TF.affine(image, angle=0, translate=[change, 0], scale=1,
                                            shear=[0, 0], interpolation=InterpolationMode.BILINEAR,
                                            fill=fill)
                    else:
                        changed = TF.rotate(image, angle=change,
                                            interpolation=InterpolationMode.BILINEAR, fill=fill)
                    changed_images.append(changed)
                    metadata.append((path, change))

            images = torch.stack(changed_images)
            level_scores = []
            for batch in images.split(batch_size):
                batch = batch.to(device)
                level_scores.extend(anomaly_scores(batch, model(batch)).cpu().tolist())
            for (path, change), score in zip(metadata, level_scores):
                individual_rows.append({"variation": kind, "magnitude": magnitude, "unit": unit,
                                        "signed_change": change, "path": path, "label": 0,
                                        "score": score, "threshold": threshold,
                                        "prediction": int(score > threshold)})
            scores = np.asarray(level_scores)
            summary_rows.append({"variation": kind, "magnitude": magnitude, "unit": unit,
                                 "base_images": len(normal_samples), "evaluated_images": len(scores),
                                 "mean_score": float(scores.mean()),
                                 "false_positive_rate": float((scores > threshold).mean()),
                                 "threshold": threshold})
            print(f"Variation {kind:10s} +/-{magnitude:2d} {unit:7s} | "
                  f"Mean Score: {scores.mean():.6f} | FPR: {(scores > threshold).mean():.1%}", flush=True)
    return summary_rows, individual_rows

