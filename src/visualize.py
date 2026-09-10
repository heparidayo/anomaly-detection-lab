"""화면 창을 띄우지 않고 PNG를 저장하므로 Windows와 Linux 터미널에서 실행됩니다."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve

from src.inference import anomaly_scores


def save_figure(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training(history, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = [row["epoch"] for row in history]
    ax.plot(epochs, [r["train_loss"] for r in history], "o-", label="Train normal")
    ax.plot(epochs, [r["val_loss"] for r in history], "o-", label="Validation normal")
    ax.set(xlabel="Epoch", ylabel="Mean squared error", title="Learning to reconstruct normal parts")
    ax.legend()
    ax.grid(alpha=0.2)
    save_figure(fig, path)


def plot_score_histogram(labels, scores, threshold, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.histogram_bin_edges(scores, bins=24)
    ax.hist(scores[labels == 0], bins=bins, alpha=0.65, label="Test normal", color="#387ca3")
    ax.hist(scores[labels == 1], bins=bins, alpha=0.65, label="Test anomaly", color="#cf6748")
    ax.axvline(threshold, color="black", linestyle="--", label=f"Threshold = {threshold:.5f}")
    ax.set(xlabel="Anomaly score (image MSE)", ylabel="Images", title="Do normal and defect scores separate?")
    ax.legend()
    save_figure(fig, path)


def plot_confusion(metrics, path):
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(matrix, cmap="Blues", vmin=0)
    for row in range(2):
        for col in range(2):
            name = [["TN", "FP"], ["FN", "TP"]][row][col]
            ax.text(col, row, f"{name}\n{matrix[row, col]}", ha="center", va="center",
                    fontsize=16, color="white" if matrix[row, col] > matrix.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Anomaly"],
           yticklabels=["Normal", "Anomaly"], xlabel="Predicted", ylabel="Actual",
           title=f"Confusion matrix | threshold {metrics['threshold']:.5f}")
    save_figure(fig, path)


def plot_roc(labels, scores, auroc, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if auroc is not None:
        fpr, tpr, _ = roc_curve(labels, scores)
        ax.plot(fpr, tpr, label=f"AUROC = {auroc:.4f}")
    else:
        ax.text(0.5, 0.5, "AUROC requires both classes", ha="center")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance ranking")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)",
           title="ROC across score thresholds", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    save_figure(fig, path)


@torch.inference_mode()
def plot_reconstructions(model, dataset, scores, threshold, device, figures_dir):
    # 먼저 실제 FP/FN을 골라 성공한 예시만 보여주는 편향을 피한다.
    labels = np.array([label for _, label in dataset.samples])
    indices = []
    for mask in ((labels == 0) & (scores > threshold), (labels == 1) & (scores <= threshold)):
        matches = np.flatnonzero(mask)
        if len(matches):
            indices.append(int(matches[0]))
    normal = np.flatnonzero(labels == 0)
    for index in (normal[np.argmin(scores[normal])], normal[np.argmax(scores[normal])]):
        if int(index) not in indices:
            indices.append(int(index))
    seen_types = set()
    for i, (path, label) in enumerate(dataset.samples):
        defect_type = path.stem.rsplit("_", 1)[0]
        if label == 1 and defect_type not in seen_types:
            if i not in indices:
                indices.append(i)
            seen_types.add(defect_type)
    indices = indices[:8]
    images = torch.stack([dataset[i][0] for i in indices]).to(device)
    model.eval()
    reconstructed = model(images)
    # 그림은 위치를 읽기 쉬운 RGB 평균 절대 오차. scalar score는 제곱 오차 평균이다.
    errors = (images - reconstructed).abs().mean(dim=1).cpu().numpy()
    originals = images.cpu().permute(0, 2, 3, 1).numpy()
    recon = reconstructed.cpu().permute(0, 2, 3, 1).numpy()
    vmax = max(float(errors.max()), 1e-6)  # 모든 행에 같은 색 범위를 적용한다.
    fig, axes = plt.subplots(len(indices), 3, figsize=(10, 2.3 * len(indices)), squeeze=False,
                             layout="constrained")
    for row, index in enumerate(indices):
        label = labels[index]
        predicted = int(scores[index] > threshold)
        outcome = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}[(label, predicted)]
        axes[row, 0].imshow(originals[row])
        axes[row, 1].imshow(recon[row])
        heat = axes[row, 2].imshow(errors[row], cmap="inferno", vmin=0, vmax=vmax)
        name = dataset.samples[index][0].name
        axes[row, 0].set_title(f"{name}\nActual: {'anomaly' if label else 'normal'}", fontsize=9)
        axes[row, 1].set_title(f"Reconstruction | {outcome}\nScore: {scores[index]:.6f}", fontsize=9)
        axes[row, 2].set_title("Absolute error (mean over RGB)", fontsize=9)
        for ax in axes[row]:
            ax.axis("off")
    fig.colorbar(heat, ax=axes[:, 2].tolist(), shrink=0.6, label="Absolute error, shared scale")
    fig.suptitle(f"Original / reconstruction / error | threshold {threshold:.6f}")
    save_figure(fig, figures_dir / "sample_reconstruction.png")

    fig, axes = plt.subplots(1, len(indices), figsize=(2.2 * len(indices), 3), squeeze=False,
                             layout="constrained")
    for col, index in enumerate(indices):
        heat = axes[0, col].imshow(errors[col], cmap="inferno", vmin=0, vmax=vmax)
        axes[0, col].set_title(dataset.samples[index][0].stem, fontsize=8)
        axes[0, col].axis("off")
    fig.colorbar(heat, ax=axes.ravel().tolist(), shrink=0.7, label="Absolute RGB error")
    fig.suptitle("Error heatmaps (not defect segmentation ground truth)")
    save_figure(fig, figures_dir / "error_heatmap.png")


def plot_variations(rows, path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for col, kind in enumerate(("brightness", "position", "rotation")):
        values = [row for row in rows if row["variation"] == kind]
        x = [r["magnitude"] for r in values]
        axes[0, col].plot(x, [r["mean_score"] for r in values], "o-", color="#387ca3")
        axes[0, col].axhline(values[0]["threshold"], color="black", linestyle="--", label="Fixed threshold")
        axes[0, col].set(title=kind.capitalize(), ylabel="Mean anomaly score")
        axes[0, col].legend(fontsize=8)
        axes[1, col].plot(x, [r["false_positive_rate"] for r in values], "o-", color="#cf6748")
        axes[1, col].set(ylabel="False positive rate", ylim=(-0.03, 1.03))
        for row in range(2):
            axes[row, col].set_xlabel(f"Additional +/- variation ({values[0]['unit']})")
            axes[row, col].grid(alpha=0.2)
    fig.suptitle("Same normal test images; model and threshold stay fixed")
    save_figure(fig, path)

