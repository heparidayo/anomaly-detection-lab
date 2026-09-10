"""데이터 생성부터 평가까지 순서대로 읽을 수 있는 실행 진입점."""

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch

from src.dataset import make_loaders
from src.evaluate import compare_thresholds, compute_metrics, run_variation_experiments, save_csv
from src.generate_dataset import generate_dataset
from src.inference import benchmark_inference, score_loader
from src.model import AutoEncoder
from src.threshold import choose_threshold, classify
from src.train import set_seed, train_model
from src.visualize import (
    plot_confusion, plot_reconstructions, plot_roc, plot_score_histogram,
    plot_training, plot_variations,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Small educational image anomaly detection pipeline.")
    parser.add_argument("--smoke-test", action="store_true", help="80/20/20/20 images and 3 epochs, isolated outputs.")
    parser.add_argument("--data-dir", type=Path, help="Existing four-folder dataset, or empty folder to generate.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, help="Default: 15 (smoke: 3).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--image-size", type=int, default=128, help="Multiple of 16.")
    parser.add_argument("--latent-channels", type=int, default=64)
    parser.add_argument("--percentile", type=float, default=95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--quiet-scores", action="store_true", help="Save per-image scores without printing every image.")
    args = parser.parse_args()
    args.epochs = args.epochs if args.epochs is not None else (3 if args.smoke_test else 15)
    if args.epochs < 1 or args.batch_size < 1 or args.latent_channels < 1:
        parser.error("epochs, batch-size and latent-channels must be positive.")
    if args.image_size < 16 or args.image_size % 16:
        parser.error("image-size must be a positive multiple of 16.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive.")
    if not 0 < args.percentile < 100:
        parser.error("percentile must be between 0 and 100, exclusive.")
    if args.seed < 0 or args.seed >= 2**32:
        parser.error("seed must be between 0 and 2**32 - 1.")
    return args


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    default_output = root / "outputs" / "smoke" if args.smoke_test else root / "outputs"
    output_dir = (args.output_dir or default_output).resolve()
    default_data = output_dir / "data" if args.smoke_test else root / "data"
    data_dir = (args.data_dir or default_data).resolve()
    model_dir, figures_dir, metrics_dir = [output_dir / name for name in ("models", "figures", "metrics")]
    for folder in (model_dir, figures_dir, metrics_dir):
        folder.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use --device cpu or install a CUDA PyTorch build.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    # 작은 CPU 모델에서는 스레드가 너무 많으면 동기화 비용이 더 커질 수 있다.
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    set_seed(args.seed)
    print(f"Device: {device} | CPU threads: {torch.get_num_threads()}", flush=True)

    print("\n[1/9] Dataset", flush=True)
    counts = (80, 20, 20, 20) if args.smoke_test else (500, 100, 100, 100)
    generate_dataset(data_dir, counts=counts, seed=args.seed, image_size=args.image_size)
    loaders = make_loaders(data_dir, args.batch_size, args.image_size, args.seed)
    for split, loader in loaders.items():
        labels = [label for _, label in loader.dataset.samples]
        print(f"{split}: {len(labels)} images | normal={labels.count(0)} | anomaly={labels.count(1)}")
    # val은 shuffle하지 않으므로 shape 확인이 train의 난수 순서를 소비하지 않는다.
    images, _, _ = next(iter(loaders["val"]))
    print(f"Image -> Tensor [C,H,W]: {tuple(images[0].shape)}")
    print(f"DataLoader -> Batch [B,C,H,W]: {tuple(images.shape)} | range=[0,1]")

    print("\n[2/9] Train on normal images only", flush=True)
    model = AutoEncoder(args.latent_channels).to(device)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
    model_path = model_dir / "autoencoder.pt"
    history, best_epoch = train_model(
        model, loaders["train"], loaders["val"], device, args.epochs, args.learning_rate,
        model_path, metrics_dir, args.image_size, args.latent_channels,
    )

    print("\n[3/9] Calibrate threshold using validation normal only", flush=True)
    val_scores, val_labels, val_paths = score_loader(model, loaders["val"], device, not args.quiet_scores)
    if np.any(val_labels != 0):
        raise ValueError("Threshold calibration requires normal validation images.")
    threshold = choose_threshold(val_scores, args.percentile)
    print(f"Validation {args.percentile:g} percentile threshold: {threshold:.8f}")
    save_csv(metrics_dir / "validation_scores.csv", [
        {"path": str(Path(path).relative_to(data_dir)), "label": int(label), "score": float(score)}
        for path, label, score in zip(val_paths, val_labels, val_scores)
    ])
    # 모델과 그 모델에서 계산한 threshold를 함께 보관해 다른 실험 값이 섞이지 않게 한다.
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    checkpoint.update({"threshold": threshold, "percentile": args.percentile,
                       "score_definition": "mean((x - reconstruction)**2), RGB in [0,1]"})
    torch.save(checkpoint, model_path)

    print("\n[4/9] Test inference and evaluation", flush=True)
    test_scores, labels, paths = score_loader(model, loaders["test"], device, not args.quiet_scores)
    predicted = classify(test_scores, threshold)
    metrics = compute_metrics(labels, test_scores, threshold)
    outcomes = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}
    save_csv(metrics_dir / "test_scores.csv", [
        {"path": str(Path(path).relative_to(data_dir)), "label": int(label), "score": float(score),
         "threshold": threshold, "prediction": int(pred), "outcome": outcomes[(int(label), int(pred))]}
        for path, label, score, pred in zip(paths, labels, test_scores, predicted)
    ])

    print("\n[5/9] Threshold comparison (test results do not select the threshold)", flush=True)
    comparison = compare_thresholds(val_scores, labels, test_scores)
    save_csv(metrics_dir / "threshold_comparison.csv", comparison)
    for row in comparison:
        print(f"{row['method']} {row['percentile']} | threshold={row['threshold']:.6f} | "
              f"TP={row['tp']} TN={row['tn']} FP={row['fp']} FN={row['fn']} | F1={row['f1']:.4f}")

    defect_rows = []
    groups = sorted({Path(p).stem.rsplit("_", 1)[0] for p, label in zip(paths, labels) if label == 1})
    for group in groups:
        mask = np.array([label == 1 and Path(path).stem.rsplit("_", 1)[0] == group
                         for path, label in zip(paths, labels)])
        defect_rows.append({"defect": group, "images": int(mask.sum()),
                            "tp": int(predicted[mask].sum()), "fn": int((predicted[mask] == 0).sum()),
                            "recall": float(predicted[mask].mean()), "mean_score": float(test_scores[mask].mean())})
    save_csv(metrics_dir / "defect_breakdown.csv", defect_rows)

    print("\n[6/9] Normal variation experiments", flush=True)
    variation_rows, variation_scores = run_variation_experiments(
        model, loaders["test"].dataset, device, threshold, args.batch_size)
    for row in variation_scores:
        row["path"] = str(Path(row["path"]).relative_to(data_dir))
    save_csv(metrics_dir / "normal_variation.csv", variation_rows)
    save_csv(metrics_dir / "variation_scores.csv", variation_scores)

    print("\n[7/9] Warm-up and single-image inference benchmark", flush=True)
    timing = benchmark_inference(model, loaders["test"].dataset[0][0], device)

    print("\n[8/9] Save figures", flush=True)
    plot_training(history, figures_dir / "training_loss.png")
    plot_score_histogram(labels, test_scores, threshold, figures_dir / "score_histogram.png")
    plot_confusion(metrics, figures_dir / "confusion_matrix.png")
    plot_roc(labels, test_scores, metrics["auroc"], figures_dir / "roc_curve.png")
    plot_reconstructions(model, loaders["test"].dataset, test_scores, threshold, device, figures_dir)
    plot_variations(variation_rows, figures_dir / "normal_variation.png")

    print("\n[9/9] Save experiment record", flush=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"data_dir": str(data_dir), "output_dir": str(output_dir)})
    dataset_info_path = data_dir / "dataset_info.json"
    dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8")) if dataset_info_path.exists() else None
    record = {
        "config": config, "dataset_info": dataset_info,
        "actual_counts": {split: len(loader.dataset) for split, loader in loaders.items()},
        "best_epoch": best_epoch, "metrics": metrics, "inference_timing": timing,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
                        "cpu_threads": torch.get_num_threads(),
                        **{name: importlib.metadata.version(name)
                           for name in ("torchvision", "numpy", "Pillow", "matplotlib", "scikit-learn")}},
    }
    (metrics_dir / "final_metrics.json").write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    (metrics_dir / "inference_time.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")

    print("\n=== Final Result ===")
    print(f"Threshold: {threshold:.8f} (validation normal {args.percentile:g} percentile)")
    for key in ("accuracy", "precision", "recall", "f1", "auroc"):
        value = metrics[key]
        name = key.upper() if key in ("f1", "auroc") else key.capitalize()
        formatted = f"{value:.4f}" if value is not None else "unavailable"
        print(f"{name}: {formatted}")
    print(f"False Positive: {metrics['fp']}")
    print(f"False Negative: {metrics['fn']}")
    print(f"Average inference time: {timing['mean_ms_per_image']:.3f} ms/image ({timing['device_name']})")
    print("Timing scope: model forward + MSE only; batch size 1.")
    print(f"Outputs: {output_dir}")
    print("Educational synthetic-data baseline; inspect FP/FN before interpreting performance.")


if __name__ == "__main__":
    main()
