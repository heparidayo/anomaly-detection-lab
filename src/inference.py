"""가중치를 바꾸지 않고 이미지별 MSE 점수와 한 장 추론 시간을 구합니다."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.model import AutoEncoder


def anomaly_scores(images, reconstructed):
    # 배치 축(0)을 남겨야 이미지마다 점수 하나가 나온다: [B,C,H,W] → [B].
    return ((images - reconstructed) ** 2).mean(dim=(1, 2, 3))


@torch.inference_mode()
def score_loader(model, loader, device, print_scores=False):
    model.eval()
    all_scores, all_labels, all_paths = [], [], []
    for images, labels, paths in loader:
        images = images.to(device)
        scores = anomaly_scores(images, model(images)).cpu().numpy()
        all_scores.extend(scores.tolist())
        all_labels.extend(labels.tolist())
        all_paths.extend(paths)
        if print_scores:
            for path, score in zip(paths, scores):
                print(f"{Path(path).parent.name}/{Path(path).name} : {score:.8f}")
    return np.asarray(all_scores), np.asarray(all_labels), all_paths


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = AutoEncoder(checkpoint["latent_channels"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def predict_image(image_path, model_path, device):
    model, checkpoint = load_model(model_path, device)
    transform = transforms.Compose([
        transforms.Resize((checkpoint["image_size"], checkpoint["image_size"])),
        transforms.ToTensor(),
    ])
    with Image.open(image_path) as image:
        images = transform(image.convert("RGB")).unsqueeze(0).to(device)
    score = anomaly_scores(images, model(images)).item()
    threshold = checkpoint["threshold"]
    return {"image": str(image_path), "score": score, "threshold": threshold,
            "prediction": "anomaly" if score > threshold else "normal", "device": str(device)}


@torch.inference_mode()
def benchmark_inference(model, image, device, warmup=10, repeats=50):
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup nonnegative")
    model.eval()
    images = image.unsqueeze(0).to(device)  # batch size 1; 장치 전송은 측정에 포함하지 않는다.

    def synchronize():
        # CUDA는 비동기 실행이므로 완료를 기다리지 않으면 실제보다 짧게 측정된다.
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        anomaly_scores(images, model(images))
    synchronize()
    times_ms = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        anomaly_scores(images, model(images))
        synchronize()
        times_ms.append((time.perf_counter() - start) * 1000)
    return {"device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "batch_size": 1, "warmup": warmup, "repeats": repeats,
            "mean_ms_per_image": float(np.mean(times_ms)),
            "std_ms_per_image": float(np.std(times_ms)),
            "scope": "forward + MSE; excludes image loading, preprocessing and device transfer"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict one image with the saved model and threshold.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs/models/autoencoder.pt")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps(predict_image(args.image, args.model, device), indent=2))

