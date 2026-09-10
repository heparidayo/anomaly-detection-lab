"""정상 이미지를 정답으로 삼아 입력과 복원의 차이를 줄입니다."""

import csv
import random

import numpy as np
import torch
from torch import nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 같은 환경의 반복 실행에서 변동을 줄인다. 다른 OS/장치 간 완전 일치는 보장하지 않는다.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_model(model, train_loader, val_loader, device, epochs, learning_rate,
                model_path, metrics_dir, image_size=128, latent_channels=64):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for images, labels, _ in train_loader:
            if torch.any(labels != 0):
                raise ValueError("Training must contain normal images only.")
            images = images.to(device)  # [B,C,H,W], 여기서는 [B,3,128,128].
            optimizer.zero_grad()  # 이전 배치 기울기가 이번 배치에 더해지지 않게 한다.
            reconstructed = model(images)  # forward pass
            loss = criterion(reconstructed, images)  # 정답은 class label이 아니라 입력 이미지.
            loss.backward()  # loss를 줄이려면 가중치를 어느 방향으로 바꿀지 계산한다.
            optimizer.step()  # Adam이 기울기를 이용해 실제 가중치를 바꾼다.
            # 마지막 배치가 작아도 모든 이미지가 동일한 비중을 갖도록 누적한다.
            train_loss_sum += loss.item() * images.size(0)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():  # validation은 가중치를 바꾸지 않아 기울기 기록이 필요 없다.
            for images, labels, _ in val_loader:
                if torch.any(labels != 0):
                    raise ValueError("Validation must contain normal images only.")
                images = images.to(device)
                loss = criterion(model(images), images)
                val_loss_sum += loss.item() * images.size(0)

        train_loss = train_loss_sum / len(train_loader.dataset)
        val_loss = val_loss_sum / len(val_loader.dataset)
        if not np.isfinite([train_loss, val_loss]).all():
            raise RuntimeError("Non-finite loss. Check images and learning rate.")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}", flush=True)

        # test 성적을 보고 모델을 고르면 평가 정답을 미리 본 셈이 된다.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "val_loss": val_loss, "image_size": image_size,
                        "latent_channels": latent_channels}, model_path)

    with (metrics_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded best validation checkpoint: epoch {checkpoint['epoch']}")
    return history, checkpoint["epoch"]
