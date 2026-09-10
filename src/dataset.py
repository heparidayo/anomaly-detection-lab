"""Image → Tensor → Batch 순서를 작은 Dataset과 DataLoader로 표현합니다."""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.generate_dataset import IMAGE_EXTENSIONS


class PartDataset(Dataset):
    def __init__(self, data_dir, split, image_size=128):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        self.samples = []
        # 학습·validation에는 정상만 넣는다. 불량 label은 평가할 때만 사용한다.
        classes = [("normal", 0), ("anomaly", 1)] if split == "test" else [("normal", 0)]
        for class_name, label in classes:
            folder = Path(data_dir) / split / class_name
            paths = sorted(p for p in folder.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
            if not paths:
                raise ValueError(f"No images in {folder}")
            self.samples.extend((path, label) for path in paths)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            # [H,W,C] uint8(0~255) → [C,H,W] float32(0~1).
            # 모델 출력 Sigmoid 범위와 입력 범위를 맞춰야 MSE 비교가 가능하다.
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label, str(path)


def make_loaders(data_dir, batch_size=32, image_size=128, seed=42):
    loaders = {}
    generator = torch.Generator().manual_seed(seed)
    for split in ("train", "val", "test"):
        dataset = PartDataset(data_dir, split, image_size)
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=(split == "train"),
            # Windows에서도 별도 worker 프로세스 없이 간단히 실행한다.
            num_workers=0, generator=generator if split == "train" else None,
        )
    return loaders
