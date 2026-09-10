"""Pillow로 정상 편차와 네 종류의 불량이 있는 작은 부품을 그립니다."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance


IMAGE_SIZE = 128
NORMAL_BRIGHTNESS = 0.08  # ±8%도 정상이다. 이 범위를 바꾸는 것이 실습 2.
DEFECT_TYPES = ("scratch", "missing_hole", "broken_edge", "dark_stain")
SPLITS = ("train/normal", "val/normal", "test/normal", "test/anomaly")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def draw_part(rng, defect=None, image_size=IMAGE_SIZE):
    """정상과 불량에 동일한 촬영 편차를 넣어, 편차 자체가 정답 힌트가 되지 않게 합니다."""
    if defect is not None and defect not in DEFECT_TYPES:
        raise ValueError(f"Unknown defect: {defect}")

    background = (35, 38, 42)
    metal = tuple(int(x) for x in np.clip(170 + rng.uniform(-6, 6, 3), 0, 255))
    image = Image.new("RGB", (128, 128), background)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((27, 22, 101, 106), radius=8, fill=metal)
    holes = [(44, 43), (84, 43), (44, 85), (84, 85)]
    missing = int(rng.integers(4)) if defect == "missing_hole" else -1
    for index, (x, y) in enumerate(holes):
        if index != missing:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=background)

    if defect == "scratch":
        y = int(rng.integers(52, 66))
        draw.line([(36, y), (62, y + 9), (91, y - 4)], fill=(35, 35, 38), width=3)
        draw.line([(61, y + 9), (71, y + 19)], fill=(55, 55, 58), width=2)
    elif defect == "broken_edge":
        y = int(rng.integers(49, 66))
        draw.polygon([(102, y), (81, y + 7), (86, y + 16), (102, y + 23)], fill=background)
    elif defect == "dark_stain":
        x = int(rng.integers(53, 64))
        y = int(rng.integers(52, 62))
        draw.ellipse((x - 10, y - 9, x + 13, y + 12), fill=(66, 61, 55))

    # 결함도 부품과 함께 움직여야 촬영 위치/각도가 결함 유무와 독립적이다.
    scale = float(rng.uniform(0.96, 1.04))
    dx, dy = rng.integers(-5, 6, size=2)
    image = image.transform(
        image.size, Image.Transform.AFFINE,
        (1 / scale, 0, 64 - (64 + int(dx)) / scale,
         0, 1 / scale, 64 - (64 + int(dy)) / scale),
        resample=Image.Resampling.BILINEAR, fillcolor=background,
    )
    image = image.rotate(float(rng.uniform(-3, 3)), resample=Image.Resampling.BILINEAR,
                         fillcolor=background)
    image = ImageEnhance.Brightness(image).enhance(
        float(rng.uniform(1 - NORMAL_BRIGHTNESS, 1 + NORMAL_BRIGHTNESS)))
    pixels = np.asarray(image, dtype=np.float32)
    pixels += rng.normal(0, 1.5, pixels.shape)  # 센서 잡음도 정상 촬영에서 생긴다.
    image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    if image_size != 128:
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return image


def generate_dataset(data_dir, counts=(500, 100, 100, 100), seed=42, image_size=IMAGE_SIZE):
    data_dir = Path(data_dir)
    if len(counts) != 4 or any(count < 1 for count in counts):
        raise ValueError("Four positive image counts are required.")
    if image_size < 16 or image_size % 16:
        raise ValueError("image_size must be a positive multiple of 16.")

    existing = [list((data_dir / split).glob("*")) for split in SPLITS]
    present = [sum(p.suffix.lower() in IMAGE_EXTENSIONS for p in files) for files in existing]
    if all(present):
        print(f"Using existing dataset: {data_dir.resolve()} ({present})")
        return False
    # 일부만 있는 데이터에 합성 이미지를 섞으면 실험을 해석하기 어려우므로 중단한다.
    any_images = data_dir.exists() and any(
        p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in data_dir.rglob("*"))
    if any_images:
        raise ValueError("Dataset is incomplete. Fill all four folders or use a NEW --data-dir.")

    # split별 난수열: train 개수를 바꿔도 val/test가 바뀌지 않는다.
    split_seeds = np.random.SeedSequence(seed).spawn(4)
    for split, count, split_seed in zip(SPLITS, counts, split_seeds):
        folder = data_dir / split
        folder.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(split_seed)
        for index in range(count):
            defect = DEFECT_TYPES[index % len(DEFECT_TYPES)] if split == "test/anomaly" else None
            name = f"{defect or 'normal'}_{index:04d}.png"
            draw_part(rng, defect, image_size).save(folder / name)
        print(f"Generated {split}: {count} images")

    info = {"seed": seed, "image_size": image_size, "counts": dict(zip(SPLITS, counts)),
            "normal_variation": {"position_px": 5, "rotation_degrees": 3,
                                 "brightness_fraction": NORMAL_BRIGHTNESS,
                                 "color_rgb_offset": 6, "noise_std_uint8": 1.5,
                                 "scale_fraction": 0.04}, "defect_types": DEFECT_TYPES}
    (data_dir / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic normal/defect images.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    args = parser.parse_args()
    generate_dataset(args.data_dir, seed=args.seed, image_size=args.image_size)
