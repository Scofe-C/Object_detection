import os
import random
from pathlib import Path
from PIL import Image

DATA_DIR = r"D:\NEU\IE7615\data\data_train"
OUTPUT_DIR = r"D:\NEU\IE7615\data\con_res"
EXTENSIONS = {".jpg", ".jpeg"}
GRIDS = [3, 3, 3, 4]  # first 3 are 3x3, last is 4x4


def get_random_image(folder: Path) -> Path:
    files = [f for f in folder.iterdir() if f.suffix.lower() in EXTENSIONS]
    return random.choice(files)


def fit_to_cell(img: Image.Image, cell_w: int, cell_h: int) -> Image.Image:
    ratio = min(cell_w / img.width, cell_h / img.height)
    resized = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)

    cell = Image.new("RGB", (cell_w, cell_h), (255, 255, 255))
    cell.paste(resized, ((cell_w - resized.width) // 2, (cell_h - resized.height) // 2))
    return cell


def build_grid(image_paths: list[Path], grid_size: int) -> Image.Image:
    images = [Image.open(p).convert("RGB") for p in image_paths]

    cell_w = max(img.width for img in images)
    cell_h = max(img.height for img in images)
    cells = [fit_to_cell(img, cell_w, cell_h) for img in images]

    grid = Image.new("RGB", (cell_w * grid_size, cell_h * grid_size), (255, 255, 255))
    for i, cell in enumerate(cells):
        grid.paste(cell, ((i % grid_size) * cell_w, (i // grid_size) * cell_h))

    for img in images:
        img.close()
    return grid


def main():
    base = Path(DATA_DIR)
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    class_folders = sorted([d for d in base.iterdir() if d.is_dir()])
    print(f"Found {len(class_folders)} classes")

    # Shuffle and distribute: 9, 9, 9, 12 guaranteed unique
    shuffled = class_folders.copy()
    random.shuffle(shuffled)

    assignments = [
        shuffled[0:9],    # image 1: 9 classes
        shuffled[9:18],   # image 2: 9 classes
        shuffled[18:27],  # image 3: 9 classes
        shuffled[27:39],  # image 4: 12 classes (guaranteed)
    ]

    # Fill last image to 16 with random extras
    extras = random.choices(class_folders, k=4)
    assignments[3].extend(extras)
    random.shuffle(assignments[3])

    for idx, (classes, grid_size) in enumerate(zip(assignments, GRIDS)):
        paths = [get_random_image(c) for c in classes]
        grid = build_grid(paths, grid_size)

        out_path = out / f"Group_11_con_{idx + 1}.jpg"
        grid.save(out_path, "JPEG", quality=90)
        print(f"Saved: {out_path.name} ({grid_size}x{grid_size}, {len(classes)} images)")
        grid.close()


if __name__ == "__main__":
    main()