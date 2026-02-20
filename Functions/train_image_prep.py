import os
import re
import shutil
import zipfile
from PIL import Image

ZIP_PATH = r"D:\NEU\IE7615\data\raw_1_train.zip"
EXTRACT_DIR = r"D:\NEU\IE7615\data\dataset_raw"
OUTPUT_DIR = r"D:\NEU\IE7615\data\data_train"
TARGET_SIZE = (224, 224)
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.bmp', '.webp', '.tiff', '.tif'}

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# Clean & extract
for d in [EXTRACT_DIR, OUTPUT_DIR]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)

with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    zf.extractall(EXTRACT_DIR)

# Process
folder_re = re.compile(r'images_OBJ(\d{3})', re.IGNORECASE)

for root, dirs, _ in os.walk(EXTRACT_DIR):
    for d in dirs:
        match = folder_re.match(d)
        if not match:
            continue

        class_id = f"OBJ{match.group(1)}"
        src = os.path.join(root, d)
        dst = os.path.join(OUTPUT_DIR, class_id)
        os.makedirs(dst, exist_ok=True)

        count = 0
        for f in os.listdir(src):
            if os.path.splitext(f)[1].lower() not in EXTENSIONS:
                continue
            try:
                with Image.open(os.path.join(src, f)) as img:
                    img = img.convert('RGB').resize(TARGET_SIZE, Image.LANCZOS)
                    img.save(os.path.join(dst, f"{class_id}_{count:04d}.jpg"), 'JPEG', quality=95)
                    count += 1
            except Exception as e:
                print(f"  Skip {f}: {e}")

        print(f"{class_id}: {count} images")

print("Done")