from pathlib import Path
import numpy as np
from PIL import Image


IMAGE_RESAMPLE = Image.Resampling.BILINEAR

# Load one image 
def load_image(path, img_size=64):
    img = Image.open(path).convert("L")
    img = img.resize((img_size, img_size), resample=IMAGE_RESAMPLE)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = arr[None, :, :]
    return arr

# Training split for one MVtex cat
class MVTecTrainDataset:

    def __init__(self, root_dir, category="carpet", img_size=64):
        self.image_dir = Path(root_dir) / category / "train" / "good"
        self.paths = sorted(self.image_dir.glob("*.png"))
        self.img_size = img_size

        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        x = load_image(self.paths[idx], self.img_size)
        return x, x

# Test split 
class MVTecTestDataset:

    def __init__(self, root_dir, category="carpet", img_size=64):
        self.test_dir = Path(root_dir) / category / "test"
        self.img_size = img_size
        self.samples = []

        for folder in sorted(self.test_dir.iterdir()):
            if not folder.is_dir():
                continue

            label = 0 if folder.name == "good" else 1

            for path in sorted(folder.glob("*.png")):
                self.samples.append((path, label, folder.name))

        if len(self.samples) == 0:
            raise RuntimeError(f"No test images found in {self.test_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, defect_type = self.samples[idx]
        x = load_image(path, self.img_size)
        return x, label, defect_type, str(path)

# Yield small numpy batches
def batch_loader(dataset, batch_size=4, shuffle=True):
    indices = np.arange(len(dataset))

    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]

        xs = []
        ys = []

        for idx in batch_idx:
            x, y = dataset[idx]
            xs.append(x)
            ys.append(y)

        xs = np.stack(xs, axis=0)
        ys = np.stack(ys, axis=0)

        yield xs, ys
