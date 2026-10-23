from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms

class VinaFoodDataset(Dataset):
    def __init__(self, folder_path, transform=None, image_size=224):
        folder_path = Path(folder_path)

        self.classes = sorted(d.name for d in folder_path.iterdir() if d.is_dir())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        self.sample = []

        for cls in self.classes:
            class_folder = folder_path / cls
            for path in sorted(class_folder.rglob("*")):
                if path.is_file():
                    self.sample.append(
                        (path, self.class_to_idx[cls])
                    )

    def __len__(self) -> int:
        return len(self.sample)

    def __getitem__(self, idx: int) -> dict:
        path, label = self.sample[idx]
        image = Image.open(path).convert('RGBA').convert('RGB')

        return {
            'image': self.transform(image),
            'label': label
        }