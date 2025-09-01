# data_loader.py

import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

DOTA_LABELS = {
    'plane': 0, 'ship': 1, 'storage-tank': 2,
    'baseball-diamond': 3, 'tennis-court': 4,
    'basketball-court': 5, 'ground-track-field': 6,
    'harbor': 7, 'bridge': 8,
    'large-vehicle': 9, 'small-vehicle': 10,
    'helicopter': 11, 'roundabout': 12,
    'soccer-ball-field': 13, 'swimming-pool': 14,
}


class DotaRawDataset(Dataset):
    """
    Dataset class
    """

    def __init__(
            self,
            image_dir,
            label_dir,
            transform=None,
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def _parse_label_file(self, label_path):
        bboxes = []
        classes = []
        counts = {}
        with open(label_path, 'r') as f:
            lines = f.readlines()[2:]  # skip first 2 lines
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                points = list(map(float, parts[:8]))
                cls = parts[8]
                bbox = [(points[i], points[i + 1]) for i in range(0, 8, 2)]
                bboxes.append(bbox)
                classes.append(cls)
                if cls in counts:
                    counts[cls] += 1
                else:
                    counts.setdefault(cls, 1)
        questions = [{'question': 'How many {} in the picture?'.format(cls),
                      'result': str(count)} for cls, count in counts.items()]
        return bboxes, classes, questions

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.split('.')[0] + '.txt')
        image = Image.open(image_path).convert('RGB')
        if os.path.exists(label_path):
            bboxes, classes, questions = self._parse_label_file(label_path)
        else:
            bboxes, classes, questions = [], [], []
        if self.transform:
            image = self.transform(image)
        return image, {'boxes': bboxes, 'labels': classes, 'questions': questions}


def custom_collate_fn(batch):
    """
    customized collate_fn, support irregular size of images and multiply labels
    """
    images, targets = zip(*batch)
    return list(images), list(targets)


def get_data_loader(
        base_dir: str,
        batch_size=5,
        shuffle=True,
        num_workers=2,
        collate_fn=custom_collate_fn
):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = DotaRawDataset(
        image_dir=os.path.join(base_dir, 'images'),
        label_dir=os.path.join(base_dir, 'labelTxt'),
        transform=transform
    )
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


if __name__ == "__main__":
    train_loader = get_data_loader('../data/train')
    for batch_idx, (images, labels) in enumerate(train_loader):
        for question in labels:
            print(question)
