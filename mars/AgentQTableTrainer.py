import os

from PIL import Image

from agent import RLQtableAgent, ACTIONS
from torch.utils.data import Dataset, DataLoader

from mars.detector import YoloV11Detector
from mars.env import SearchEnv

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
    Loading DOTA PyTorch Dataset
    """

    def __init__(
            self,
            image_dir,
            label_dir,
            transform=None,
            joint_transform=None,
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.joint_transform = joint_transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.tif'))]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def _parse_label_file(self, label_path):
        bboxes = []
        classes = []
        with open(label_path, 'r') as f:
            lines = f.readlines()[2:]
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                points = list(map(float, parts[:8]))
                cls = parts[8]
                bbox = [(points[i], points[i + 1]) for i in range(0, 8, 2)]
                bboxes.append(bbox)
                classes.append(cls)
        return bboxes, classes

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.split('.')[0] + '.txt')

        image = Image.open(image_path).convert('RGB')

        if os.path.exists(label_path):
            bboxes, classes = self._parse_label_file(label_path)
        else:
            bboxes, classes = [], []

        if self.joint_transform:
            return self.joint_transform(image, bboxes, classes)

        if self.transform:
            image = self.transform(image)
        return image, {'boxes': bboxes, 'labels': classes}


def custom_collate_fn(batch):
    """
    Customize collate_fn
    """
    images, targets = zip(*batch)
    return list(images), list(targets)


class AgentQTableTrainer:
    def __init__(self, model_path='qtable.pkl', train_path='..\data\train', val_path='..\data\val',
                 max_steps: int = 100, batch_size: int = 2, num_workers: int = 2):
        self.agent = RLQtableAgent(training=True, load=True, model=model_path)
        self.train_dataset = DotaRawDataset(
            image_dir=os.path.join(train_path, 'images'),
            label_dir=os.path.join(train_path, 'labelTxt')
        )
        self.max_steps = max_steps
        self.data_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            collate_fn=custom_collate_fn,
        )
        self.env = SearchEnv()
        self.env.set_detector(YoloV11Detector())

    def train(self):
        for images, targets in self.data_loader:
            # images = [img.to(device) for img in images]
            # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # loss_dict = _fine_train(model, images, targets, separate=separate)
            # losses = sum(loss for loss in loss_dict.values())
            # optimizer.zero_grad()
            # losses.backward()
            # optimizer.step()
            #
            # total_loss += losses.item()
            # count += len(images)
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            for image, target in zip(images, targets):
                self.env.set_image(image)
                self.env.set_target(target)
                i, j = self.env._cell_of(self.env.cx, self.env.cy)
                status = (i, j, self.env.scale_idx, ACTIONS)
                for i in range(self.max_steps):
                    action = self.agent.select_action(*status)
                    if not action:
                        print('finished with no action.')
                        break
                    status_old = status
                    status, reward, obbs, status_new, window = self.env.step(action)
                    self.agent.update(status_old, action, reward, status_new)
                    status = status_new
                self.agent.save()



if __name__ == '__main__':
    trainer = AgentQTableTrainer()
    trainer.train()
