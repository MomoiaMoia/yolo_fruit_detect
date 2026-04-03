import glob

import torch

from os import path as osp
from PIL import Image

from torchvision import transforms

def polygan2box(text, format='xyxy'):
    text = text.replace("\n", "").split(" ")
    class_id = int(text[0])
    poly = list(map(float, text[1:]))
    
    xs = poly[::2]
    ys = poly[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if format == 'xyxy':
        return [class_id, x_min, y_min, x_max, y_max]
    elif format == 'xywh':
        return [class_id, x_min, y_min, x_max - x_min, y_max - y_min]
    elif format == 'cxcywh':
        return [class_id, (x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min]

trans_image = transforms.Compose([
    transforms.ToTensor(),
])

class StrawberryDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        image2label = lambda x: x.replace('.jpg', '.txt').replace('images', 'labels')
        self.label_paths = [image2label(image_path) for image_path in self.image_paths]
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        image = trans_image(image)

        label_path = self.label_paths[index]
        lines = open(label_path, 'r').readlines()
        labels = [polygan2box(line, format='cxcywh') for line in lines]
        return image, labels

    @staticmethod
    def collate_fn(batch):
        imgs, all_labels = zip(*batch)
        imgs = torch.stack(imgs)

        batch_idx = []
        cls = []
        bboxes = []

        for i, labels in enumerate(all_labels):
            for label in labels:
                c, cx, cy, w, h = label
                batch_idx.append(i)
                cls.append([c])
                bboxes.append([cx, cy, w, h])

        if len(batch_idx) == 0:
            return {
                "img": imgs,
                "batch_idx": torch.empty((0,), dtype=torch.long),
                "cls": torch.empty((0, 1), dtype=torch.float32),
                "bboxes": torch.empty((0, 4), dtype=torch.float32),
            }

        return {
            "img": imgs,
            "batch_idx": torch.tensor(batch_idx, dtype=torch.long),
            "cls": torch.tensor(cls, dtype=torch.float32),
            "bboxes": torch.tensor(bboxes, dtype=torch.float32),
        }
    
    def __len__(self):
        return len(self.image_paths)



