import torch
import albumentations as A
import numpy as np

from PIL import Image


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
    
aug = A.Compose([
    # 1st
    A.Affine(translate_percent=(-0.0625, 0.0625), scale=(0.9, 1.1), rotate=(-10, 10), p=0.5),
    A.RandomSizedBBoxSafeCrop(640, 640, p=0.3),
    # 2nd
    A.Blur(p=0.01),
    A.MedianBlur(p=0.01),
    A.ToGray(p=0.01),
    A.CLAHE(p=0.01),
    A.RandomBrightnessContrast(p=0.01),
    A.RandomGamma(p=0.01),
    A.ImageCompression(compression_type='jpeg', quality_range=(75, 100), p=0.01),
    # 3rd
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    # normalize and to tensor
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


class StrawberryDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, augment=False):
        self.image_paths = image_paths
        self.augment = augment
        image2label = lambda x: x.replace('.jpg', '.txt').replace('images', 'labels')
        self.label_paths = [image2label(image_path) for image_path in self.image_paths]
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = np.asarray(Image.open(image_path).convert('RGB'))

        label_path = self.label_paths[index]
        lines = open(label_path, 'r').readlines()
        labels = [polygan2box(line, format='cxcywh') for line in lines]

        if self.augment:
            class_labels = [label[0] for label in labels]
            bboxes = [label[1:] for label in labels]
            augmented = aug(image=image, bboxes=bboxes, class_labels=class_labels)
            image = augmented["image"]
            labels = [[c, *bbox] for c, bbox in zip(augmented["class_labels"], augmented["bboxes"])]
        
        image = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
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



