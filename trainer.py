import os
import time
import glob
import shutil
import random

import yaml
import torch
import numpy as np

from tqdm import tqdm
from types import SimpleNamespace
from os import path as osp
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import ops
# from datasets import StrawberryDataset
from ultralytics.data.build import build_yolo_dataset
from metrics import mean_ap, SpeedMeter, Timer

def seed_everything(seed):
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    
class Trainer():
    def __init__(self, model_cfg_path, train_cfg_path):
        seed_everything(42)
        
        default_cfg = yaml.load(open("ultralytics/cfg/default.yaml", 'r'), Loader=yaml.FullLoader)
        self.train_cfg = yaml.load(open(train_cfg_path, 'r'), Loader=yaml.FullLoader)
        model_cfg = yaml.load(open(model_cfg_path, 'r'), Loader=yaml.FullLoader)
        model_cfg["scale"] = "n"
        
        self.device = torch.device(self.train_cfg['trainer']['device'])
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            print(" [!!!] Config asigned to use CUDA but no GPU found. Falling back to CPU.")
            self.device = torch.device('cpu')

        self.model = DetectionModel(cfg=model_cfg, ch=3, verbose=False).to(self.device)
        self.model.args = SimpleNamespace(**{**default_cfg, **model_cfg})

        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.train_cfg['trainer']['lr'],
                                          weight_decay=self.train_cfg['trainer']['wd'],
                                          betas=self.train_cfg['trainer']['betas'])

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                                                    T_max=self.train_cfg['trainer']['epochs'],
                                                                    eta_min=self.train_cfg['trainer']['lr'] * 0.01,
                                                                    last_epoch=-1)

        self.loss = self.model.init_criterion()
        
        self.train_ds = build_yolo_dataset(SimpleNamespace(**default_cfg), 
                                           img_path=osp.join(self.train_cfg['dataset']['root_dir'], "train"), 
                                           batch=self.train_cfg['dataset']['batch_size'], 
                                           data=self.train_cfg['dataset'],
                                           mode="train")
        self.val_ds = build_yolo_dataset(SimpleNamespace(**default_cfg), 
                                         img_path=osp.join(self.train_cfg['dataset']['root_dir'], "val"), 
                                         batch=1, 
                                         data=self.train_cfg['dataset'], 
                                         mode="val")
        
        self.train_dl = DataLoader(self.train_ds, 
                                   batch_size=self.train_cfg['dataset']['batch_size'], 
                                   shuffle=True, 
                                   num_workers=self.train_cfg['dataset']['num_workers'], 
                                   persistent_workers=True,
                                   pin_memory=True,
                                   collate_fn=self.train_ds.collate_fn)
        self.val_dl = DataLoader(self.val_ds, 
                                 batch_size=1, 
                                 shuffle=False,
                                 num_workers=self.train_cfg['dataset']['num_workers'],
                                 persistent_workers=True,
                                 pin_memory=True,
                                 collate_fn=self.val_ds.collate_fn)
        
        self.total_train = len(self.train_dl)
        self.total_val = len(self.val_dl)
        
        self._reset_folders()
        
        EXP_TIME = time.strftime("%Y%m%d-%H%M%S")
        LOG_DIR = osp.join(self.train_cfg['trainer']['log_dir'], EXP_TIME)
        self.writer = SummaryWriter(log_dir=LOG_DIR)
        
    def train(self):
        self.model.train()
        for epoch in range(self.train_cfg['trainer']['epochs']):
            loss_total, loss_box, loss_cls, loss_dfl = 0.0, 0.0, 0.0, 0.0
            for batch in tqdm(self.train_dl, desc=f"Epoch {epoch + 1}/{self.train_cfg['trainer']['epochs']}", dynamic_ncols=True):
                images = batch["img"].float() / 255.0
                images = images.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss, loss_items = self.loss(outputs, batch)
                loss.backward()
                self.optimizer.step()

                loss_total += loss.item()
                loss_box += loss_items[0]
                loss_cls += loss_items[1]
                loss_dfl += loss_items[2]
            
            self.writer.add_scalar('Loss/Total', loss_total / self.total_train, epoch)
            self.writer.add_scalar('Loss/Box', loss_box / self.total_train, epoch)
            self.writer.add_scalar('Loss/Cls', loss_cls / self.total_train, epoch)
            self.writer.add_scalar('Loss/DFL', loss_dfl / self.total_train, epoch)

            if (epoch + 1) % self.train_cfg['trainer']['metric_every'] == 0:
                self.validate(epoch, split="train")
                self.validate(epoch, split="val")
            if (epoch + 1) % self.train_cfg['trainer']['save_every'] == 0:
                self._save_all(epoch)

            self.scheduler.step()

        self._save_all(self.train_cfg['trainer']['epochs'])

    def validate(self, epoch=0, split="val"):
        self.model.eval()
        pred_list = []
        target_list = []
        
        if split == "train":
            dl = self.train_dl
            total = self.total_train * 0.2
        elif split == "val":
            dl = self.val_dl
            total = self.total_val
        
        with torch.no_grad():
            sample = 0
            for batch in tqdm(dl, desc="Validating", dynamic_ncols=True):
                images = batch["img"].float() / 255.0
                images = images.to(self.device)
                pred, _ = self.model(images)
                pred = ops.non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

                batch_size = images.shape[0]
                img_h, img_w = images.shape[2], images.shape[3]

                for i in range(batch_size):
                    pred_i = pred[i].detach().cpu().numpy() if len(pred) > i else np.zeros((0, 6), dtype=np.float32)
                    pred_list.append(pred_i)

                    idx = batch["batch_idx"] == i
                    cls = batch["cls"][idx].detach().cpu().numpy().reshape(-1)
                    bboxes = batch["bboxes"][idx].detach().cpu().numpy()

                    if len(bboxes):
                        bboxes_xyxy = ops.xywh2xyxy(torch.from_numpy(bboxes)).numpy()
                        bboxes_xyxy[:, [0, 2]] *= img_w
                        bboxes_xyxy[:, [1, 3]] *= img_h
                        targets_i = np.column_stack([cls, bboxes_xyxy]).astype(np.float32)
                    else:
                        targets_i = np.zeros((0, 5), dtype=np.float32)

                    target_list.append(targets_i)

                sample += 1
                if split == "train" and sample >= total:
                    break

        num_classes = 3
        metrics = mean_ap(pred_list, target_list, num_classes=num_classes)

        self.writer.add_scalar(f"{split.upper()}/Precision", metrics["precision"], epoch)
        self.writer.add_scalar(f"{split.upper()}/Recall", metrics["recall"], epoch)
        self.writer.add_scalar(f"{split.upper()}/F1", metrics["f1"], epoch)
        self.writer.add_scalar(f"{split.upper()}/mAP50", metrics["map50"], epoch)
        self.writer.add_scalar(f"{split.upper()}/mAP50_95", metrics["map50_95"], epoch)

        print(
            f"{split.upper()} Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
            f"F1: {metrics['f1']:.4f}, mAP50: {metrics['map50']:.4f}, mAP50-95: {metrics['map50_95']:.4f}"
        )
        
        self.model.train()
        return metrics

    def _save_all(self, epoch):
        ckpt_path = osp.join(self.train_cfg['trainer']['ckpt_dir'], f"model_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, ckpt_path)

    def _reset_folders(self):
        shutil.rmtree(self.train_cfg['trainer']['ckpt_dir'], ignore_errors=True)
        os.makedirs(self.train_cfg['trainer']['ckpt_dir'], exist_ok=True)
        
        os.makedirs(self.train_cfg['trainer']['log_dir'], exist_ok=True)

    