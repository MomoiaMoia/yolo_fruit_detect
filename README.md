### **YOLO v12n Fruit Ripeness Classification**

### 环境安装

1. 前往 https://pytorch.org/get-started/locally/ 安装 torch, torchvision

2. 执行 ```pip install -r requirements.txt``` 安装余下环境。

### 数据准备

草莓成熟度分类数据集，[点此下载](https://www.kaggle.com/datasets/mahyeks/multi-class-strawberry-ripeness-detection-dataset/data)。

目录组织方式：

strawberry_cls  
├─images  
└─labels

在 ```cfg/train_cfg.yaml``` 中修改 ```dataset_dir``` 目录。

### 开始训练

执行 ```python main.py```

### 检查日志

执行 ```tensorboard --logdir logs```
