from trainer import Trainer

if __name__ == "__main__":
    model_cfg_path = 'cfgs/yolov12.yaml'
    train_cfg_path = 'cfgs/train_cfg.yaml'

    trainer = Trainer(model_cfg_path, train_cfg_path)
    trainer.train()