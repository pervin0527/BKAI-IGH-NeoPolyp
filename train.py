import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import yaml
import torch

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from mmseg.models import build_segmentor

from model_confing import get_model_cfg
from data.BKAIDataset import BKAIDataset
from metrics.loss import FocalLoss_Ori
from utils import save_config_to_yaml
from metrics.metric import intersectionAndUnionGPU, AverageMeter


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    ## Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min([os.cpu_count(), config["batch_size"] if config["batch_size"] > 1 else 0, 8])

    ## Dataset
    train_dataset = BKAIDataset(config)
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=num_workers)
    total_steps = len(train_dataloader)

    ## Model
    model = build_segmentor(get_model_cfg(config)).to(device)
    model.init_weights()
    os.system("clear")

    ## Loss & Optimizer
    criterion = FocalLoss_Ori(num_class=config["num_classes"], alpha=config["alpha"], gamma=config["gamma"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["init_lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader) * config["epochs"], eta_min=config["init_lr"] / 100)

    ## Meter
    train_loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    ## Save Config
    save_dir = config["save_dir"]
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = f"{save_dir}/{current_time}"

    if not os.path.isdir(save_path):
        print(save_path)
        os.makedirs(f"{save_path}/weights")
        config["save_dir"] = save_path

    save_config_to_yaml(config, save_path)

    max_dice = 0.0 
    for ep in range(1, 1+config["epochs"]):
        train_loss_meter.reset()
        intersection_meter.reset()
        union_meter.reset()
        target_meter.reset()
        model.train()

        for batch_id, (x, y) in enumerate(tqdm(train_dataloader), start=1):
            if ep <= 1:
                optimizer.param_groups[0]["lr"] = (ep * batch_id) / (1.0 * total_steps) * config["init_lr"]
            else:
                scheduler.step()
            optimizer.zero_grad()
            n = x.shape[0]
            x = x.to(device).float()
            y = y.to(device).long()
            y_hat = model.forward_dummy(x) #(B, C, H, W)
            loss = criterion(y_hat, y) #(B, C, H, W) >< (B, H, W)
            loss.backward()
            optimizer.step()

            #save metrics
            with torch.no_grad():
                train_loss_meter.update(loss.item())
                y_hat_mask = y_hat.argmax(dim=1).squeeze(1) # (B, C, H, W) -> (B, 1, H, W) -> (B, H, W)
                intersection, union, target = intersectionAndUnionGPU(y_hat_mask.float(), y.float(), 3)
                intersection_meter.update(intersection)
                union_meter.update(union)
                target_meter.update(target)

        #compute iou, dice
        with torch.no_grad():
            iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) #vector 3D
            dice_class = (2 * intersection_meter.sum) / (intersection_meter.sum + union_meter.sum + 1e-10) #vector 3D

            mIoU = torch.mean(iou_class[1:]) #mean iou class 1 and class 2
            mDice = torch.mean(dice_class[1:]) #mean dice class 1 and class 2

        print(f"EP : {ep}, current_lr = {scheduler.get_last_lr()}, train_loss = {train_loss_meter.avg:.4f}, IoU = {mIoU:.4f}, dice = {mDice:.4f} \n")

        if mDice > max_dice:
            max_dice = mDice
            early_stopping_counter = 0
            torch.save(model.state_dict(), f"{save_path}/weights/best_dice_ckpt_ep_{ep}.pth")
        else:
            early_stopping_counter += 1

        # Check for early stopping
        if early_stopping_counter >= config["early_stop"]:
            print("Early stopping triggered")
            break