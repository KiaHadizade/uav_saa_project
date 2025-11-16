'''
Saves the best weights
'''

import os, argparse, yaml, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from depth_unet import UNetSmall
from utils import DepthDataset
import numpy as np

def train_one_epoch(model, loader, optim, device, criterion):
    model.train()
    running = 0.0
    for imgs, masks in tqdm(loader, desc="train"):
        imgs = imgs.to(device); masks = masks.to(device)
        logits = model(imgs)  # [B,C,H,W]
        loss = criterion(logits, masks)
        optim.zero_grad(); loss.backward(); optim.step()
        running += loss.item()
    return running/len(loader)

def val_one_epoch(model, loader, device, criterion):
    model.eval()
    running = 0.0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="val"):
            imgs = imgs.to(device); masks = masks.to(device)
            logits = model(imgs)
            loss = criterion(logits, masks)
            running += loss.item()
    return running/len(loader)

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = DepthDataset(cfg['data']['images_dir'], cfg['data']['masks_dir'], input_size=cfg['data']['input_size'])
    n = len(ds)
    nval = max(1, int(0.1*n))
    train_ds = torch.utils.data.Subset(ds, list(range(0, n-nval)))
    val_ds = torch.utils.data.Subset(ds, list(range(n-nval, n)))
    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    
    cfg['train']['lr'] = float(cfg['train']['lr'])
    cfg['train']['weight_decay'] = float(cfg['train']['weight_decay'])

    model = UNetSmall(n_classes=cfg['model']['n_classes'], base=cfg['model']['base_ch']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    best_val = 1e9
    os.makedirs('checkpoints', exist_ok=True)
    for epoch in range(cfg['train']['epochs']):
        print(f"Epoch {epoch+1}/{cfg['train']['epochs']}")
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_loss = val_one_epoch(model, val_loader, device, criterion)
        print(f"Train loss: {tr_loss:.4f} | Val loss: {val_loss:.4f} | time: {time.time()-t0:.1f}s")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f'checkpoints/depth_best.pth')
            print("Saved best model.")
    print("Training finished. Best val:", best_val)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="configs/default.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    main(cfg)
