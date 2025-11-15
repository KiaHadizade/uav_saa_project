'''
Helper functions: simple data loader, preprocessing, sliding-window aggregation
'''

import os, glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DepthDataset(Dataset):
    def __init__(self, images_dir, masks_dir, input_size=320, augment=False):
        self.images = sorted(glob.glob(os.path.join(images_dir, '*.jpg')) + glob.glob(os.path.join(images_dir,'*.png')))
        self.masks = {}
        for p in sorted(glob.glob(os.path.join(masks_dir,'*.png')) + glob.glob(os.path.join(masks_dir,'*.jpg'))):
            name = os.path.basename(p)
            self.masks[name] = p
        self.inp = input_size
        self.augment = augment

    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        imp = self.images[idx]
        name = os.path.basename(imp)
        img = cv2.imread(imp)[:,:,::-1]  # BGR->RGB
        H,W = img.shape[:2]
        # load mask if exists
        mpath = self.masks.get(name)
        if mpath and os.path.exists(mpath):
            mask = cv2.imread(mpath, cv2.IMREAD_UNCHANGED)
            if mask is None: mask = 4*np.ones((H,W),dtype=np.uint8)
        else:
            mask = 4*np.ones((H,W),dtype=np.uint8)
        # center crop or resize to square
        img = cv2.resize(img, (self.inp, self.inp), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.inp, self.inp), interpolation=cv2.INTER_NEAREST)
        img = img.astype('float32')/255.0
        img = img.transpose(2,0,1)
        return torch.from_numpy(img), torch.from_numpy(mask.astype('long'))

def aggregate_mask_prediction(pred_probs, method='mean'):
    """
    pred_probs: torch tensor (C,H,W) or numpy
    returns class integer 0..(C-1) via sliding-window mean/min/max on center region
    here as a quick heuristic: average over whole map then argmax
    """
    if isinstance(pred_probs, torch.Tensor): arr = pred_probs.detach().cpu().numpy()
    else: arr = pred_probs
    C = arr.shape[0]
    # simple: mean pooling -> class
    mean_vals = arr.reshape(C, -1).mean(axis=1)
    return int(mean_vals.argmax())

def visualize_overlay(frame, boxes, track_ids, depth_classes):
    # boxes: list of (x1,y1,x2,y2)
    for (b,tid,cls) in zip(boxes, track_ids, depth_classes):
        x1,y1,x2,y2 = b
        cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame, f"ID:{tid} Dclass:{cls}", (x1, max(10,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    return frame
