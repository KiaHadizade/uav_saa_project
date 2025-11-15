"""
Every JSON file in data/annotations contains:
{
  "image_name": "frame0001.jpg",
  "objects": [
      {"bbox":[x1,y1,x2,y2], "distance_m": 123.4},
      ...
  ]
}

Generates class-labeled masks 0..4 from the given JSON annotation
"""

import os, json, argparse
import cv2
import numpy as np

def distance_to_class(d):
    if d < 200: return 0
    if d < 400: return 1
    if d < 600: return 2
    if d < 700: return 3
    return 4

def main(images_dir, ann_dir, masks_dir, blur_ksize=5):
    os.makedirs(masks_dir, exist_ok=True)
    anns = [f for f in os.listdir(ann_dir) if f.endswith('.json')]
    for a in sorted(anns):
        path = os.path.join(ann_dir, a)
        ann = json.load(open(path))
        img_name = ann.get('image_name')
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            print("WARN: image not found", img_path); continue
        img = cv2.imread(img_path)
        H,W = img.shape[:2]
        mask = np.full((H,W), 4, dtype=np.uint8)  # background class = 4
        for obj in ann.get('objects', []):
            x1,y1,x2,y2 = map(int, obj['bbox'])
            cls = distance_to_class(float(obj.get('distance_m', 1000)))
            x1 = max(0,min(W-1,x1)); x2 = max(0,min(W,x2))
            y1 = max(0,min(H-1,y1)); y2 = max(0,min(H,y2))
            if x2>x1 and y2>y1:
                mask[y1:y2, x1:x2] = cls
        if blur_ksize and blur_ksize%2==1:
            mask = cv2.GaussianBlur(mask.astype(np.float32), (blur_ksize,blur_ksize), 0)
            mask = np.round(mask).astype(np.uint8)
        outp = os.path.join(masks_dir, img_name.replace('.jpg','.png'))
        cv2.imwrite(outp, mask)
        print("Saved:", outp)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", default="data/images")
    p.add_argument("--ann", default="data/annotations")
    p.add_argument("--masks", default="data/masks")
    args = p.parse_args()
    main(args.images, args.ann, args.masks)
