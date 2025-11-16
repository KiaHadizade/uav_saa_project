'''
Pipeline: detection via YOLOv8 → crop → depth model → tracker → visualization
'''

import cv2, torch, argparse, yaml, time, numpy as np
from ultralytics import YOLO
from depth_unet import UNetSmall
from utils import center_crop_around_bbox, aggregate_mask_prediction, visualize_overlay
from tracker_kalman import SimpleTracker

# helper (we reuse small crop fn here)
def center_crop_around_bbox(img, bbox, out_size=320):
    h,w = img.shape[:2]
    x1,y1,x2,y2 = bbox
    cx = (x1+x2)//2; cy = (y1+y2)//2
    half = out_size//2
    left = cx-half; top = cy-half; right = cx+half; bottom = cy+half
    pad_left = max(0, -left); pad_top = max(0, -top)
    pad_right = max(0, right-w); pad_bottom = max(0, bottom-h)
    img_p = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
    left += pad_left; top += pad_top
    crop = img_p[top:top+out_size, left:left+out_size]
    return crop

def preprocess_crop(crop, size=320):
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype('float32')/255.0
    img = cv2.resize(img, (size, size))
    img = img.transpose(2,0,1)
    return torch.from_numpy(img).unsqueeze(0)

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    det_model = YOLO('yolov8n.pt')  # Load detector (change to your detector weights if trained)

    # Load depth model
    depth = UNetSmall(
        n_classes=cfg['model']['n_classes'],
        base=cfg['model']['base_ch']
    ).to(device)
    depth.load_state_dict(torch.load('checkpoints/depth_best.pth', map_location=device))
    depth.eval()

    # Open camera or video file
    cap = cv2.VideoCapture(0)  # or path to video
    tracker = SimpleTracker()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame not received. Exiting...")
            break
        # detection
        results = det_model.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]
        boxes = []
        for box in results.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            boxes.append((x1,y1,x2,y2))
        depth_classes = []
        crops = [center_crop_around_bbox(frame, b, out_size=cfg['data']['input_size']) for b in boxes]
        if len(crops)>0:
            batch = torch.cat([preprocess_crop(c) for c in crops], dim=0).to(device)
            with torch.no_grad():
                logits = depth(batch)  # [B,C,H,W]
                probs = torch.softmax(logits, dim=1)
            for p in probs:
                cls = aggregate_mask_prediction(p)
                depth_classes.append(cls)
        else:
            depth_classes = []

        tids, tbboxes = tracker.update(boxes)
        # visualize: we must align classes with tracks - here we simply map by detection order (best-effort)
        # Create lists same length as tbboxes
        # mapping naive: assume order maintained; better would be matching IDs to det indices
        if len(tbboxes)>0 and len(depth_classes)>0:
            # naive align
            # ensure same length
            if len(depth_classes) < len(tbboxes):
                depth_classes += [4]*(len(tbboxes)-len(depth_classes))
            vis_classes = depth_classes[:len(tbboxes)]
        else:
            vis_classes = [4]*len(tbboxes)
        out = visualize_overlay(frame, tbboxes, tids, vis_classes)
        cv2.imshow("SAA", out)
        # Exit condition with ESC
        if cv2.waitKey(1) & 0xFF == 27:
            print("ESC pressed. Exiting...")
            break
        # Exit condition when the window is closed
        if cv2.getWindowProperty("SAA", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed. Exiting loop...")
            break
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="configs/default.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    main(cfg)
