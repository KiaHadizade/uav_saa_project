'''
Pipeline: detection via YOLOv8 → crop → depth model → tracker → visualization
'''

import cv2, torch, argparse, yaml, time, numpy as np
from ultralytics import YOLO
from depth_unet import UNetSmall
from utils import center_crop_around_bbox, aggregate_mask_prediction, visualize_overlay
from tracker_kalman import SimpleTracker

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
    cap = cv2.VideoCapture(0)  # or path to video (VideoCapture("yourvideo.mp4"))

    # Prepare video writer (saved output)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = None

    tracker = SimpleTracker()

    fps = 0
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame not received. Exiting...")
            break

        # Initialize video writer when first frame arrives
        if out_writer is None:
            h, w = frame.shape[:2]
            out_writer = cv2.VideoWriter("output_processed.mp4", fourcc, 30, (w, h))

        start = time.time()

        # YOLO detection
        results = det_model.predict(frame, imgsz=640, conf=0.3, verbose=False)[0]
        boxes = []
        for box in results.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            boxes.append((x1,y1,x2,y2))

        # Depth estimation
        depth_classes = []
        crops = [center_crop_around_bbox(frame, b, cfg['data']['input_size']) for b in boxes]

        if len(crops) > 0:
            batch = torch.cat([preprocess_crop(c, cfg['data']['input_size']) for c in crops], dim=0).to(device)

            with torch.no_grad():
                logits = depth(batch)  # [B,C,H,W]
                probs = torch.softmax(logits, dim=1)

            for p in probs:
                cls = aggregate_mask_prediction(p)
                depth_classes.append(cls)
        else:
            depth_classes = []

        # Tracking
        tids, tracked_boxes  = tracker.update(boxes)

        # visualize: we must align classes with tracks - here we simply map by detection order (best-effort)
        # Create lists same length as tracked_boxes
        # mapping naive: assume order maintained; better would be matching IDs to det indices
        if len(tracked_boxes)>0 and len(depth_classes)>0:
            # naive align
            # ensure same length
            if len(depth_classes) < len(tracked_boxes):
                depth_classes += [4] * (len(tracked_boxes) - len(depth_classes))
            vis_classes = depth_classes[:len(tracked_boxes)]
        else:
            vis_classes = [4] * len(tracked_boxes)
        
        # Draw
        out_frame = visualize_overlay(frame.copy(), tracked_boxes, tids, vis_classes)

        # FPS + Latency overlay
        end = time.time()
        latency_ms = (end - start) * 1000
        fps = 1.0 / (end - last_time)
        last_time = end

        cv2.putText(out_frame, f"FPS: {fps:.1f}", (10, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,255,50), 2)
        cv2.putText(out_frame, f"Latency: {latency_ms:.1f} ms", (10, 55),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,255,50), 2)
        
        # Save processed video
        out_writer.write(out_frame)

        # Display
        cv2.imshow("SAA", out_frame)

        # Exit condition with ESC
        if cv2.waitKey(1) & 0xFF == 27:
            print("ESC pressed. Exiting...")
            break
        # Detect window close
        if cv2.getWindowProperty("SAA", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed. Exiting loop...")
            break

    # Cleanup
    cap.release()
    if out_writer is not None:
        out_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    main(cfg)
