# UAV Sense & Avoid - Minimal Implementation

### Requirements
`pip install -r requirements.txt`

### 1. Prepare data
- Put frames into `data/images/`
- Put annotation JSONs into `data/annotations/` (see generate_masks.py format)
- Run: `python generate_masks.py --images data/images --ann data/annotations --masks data/masks`

### 2. Train depth model
`python train_depth.py --cfg configs/default.yaml`

### 3. Run inference (camera or video)
`python inference.py --cfg configs/default.yaml`

Notes:
- Detection uses `yolov8n.pt` (ultralytics). Replace with your trained detector optionally.
- Tracker is simple; for robust tracking use DeepSORT/StrongSORT.
