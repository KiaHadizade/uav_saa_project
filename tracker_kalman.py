'''
Each track includes a bounding box and a Kalman filter for the center

Note: This tracker is very simple and is suitable for initial experimentation.
If you need reliable ID stability, use StrongSORT or deep_sort_realtime
'''

import numpy as np
from scipy.optimize import linear_sum_assignment

class SimpleTrack:
    def __init__(self, bbox, tid):
        self.bbox = bbox  # x1,y1,x2,y2
        self.tid = tid
        self.age = 0
        self.missed = 0

class SimpleTracker:
    def __init__(self, iou_thres=0.3, max_missed=10):
        self.tracks = []
        self.next_id = 0
        self.iou_thres = iou_thres
        self.max_missed = max_missed

    @staticmethod
    def iou(b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2-x1)*max(0, y2-y1)
        a1 = (b1[2]-b1[0])*(b1[3]-b1[1]); a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
        union = a1 + a2 - inter
        return inter/union if union>0 else 0.0

    def update(self, detections):
        # detections: list of bbox tuples
        N = len(self.tracks); M = len(detections)
        if N==0:
            for d in detections:
                self.tracks.append(SimpleTrack(d, self.next_id)); self.next_id+=1
            return [t.tid for t in self.tracks], [t.bbox for t in self.tracks]

        # cost matrix (1 - iou)
        cost = np.ones((N,M), dtype=np.float32)
        for i,t in enumerate(self.tracks):
            for j,d in enumerate(detections):
                cost[i,j] = 1.0 - self.iou(t.bbox, d)
        row_ind, col_ind = linear_sum_assignment(cost)
        assigned_tracks = set()
        assigned_det = set()
        # update assigned
        for r,c in zip(row_ind, col_ind):
            if cost[r,c] < (1.0 - self.iou_thres):
                self.tracks[r].bbox = detections[c]
                self.tracks[r].missed = 0
                assigned_tracks.add(r); assigned_det.add(c)
        # mark unassigned tracks
        for i,t in enumerate(self.tracks):
            if i not in assigned_tracks:
                t.missed += 1
        # add new tracks
        for j,d in enumerate(detections):
            if j not in assigned_det:
                self.tracks.append(SimpleTrack(d, self.next_id)); self.next_id+=1
        # remove dead tracks
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return [t.tid for t in self.tracks], [t.bbox for t in self.tracks]
