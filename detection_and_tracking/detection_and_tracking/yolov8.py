# coding: utf-8 -*-

import numpy as np
import torch
from ultralytics import YOLO


class YOLOv8:
    def __init__(self, model):
        self.yolo = YOLO(model)
        self.yolo.to("cuda")

    def __call__(self, img):
        img = np.ascontiguousarray(img)
        results = self.yolo(img, verbose=False)[0].boxes

        # xyxy, conf, class
        return results.boxes.data

    def track(self, img, conf=0.3, iou=0.5, tracker=None):
        img = np.ascontiguousarray(img)
        results = self.yolo.track(img, verbose=False, conf=conf, iou=iou, persist=True, tracker=tracker)[0]

        # xyxy, id, conf, class
        return results.boxes.data.cpu().numpy()

    def pose(self, img, conf=0.3):
        img = np.ascontiguousarray(img)
        results = self.yolo(img, verbose=False, conf=conf)[0]

        bbs = results.boxes.data.cpu().numpy()
        poses = results.keypoints.data.cpu().numpy()
        poses = poses.reshape((-1, 17, 3))

        return bbs, poses
        # bbs: xyxy, conf, class
        # poses: (xy,conf) * 17
