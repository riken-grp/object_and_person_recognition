#!/usr/bin/env python3
import numpy as np

import cv2
import message_filters
import rclpy
from cv_bridge import CvBridge
from rclpy import qos
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from recognition_msgs.msg import Objects

from .message_builders import ObjectMessageBuilder
from .mapper import Sensor3DMapper
from .yolov8 import YOLOv8


class DetectionAndTrackingNode(Node):
    def __init__(self, name="ObjectDetectionNode"):
        super().__init__(name)
        self.declare_parameter("model", "yolov8x.pt")
        self.declare_parameter("thresh", 0.7)
        self.declare_parameter("iou", 0.7)
        self.declare_parameter("track_config", "")

        model = self.get_parameter("model").value
        self.thresh = self.get_parameter("thresh").value
        self.iou = self.get_parameter("iou").value
        self.track_config = self.get_parameter("track_config").value
        self.bridge = CvBridge()
        self.mapper = Sensor3DMapper()

        self.yolo_proc = YOLOv8(model)

        self.message_builder = ObjectMessageBuilder(
            self.yolo_proc.yolo.model.names)

        self.pub = self.create_publisher(Objects, "out_topic", qos_profile=qos.qos_profile_system_default)
        if "vis_topic" != "":
            self.pub_vis = self.create_publisher(Image, "vis_topic", qos_profile=qos.qos_profile_system_default)
        else:
            self.pub_vis = None

        sub_img = message_filters.Subscriber(self, Image, "in_topic", qos_profile=qos.qos_profile_system_default)
        sub_depth = message_filters.Subscriber(self, Image, "in_depth_topic", qos_profile=qos.qos_profile_system_default)
        sub_cam = message_filters.Subscriber(self, CameraInfo, "cinfo_topic", qos_profile=qos.qos_profile_system_default)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [sub_img, sub_depth, sub_cam], 10, 0.01
        )
        self.ts.registerCallback(self.callback)

    def callback(self, imgmsg, dimgmsg, cinfo):
        img = self.bridge.imgmsg_to_cv2(imgmsg, "bgr8")
        depth_img = self.bridge.imgmsg_to_cv2(dimgmsg)
        mdimg = cv2.medianBlur(depth_img, ksize=5)

        detections = self.yolo_proc.track(img, conf=self.thresh, iou=self.iou, tracker=self.track_config)
        if len(detections) > 0 and len(detections[0]) < 7:
            return

        # detections: xyxy, id, conf, class
        detections = detections[detections[:, 4] > self.thresh]
        detections[:, 2] = detections[:, 2] - detections[:, 0]
        detections[:, 3] = detections[:, 3] - detections[:, 1]
        # detections: xywh, id, conf, class
        detections3d = self.mapper.get_3d_value(detections, mdimg, cinfo)

        # publish
        self.publish(imgmsg.header, detections3d, img)

    def publish(self, header, detections, img):
        msg, markers, oimg = self.message_builder.build_from_tracklets(
            header, detections, img, self.pub_vis is not None
        )

        if self.pub_vis is not None:
            mimg = self.bridge.cv2_to_imgmsg(oimg, "bgr8")
            mimg.header = header
            self.pub_vis.publish(mimg)

        self.pub.publish(msg)


def main():
    rclpy.init()
    node = DetectionAndTrackingNode()
    rclpy.spin(node)
    rclpy.shutdown()
