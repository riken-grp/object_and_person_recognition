#!/usr/bin/env python
# coding: utf-8 -*-

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy import qos
from rclpy.node import Node
from recognition_msgs.msg import Objects, Persons
from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import MarkerArray

from . import configs as cfg
from .mapper import Sensor3DMapper
from .message_builders import PersonMessageBuilder
from .yolov8 import YOLOv8


class PoseProc(Node):
    def __init__(self, name="pose_estimation_node"):
        super().__init__(name)
        self.declare_parameter("model", "yolov8x-pose.pt")
        self.declare_parameter("thresh", 0.3)
        self.declare_parameter("world_frame_id", "map")

        model = self.get_parameter("model").value
        self.thresh = self.get_parameter("thresh").value
        world_frame_id = self.get_parameter("world_frame_id").value

        self.yolo_proc = YOLOv8(model)

        self.bridge = CvBridge()
        self.mapper = Sensor3DMapper()

        self.message_builder = PersonMessageBuilder()

        self.pub = self.create_publisher(Persons, "out_topic", qos_profile=qos.qos_profile_system_default)
        self.marker_pub = self.create_publisher(MarkerArray, "marker_topic", qos_profile=qos.qos_profile_system_default)
        if "vis_topic" != "":
            self.vis_pub = self.create_publisher(Image, "vis_topic", qos_profile=qos.qos_profile_system_default)
        else:
            self.vis_pub = None

        sub_img = message_filters.Subscriber(self, Image, "in_topic", qos_profile=qos.qos_profile_system_default)
        sub_depth = message_filters.Subscriber(self, Image, "in_depth_topic", qos_profile=qos.qos_profile_system_default)
        sub_cam = message_filters.Subscriber(self, CameraInfo, "cinfo_topic", qos_profile=qos.qos_profile_system_default)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [sub_img, sub_depth, sub_cam], 10, 0.1
        )
        self.ts.registerCallback(self.callback)

    def callback(self, imgmsg, depthmsg, cinfo):
        color_img = self.bridge.imgmsg_to_cv2(imgmsg, "bgr8")
        depth_img = self.bridge.imgmsg_to_cv2(depthmsg)
        mdimg = cv2.medianBlur(depth_img, ksize=5)

        # Pose Estimation
        detections, poses = self.yolo_proc.pose(color_img, conf=self.thresh)
        # detections: xyxy, conf, class
        detections = detections[detections[:, 4] > self.thresh]
        detections[:, 2] = detections[:, 2] - detections[:, 0]
        detections[:, 3] = detections[:, 3] - detections[:, 1]
        # detections: xywh, conf, class

        detections3d = self.mapper.get_3d_value(detections, mdimg, cinfo)
        poses3d = self.mapper.get_3d_pose(poses, mdimg, cinfo)

        # publish
        self.publish(imgmsg.header, detections3d, poses3d, color_img)

    def publish(self, header, detections3d, poses3d, color_img):
        msg, markers, show_img = self.message_builder.build_from_tracklets(header, detections3d, poses3d, color_img, self.vis_pub is not None)

        self.pub.publish(msg)
        self.marker_pub.publish(markers)
        if self.vis_pub is not None:
            imsg = self.bridge.cv2_to_imgmsg(show_img, "bgr8")
            self.vis_pub.publish(imsg)


def main():
    rclpy.init()
    node = PoseProc()
    rclpy.spin(node)
    rclpy.shutdown()
