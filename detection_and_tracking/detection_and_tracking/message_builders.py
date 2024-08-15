import cv2
import numpy as np
import rclpy
from recognition_msgs.msg import Object, Objects, Person, Persons, Pose
from visualization_msgs.msg import Marker, MarkerArray

from . import configs as cfg


class MessageBuilder():
    def __init__(self):
        pass

    def visualize_image(self, oimg, x, y, w, h, name, cls_id):
        c = cfg.colors[cls_id % len(cfg.colors)].tolist()
        x, y, w, h = map(int, (x, y, w, h))
        oimg = cv2.rectangle(oimg, (x, y), (x + w, y + h), c, 2)
        oimg = cv2.putText(
            oimg,
            name,
            (x, y),
            cv2.FONT_HERSHEY_PLAIN,
            4,
            c,
            2,
            cv2.LINE_AA,
        )
        return oimg


class ObjectMessageBuilder(MessageBuilder):
    def __init__(self, names):
        super().__init__()
        self.names = names
        self.colors = cfg.colors

    def build_from_tracklets(self, header, tracklets, img, is_visualize):
        msg = Objects()
        msg.header = header
        oimg = None

        if is_visualize:
            oimg = np.copy(img)

        m_objects = []
        m_markers = MarkerArray()
        for t in tracklets:
            x, y, w, h = t[:4]
            lo_sensor = t[cfg.LOCATION_SENSOR_COLUMN]
            cls_id = int(t[cfg.CLASS_ID_COLUMN])
            conf = t[cfg.CONFIDENCE_COLUMN]
            instance_id = int(t[cfg.INSTANCE_ID_COLUMN])

            om = self.build_object_2d(msg.header, x, y, w, h, conf, cls_id)
            om.instance_id = instance_id
            om.location_sensor = lo_sensor.tolist()
            m_objects.append(om)

            if is_visualize:
                oimg = self.visualize_image(oimg, x, y, w, h, om.name, cls_id)

        msg.objects = m_objects

        return msg, m_markers, oimg

    def set_object_info(self, header, cls_id, location, conf):
        om = Object()
        om.header = header
        om.name = self.names[cls_id]
        om.class_id = cls_id
        om.instance_id = -1
        om.location = list(map(float, location))
        om.confidence = float(conf)
        return om

    def build_object_2d(self, header, x, y, w, h, conf, cls_id):
        om = self.set_object_info(header, cls_id, [x, y, w, h], conf)
        return om

    def build_from_detections(self, header, loc, detections, img, is_visualize):
        msg = Objects()
        msg.header = header
        msg.pose = loc.tolist()
        oimg = None

        if is_visualize:
            oimg = np.copy(img)

        m_objects = []
        for p in detections:
            x, y, w, h, conf, cls_id = p
            cls_id = int(cls_id)

            om = self.build_object_2d(msg.header, x, y, w, h, conf, cls_id)
            m_objects.append(om)

            if is_visualize:
                oimg = self.visualize_image(oimg, x, y, w, h, om.name, cls_id)

        msg.objects = m_objects

        return msg, oimg

    def set_marker(self, header, x, y, z, cls_id):
        mm = Marker()
        mm.header = header
        mm.lifetime = rclpy.duration.Duration(seconds=0.1).to_msg()
        mm.type = Marker.SPHERE
        mm.id = cls_id
        c = cfg.colors[cls_id % len(cfg.colors)] / 255
        mm.color.r, mm.color.b, mm.color.g, mm.color.a = c[0], c[1], c[2], 1.0
        mm.scale.x, mm.scale.y, mm.scale.z = 0.1, 0.1, 0.1
        mm.pose.position.x, mm.pose.position.y, mm.pose.position.z = x, y, z
        mm.pose.orientation.w = 1.0
        mm.pose.orientation.x = 0.0
        mm.pose.orientation.y = 0.0
        mm.pose.orientation.z = 0.0

        return mm


class PersonMessageBuilder(MessageBuilder):
    def __init__(self):
        super().__init__()

    def build_from_tracklets(self, header, detections3d, poses3d, img, is_visualize):
        msg = Persons()
        mmsg = MarkerArray()
        msg.header = header
        oimg = None

        if is_visualize:
            oimg = np.copy(img)

        for instance_id, (det, pose) in enumerate(zip(detections3d, poses3d)):
            lo_image = det[:4].tolist()
            lo_sensor = det[cfg.POSE_LOCATION_SENSOR_COLUMN].tolist()
            conf = det[cfg.POSE_CONFIDENCE_COLUMN]
            cls_id = int(det[cfg.POSE_CLASS_ID_COLUMN])
            pm = self.build_person_2d(header, lo_image, conf, cls_id)
            pm.instance_id = -1
            pm.body_location_sensor = lo_sensor

            ppm = self.build_pose_2d(pose)
            ppm3 = self.build_pose_3d(pose)

            pm.pose = ppm
            pm.pose_sensor = ppm3

            pp11 = pose[11, cfg.POSE_SENSOR_COLUMN]
            pp12 = pose[12, cfg.POSE_SENSOR_COLUMN]
            x, y, z = (pp11 + pp12) / 2
            pmm = self.set_marker(header, x, y, z, instance_id)
            msg.persons.append(pm)
            mmsg.markers.append(pmm)

            if is_visualize:
                x, y, w, h = det[:4]
                oimg = self.visualize_image(oimg, x, y, w, h, "person", 0)
                oimg = self.pose_visualize(oimg, pose)

        return msg, mmsg, oimg

    def build_pose_2d(self, pose):
        pm = Pose()
        pm.dim = 2
        pm.keypoints = pose[:, cfg.POSE_IMAGE_COLUMN].reshape((-1, )).tolist()
        pm.confidences = pose[:, cfg.POSE_IMAGE_CONF_COLUMN].reshape((-1, )).tolist()

        return pm

    def build_pose_3d(self, pose):
        pm = Pose()
        pm.dim = 3
        pm.keypoints = pose[:, cfg.POSE_SENSOR_COLUMN].reshape((-1, )).tolist()
        pm.confidences = pose[:, cfg.POSE_IMAGE_CONF_COLUMN].reshape((-1, )).tolist()

        return pm

    def pose_visualize(self, show_img, pose):
        c = (0, 255, 255)
        pdata = pose[:, :2]
        confs = pose[:, 5]

        for d, c in zip(pdata, confs):
            kp = list(map(int, d[:2]))
            if c > 0:
                show_img = cv2.circle(show_img, kp, 4, (255, 255, 0), -1)

        for b in cfg.bones:
            show_img = self.draw_part_line(
                show_img, pdata, confs, b[0], b[1])

        return show_img

    def set_marker(self, header, x, y, z, cls_id):
        mm = Marker()
        mm.header = header
        mm.lifetime = rclpy.duration.Duration(seconds=0.1).to_msg()
        mm.type = Marker.SPHERE
        mm.id = cls_id
        c = cfg.colors[cls_id % len(cfg.colors)] / 255
        mm.color.r, mm.color.b, mm.color.g, mm.color.a = c[0], c[1], c[2], 1.0
        mm.scale.x, mm.scale.y, mm.scale.z = 0.1, 0.6, 0.1
        mm.pose.position.x, mm.pose.position.y, mm.pose.position.z = x, y, z
        mm.pose.orientation.w = 1.0
        mm.pose.orientation.x = 0.0
        mm.pose.orientation.y = 0.0
        mm.pose.orientation.z = 0.0

        return mm

    def build_from_targets(self, header, loc, targets, img, is_visualize):
        msg = Persons()
        msg.header = header
        msg.pose = loc
        oimg = None

        if is_visualize:
            oimg = np.copy(img)

        m_persons = []
        for target_id, t in targets.items():
            body, pose, imgs = t.get_latest()

            # name_id, name = t.get_name()
            name = t.consensus_name
            name_id = t.name_id[name]
            pm = self.build_person_all(
                header, body, pose, [name_id, name])
            m_persons.append(pm)

            if is_visualize:
                bodyl = body.body_location
                if not np.any(np.isnan(bodyl)):
                    x, y, w, h = map(int, body.body_location)
                    oimg = self.visualize_image(
                        oimg, x, y, w, h, name, body.instance_id)

        msg.persons = m_persons

        return msg, oimg

    def set_person_info(self, header, cls_id, location, conf):
        pm = Person()
        pm.header = header
        pm.name = ""
        pm.class_id = -1
        pm.instance_id = -1
        pm.body_location = location
        return pm

    def build_person_2d(self, header, lo_image, conf, cls_id):
        pm = self.set_person_info(header, cls_id, lo_image, conf)
        return pm

    def set_person_info_all(self, header, body, pose, name):
        pm = Person()
        pm.header = header
        pm.name = name[1]
        pm.class_id = name[0]
        pm.instance_id = body.instance_id

        pm.body_location = body.body_location
        if pose is not None:
            pm.pose = pose.pose

        pm.body_location_sensor = body.body_location_sensor
        if pose is not None:
            pm.pose_sensor = pose.pose_sensor

        pm.body_location_global = body.body_location_global
        if pose is not None:
            pm.pose_global = pose.pose_global
        return pm

    def build_person_all(self, header, body, pose, name):
        om = self.set_person_info_all(header, body, pose, name)
        return om

    def draw_part_line(self, img, kps, conf, i, j):
        if conf[i] > 0 and conf[j] > 0 and kps[i][0] > 0 and kps[j][0] > 0:
            k1 = list(map(int, kps[i]))
            k2 = list(map(int, kps[j]))
            img = cv2.line(img, k1, k2, (255, 255, 255), 3)
        return img
