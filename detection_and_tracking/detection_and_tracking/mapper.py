# coding: utf-8

import math

import numpy as np
import rclpy

from . import configs as cfg

_EPS = np.finfo(float).eps * 4.0


class Sensor3DMapper:
    def __init__(self):
        pass

    def translation_matrix(self, direction):
        M = np.identity(4)
        M[:3, 3] = direction[:3]
        return M

    def quaternion_matrix(self, quaternion):
        q = np.array(quaternion[:4], dtype=np.float64, copy=True)
        nq = np.dot(q, q)
        if nq < _EPS:
            return np.identity(4)
        q *= math.sqrt(2.0 / nq)
        q = np.outer(q, q)
        return np.array((
            (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], 0.0),
            (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], 0.0),
            (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], 0.0),
            (0.0, 0.0, 0.0, 1.0)
        ), dtype=np.float64)

    def get_3d_value(self, detections, dimg, cinfo):
        # image to sensor coordinate
        points_3d = self.get_bb_to_center_3d(detections, dimg, cinfo)
        dets = np.hstack((detections, points_3d))
        return dets

    def get_3d_pose(self, poses, dimg, cinfo):
        if len(poses) == 0:
            return poses

        n_persons = len(poses)
        points = poses.reshape(-1, 3)

        points_3d, flags = self.get_3d(points, dimg, cinfo)
        points_3d = points_3d.reshape((n_persons, -1, 3))
        flags = flags.reshape((n_persons, -1))

        poses = np.concatenate((poses[:, :,  :2], points_3d, poses[:, :, 2:3]), axis=2)

        return poses

    def get_3d_face(self, associated, dimg, cinfo):
        if len(associated.persons) == 0:
            return associated

        n_persons = len(associated.persons)
        fflag = [False] * n_persons
        bbs = np.zeros((n_persons, 4))
        for i, person in enumerate(associated.persons):
            if len(person.face_location) != 4:
                continue
            floc = np.asarray(person.face_location).reshape((-1, 4))
            bbs[i] = floc
            fflag[i] = True

        fcenter = np.zeros((n_persons, 2))  # center (x, y)
        fcenter[:, 0] = bbs[:, 0] + bbs[:, 2] / 2
        fcenter[:, 1] = bbs[:, 1] + bbs[:, 3] / 2

        fcenter_3d, flags = self.get_3d(fcenter, dimg, cinfo)
        fcenter_3d_global = self.project_to_map(fcenter_3d, cinfo)

        for i, person in enumerate(associated.persons):
            if fflag[i] and flags[i]:
                person.face_location_sensor = fcenter_3d[i, :3].tolist()
                person.face_location_global = fcenter_3d_global[i, :3].tolist()

        return associated

    def get_bb_to_center_3d(self, dets, dimg, cinfo):
        cx = dets[:, 0] + dets[:, 2] // 2
        cy = dets[:, 1] + dets[:, 3] // 2
        points = np.vstack((cx, cy)).T
        points_3d, flags = self.get_3d(points, dimg, cinfo)

        return points_3d

    def get_3d(self, points, depth_img, dinfo):
        # image (x,y) to sensor (Xs,Ys,Zs)
        # return (Xs,Ys,Zs), flags
        points_pos = np.floor(points).astype(np.int64)

        idx = points_pos[:, 1] * depth_img.shape[1] + points_pos[:, 0]
        flag = idx >= 0
        flag[flag] = idx[flag] < (depth_img.shape[0] * depth_img.shape[1])

        depth_values = np.zeros((len(points_pos), 1))
        depth_values[flag] = depth_img.reshape(-1, 1)[idx[flag]]
        depth_values /= 1000.0

        flag[(depth_values == 0).reshape((-1,))] = False

        xy = (points[:, :2] - np.array([dinfo.k[2], dinfo.k[5]])) / np.array(
            [dinfo.k[0], dinfo.k[4]]
        )
        xy = xy * depth_values

        return np.hstack((xy, depth_values)), flag

