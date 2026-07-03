"""ALOHA 시뮬 유틸 — 2개 작업 전용 최소본 (물체 초기 포즈 샘플링)."""
import numpy as np


def sample_box_pose():
    """transfer_cube: 큐브 초기 포즈 [xyz + quat(wxyz)]"""
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]
    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    """insertion: (peg 포즈, socket 포즈)"""
    ranges = np.vstack([[0.1, 0.2], [0.4, 0.6], [0.05, 0.05]])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    peg_pose = np.concatenate([peg_position, np.array([1, 0, 0, 0])])

    ranges = np.vstack([[-0.2, -0.1], [0.4, 0.6], [0.05, 0.05]])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    socket_pose = np.concatenate([socket_position, np.array([1, 0, 0, 0])])
    return peg_pose, socket_pose
