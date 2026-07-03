import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from utils.constants import SIM_TASK_CONFIGS, PUPPET_GRIPPER_POSITION_CLOSE, PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from env.ee_sim_env import make_ee_sim_env
from env.gripper_controller import GripperController

import IPython
e = IPython.embed


def _slerp_mid(q1, q2):
    """Slerp midpoint between two quaternions (length-4 array-likes).

    Linear quat interp distorts intermediate orientations; using a true
    slerp midpoint as an extra waypoint splits a single big rotation into
    two smaller ones, each with much smaller linear-interp error.
    """
    a = np.asarray(q1, dtype=np.float64)
    b = np.asarray(q2, dtype=np.float64)
    if np.dot(a, b) < 0:
        b = -b  # take shorter rotation path
    s = a + b
    return s / np.linalg.norm(s)


class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None
        
        # ✨ GripperController: 0~1 범위에서 보간
        self.gripper_controller = GripperController(
            open_duration=50, 
            close_duration=50
        )
        self.prev_left_gripper_cmd = 1.0
        self.prev_right_gripper_cmd = 1.0

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        """
        위치와 자세는 선형 보간
        그리퍼는 이진 명령만 추출 (0 또는 1)
        """
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        # ✨ CHANGED: gripper는 보간하지 않고 이진값만 추출
        curr_grip = 1.0 if curr_waypoint['gripper'] > 0.5 else 0.0
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = 1.0 if next_waypoint['gripper'] > 0.5 else 0.0
        
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        # ✨ CHANGED: gripper는 이진 명령만 (현재 방향점의 값)
        gripper = curr_grip
        
        return xyz, quat, gripper

    def get_current_binary_states(self):
        """
        현재 이진 그리퍼 명령 상태 반환 (ee_sim_env에 설정하기 위함)
        """
        return (self.last_left_gripper_cmd, self.last_right_gripper_cmd)

    def __call__(self, ts):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)
            self.last_left_gripper_cmd = 1.0
            self.last_right_gripper_cmd = 1.0
            # ✨ GripperController 초기화 (열린 상태로)
            self.gripper_controller.reset()
            self.prev_left_gripper_cmd = 1.0
            self.prev_right_gripper_cmd = 1.0

        # obtain left and right waypoints
        if self.left_trajectory[0]['t'] == self.step_count:
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left_waypoint = self.left_trajectory[0]

        if self.right_trajectory[0]['t'] == self.step_count:
            self.curr_right_waypoint = self.right_trajectory.pop(0)
        next_right_waypoint = self.right_trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        left_xyz, left_quat, left_gripper_cmd = self.interpolate(self.curr_left_waypoint, next_left_waypoint, self.step_count)
        right_xyz, right_quat, right_gripper_cmd = self.interpolate(self.curr_right_waypoint, next_right_waypoint, self.step_count)
        
        # ✨ 이진 명령 저장 (ee_sim_env에 설정하기 위함)
        self.last_left_gripper_cmd = left_gripper_cmd
        self.last_right_gripper_cmd = right_gripper_cmd
        
        # ✨ GripperController로 보간: 0~1 범위에서 smooth 값 생성
        left_gripper_smooth = self.gripper_controller.process_gripper(
            'left', self.prev_left_gripper_cmd, left_gripper_cmd, self.step_count
        )
        right_gripper_smooth = self.gripper_controller.process_gripper(
            'right', self.prev_right_gripper_cmd, right_gripper_cmd, self.step_count
        )
        self.prev_left_gripper_cmd = left_gripper_cmd
        self.prev_right_gripper_cmd = right_gripper_cmd

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        # ✨ ee action에는 보간된 값 (smooth)
        action_left = np.concatenate([left_xyz, left_quat, [left_gripper_smooth]])
        action_right = np.concatenate([right_xyz, right_quat, [right_gripper_smooth]])

        self.step_count += 1
        return np.concatenate([action_left, action_right])


class PickAndTransferPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        box_info = np.array(ts_first.observation['env_state'])
        box_xyz = box_info[:3]
        box_quat = box_info[3:]
        # print(f"Generate trajectory for {box_xyz=}")

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)

        meet_xyz = np.array([0.05, 0.5, 0.25])

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1}, # sleep (open)
            {"t": 100, "xyz": meet_xyz + np.array([-0.08, 0, 0.01]), "quat": meet_left_quat.elements, "gripper": 1}, # approach meet position
            {"t": 260, "xyz": meet_xyz + np.array([0.025, 0, 0.01]), "quat": meet_left_quat.elements, "gripper": 0}, # move to meet position
            {"t": 310, "xyz": meet_xyz + np.array([0.025, 0, 0.01]), "quat": meet_left_quat.elements, "gripper": 0}, # close gripper
            {"t": 360, "xyz": meet_xyz + np.array([-0.1, 0, 0.01]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # move left
            {"t": 400, "xyz": meet_xyz + np.array([-0.1, 0, 0.01]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # stay
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1}, # sleep (open)
            {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1}, # approach the cube
            {"t": 130, "xyz": box_xyz + np.array([0.005, 0, -0.025]), "quat": gripper_pick_quat.elements, "gripper": 0}, # go down
            {"t": 170, "xyz": box_xyz + np.array([0.005, 0, -0.025]), "quat": gripper_pick_quat.elements, "gripper": 0}, # close gripper
            {"t": 200, "xyz": meet_xyz + np.array([0.06, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 0}, # approach meet position
            {"t": 220, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 1}, # move to meet position
            {"t": 310, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 1}, # open gripper
            {"t": 360, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # move to right
            {"t": 400, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # stay
        ]

class InsertionPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        peg_info = np.array(ts_first.observation['env_state'])[:7]
        peg_xyz = peg_info[:3]
        peg_quat = peg_info[3:]

        socket_info = np.array(ts_first.observation['env_state'])[7:]
        socket_xyz = socket_info[:3]
        socket_quat = socket_info[3:]

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        gripper_pick_quat_left = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)

        meet_xyz = np.array([0, 0.5, 0.2])
        lift_right = 0.01215

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1}, # sleep (open)
            {"t": 120, "xyz": socket_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # approach the cube
            {"t": 170, "xyz": socket_xyz + np.array([-0.01, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # go down
            {"t": 220, "xyz": socket_xyz + np.array([-0.01, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([-0.1, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([-0.04, 0, 0]), "quat": gripper_pick_quat_left.elements,"gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([-0.04, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},  # insertion
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1}, # sleep (open)
            {"t": 120, "xyz": peg_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # approach the cube
            {"t": 170, "xyz": peg_xyz + np.array([0.01, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # go down
            {"t": 220, "xyz": peg_xyz + np.array([0.01, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([0.1, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([0.04, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([0.04, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion

        ]
