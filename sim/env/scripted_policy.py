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


class CubeClassificationPolicy(BasePolicy):
    """Color-sort task policy composed from 4 atomic actions.

    Atomic actions:
      action_1 (pap_red_left)    : red on left side   - L picks and places at RED_GOAL
      action_2 (pap_blue_right)  : blue on right side - R picks and places at BLUE_GOAL
      action_3 (handover_red)    : red on right side  - R picks, L receives, L places
      action_4 (handover_blue)   : blue on left side  - L picks, R receives, R places

    Scenario composition (per spawn config; episode length varies):
      A (red.x<0, blue.x>0) : action_1 || action_2 (parallel)
      B (red.x>0, blue.x<0) : action_3 -> transition -> action_4
      C (red.x<0, blue.x<0) : action_1 -> transition -> action_4
      D (red.x>0, blue.x>0) : action_2 -> transition -> action_3

    Inter-action transitions pre-position both arms at the next action's natural
    starting pose (next picker above its target cube; next dropper at approach_off
    side of meet) instead of returning to init. This avoids passing through the
    arm's near-singularity init region and keeps quat changes small.

    self.episode_len is set per-scenario inside generate_trajectory; the recording
    script truncates the saved hdf5 to that length.
    """

    # World-frame placement targets (gripper xyz when releasing the cube).
    RED_GOAL_XYZ  = np.array([-0.10, 0.30, 0.075])
    BLUE_GOAL_XYZ = np.array([+0.10, 0.30, 0.075])
    # Hard-coded compensation: in scenario C the right-arm dropper consistently
    # lands ~3 cm toward the camera (-y) and ~3 cm toward the right arm (+x)
    # relative to the commanded goal because of accumulated joint-replay
    # tracking error from R's idle-then-active motion history. Pre-bias the
    # commanded goal in the opposite direction so the actual cube position
    # ends up on the blue plate.
    BLUE_GOAL_XYZ_C = np.array([+0.10 - 0.03, 0.30 + 0.03, 0.075])
    # Handover meeting point: well inside workspace (y=0.55) and high (z=0.35) so
    # the dropper has clearance for the wrist rotation.
    MEET_XYZ      = np.array([ 0.00, 0.55, 0.35])

    # Phase lengths. Release happens during descent (gripper opens at z = goal+0.01
    # for pap, z = goal+0.025 for handover) and the segment then dwells long
    # enough for the gripper to fully actuate (50 step open_duration in
    # GripperController) and for the cube to fall and settle (~5 step) before
    # the arm retracts.
    T_PAP_SEGMENT      = 440   # direct pick-and-place
    T_HANDOVER_SEGMENT = 760   # picker-to-dropper handover including final placement
    T_TRANSITION       = 150   # inter-action settle/rotation between sequential phases

    def __init__(self, inject_noise=False):
        super().__init__(inject_noise)
        # Conservative default; overwritten per-config in generate_trajectory.
        self.episode_len = 2 * self.T_HANDOVER_SEGMENT + self.T_TRANSITION

    # ------------------------------------------------------------------------
    # Top-level composer
    # ------------------------------------------------------------------------

    def generate_trajectory(self, ts_first):
        init_left  = ts_first.observation['mocap_pose_left']
        init_right = ts_first.observation['mocap_pose_right']
        env_state  = np.array(ts_first.observation['env_state'])

        red_xyz  = env_state[:3]
        blue_xyz = env_state[7:10]

        # Per-arm grasping quat: rotate init quat 60 deg about Y (top-down grip).
        gpqr = (Quaternion(init_right[3:]) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)).elements
        gpql = (Quaternion(init_left[3:])  * Quaternion(axis=[0.0, 1.0, 0.0], degrees= 60)).elements
        # Receive quat for the dropper: 90 deg about X (sideways grip).
        meet_quat_left  = Quaternion(axis=[1.0, 0.0, 0.0], degrees= 90).elements
        meet_quat_right = Quaternion(axis=[1.0, 0.0, 0.0], degrees=-90).elements

        # Save context for the atomic-action and transition methods.
        self._init_left, self._init_right = init_left, init_right
        self._red_xyz, self._blue_xyz     = red_xyz, blue_xyz
        self._gpql, self._gpqr            = gpql, gpqr
        self._meet_quat_left              = meet_quat_left
        self._meet_quat_right             = meet_quat_right

        # Both trajectories start at the env-provided init mocap pose (gripper open).
        self.left_trajectory = [
            {"t": 0, "xyz": init_left[:3],  "quat": init_left[3:],  "gripper": 1},
        ]
        self.right_trajectory = [
            {"t": 0, "xyz": init_right[:3], "quat": init_right[3:], "gripper": 1},
        ]

        red_left  = red_xyz[0]  < 0
        blue_left = blue_xyz[0] < 0

        Tp, Th, Tt = self.T_PAP_SEGMENT, self.T_HANDOVER_SEGMENT, self.T_TRANSITION

        # Track scenario for goal-coordinate selection (C uses a pre-biased
        # blue goal to compensate for measured Phase 2 placement offset).
        if red_left and not blue_left:
            self._scenario = 'A'
            self._action_pap_red_left(t0=0)
            self._action_pap_blue_right(t0=0)
            self.episode_len = Tp
        elif not red_left and blue_left:
            self._scenario = 'B'
            self._action_handover_red(t0=0)
            self._transition_for_handover_blue(t_arrive=Th + Tt)
            self._action_handover_blue(t0=Th + Tt)
            self.episode_len = 2 * Th + Tt
        elif red_left and blue_left:
            self._scenario = 'C'
            self._action_pap_red_left(t0=0)
            self._stay_at_init('right', self._init_right, t=Tp)
            self._transition_for_handover_blue(t_arrive=Tp + Tt)
            self._action_handover_blue(t0=Tp + Tt)
            self.episode_len = Tp + Tt + Th
        else:
            self._scenario = 'D'
            self._action_pap_blue_right(t0=0)
            self._stay_at_init('left', self._init_left, t=Tp)
            self._transition_for_handover_red(t_arrive=Tp + Tt)
            self._action_handover_red(t0=Tp + Tt)
            self.episode_len = Tp + Tt + Th

        self._extend_to_end()

    # ------------------------------------------------------------------------
    # Atomic actions
    # ------------------------------------------------------------------------

    def _action_pap_red_left(self, t0):
        self._add_pap_segment('left', t0, self._red_xyz, self._gpql, self.RED_GOAL_XYZ)

    def _action_pap_blue_right(self, t0):
        self._add_pap_segment('right', t0, self._blue_xyz, self._gpqr, self.BLUE_GOAL_XYZ)

    def _action_handover_red(self, t0):
        self._add_handover_segment(
            t0,
            picker='right', target_xyz=self._red_xyz, picker_quat=self._gpqr,
            dropper='left', dropper_quat=self._meet_quat_left,
            dropper_carry_quat=self._gpql,
            goal_xyz=self.RED_GOAL_XYZ,
        )

    def _action_handover_blue(self, t0):
        # In scenario C, use the pre-biased blue goal so the actual cube
        # position (after Phase 2 tracking error) ends up on the blue plate.
        blue_goal = self.BLUE_GOAL_XYZ_C if getattr(self, '_scenario', None) == 'C' else self.BLUE_GOAL_XYZ
        self._add_handover_segment(
            t0,
            picker='left', target_xyz=self._blue_xyz, picker_quat=self._gpql,
            dropper='right', dropper_quat=self._meet_quat_right,
            dropper_carry_quat=self._gpqr,
            goal_xyz=blue_goal,
        )

    # ------------------------------------------------------------------------
    # Inter-action transitions
    # ------------------------------------------------------------------------
    # Each transition adds two waypoints per arm: a slerp-midpoint waypoint at
    # half time (smaller per-segment quat rotation, less linear-interp distortion)
    # and the target pre-position waypoint at t_arrive.

    def _transition_for_handover_blue(self, t_arrive):
        """Pre-position for action_4: L above blue, R at right approach_off."""
        above_blue = self._blue_xyz + np.array([0.0, 0.0, 0.10])
        self._append_with_quat_mid('left', t_arrive, xyz=above_blue, target_quat=self._gpql)
        approach_off_r = np.array([+0.06 * 1.33, 0.0, 0.01])
        self._append_with_quat_mid(
            'right', t_arrive,
            xyz=self.MEET_XYZ + approach_off_r,
            target_quat=self._meet_quat_right,
        )

    def _transition_for_handover_red(self, t_arrive):
        """Pre-position for action_3: R above red, L at left approach_off."""
        above_red = self._red_xyz + np.array([0.0, 0.0, 0.10])
        self._append_with_quat_mid('right', t_arrive, xyz=above_red, target_quat=self._gpqr)
        approach_off_l = np.array([-0.06 * 1.33, 0.0, 0.01])
        self._append_with_quat_mid(
            'left', t_arrive,
            xyz=self.MEET_XYZ + approach_off_l,
            target_quat=self._meet_quat_left,
        )

    # ------------------------------------------------------------------------
    # Trajectory helpers
    # ------------------------------------------------------------------------

    def _traj(self, arm):
        return self.left_trajectory if arm == 'left' else self.right_trajectory

    def _stay_at_init(self, arm, init_pose, t):
        """Hold an idle arm at its init pose by appending a waypoint at time t."""
        traj = self._traj(arm)
        if traj[-1]['t'] < t:
            traj.append({"t": t, "xyz": init_pose[:3], "quat": init_pose[3:], "gripper": 1})

    def _append_with_quat_mid(self, arm, t_arrive, xyz, target_quat):
        """Append target waypoint plus a slerp midpoint at half time so each
        sub-segment carries roughly half of the rotation."""
        traj = self._traj(arm)
        last = traj[-1]
        if last['t'] >= t_arrive:
            return
        mid_t = last['t'] + (t_arrive - last['t']) // 2
        mid_q = _slerp_mid(last['quat'], target_quat)
        mid_xyz = (np.asarray(last['xyz']) + np.asarray(xyz)) * 0.5
        traj.append({"t": mid_t, "xyz": mid_xyz, "quat": mid_q, "gripper": 1})
        traj.append({"t": t_arrive, "xyz": xyz, "quat": target_quat, "gripper": 1})

    # Recording loops in record_sim_episodes.py iterate up to SIM_TASK_CONFIGS
    # ['sim_cube_classification_record']['episode_len'] (= 1700), which is
    # larger than every scenario's actual length. Pad trajectories well past
    # that bound so the policy's interpolate() never reads an empty waypoint
    # list. Saved hdf5 length is still controlled by the recorder's loop.
    _PAD_END_T = 2000

    def _extend_to_end(self):
        """Pad both trajectories well past episode_len so interpolation never runs out."""
        target_t = max(self.episode_len, self._PAD_END_T) + 1
        for arm in ('left', 'right'):
            traj = self._traj(arm)
            last = traj[-1]
            if last['t'] < target_t:
                traj.append({"t": target_t, "xyz": last['xyz'], "quat": last['quat'], "gripper": last['gripper']})

    # ------------------------------------------------------------------------
    # Trajectory primitives
    # ------------------------------------------------------------------------

    def _add_pap_segment(self, arm, t0, target_xyz, picker_quat, goal_xyz):
        """Direct pick-and-place segment, length T_PAP_SEGMENT (440 steps).

        Timing offsets relative to t0:
          100 : above target (pre-grasp)
          150 : descend to grasp depth
          170 : close gripper
          220 : grasp dwell
          270 : lift
          320 : above goal (z = goal+0.06)
          355 : descending, z = goal+0.025 (still closed)
          370 : descending, z = goal+0.01, gripper OPEN (release while descending)
          400 : at goal_xyz, gripper open
          425 : at goal_xyz, gripper fully open + cube settled (55 step dwell from open)
          440 : retract (open, 0.10 above goal)
        """
        traj = self._traj(arm)
        grasp_x = 0.01 if arm == 'right' else -0.01
        grasp_off = np.array([grasp_x, 0, -0.03])
        traj.extend([
            {"t": t0 + 100, "xyz": target_xyz + np.array([0, 0, 0.08]),  "quat": picker_quat, "gripper": 1},
            {"t": t0 + 150, "xyz": target_xyz + grasp_off,                "quat": picker_quat, "gripper": 1},
            {"t": t0 + 170, "xyz": target_xyz + grasp_off,                "quat": picker_quat, "gripper": 0},
            {"t": t0 + 220, "xyz": target_xyz + grasp_off,                "quat": picker_quat, "gripper": 0},
            {"t": t0 + 270, "xyz": target_xyz + np.array([0, 0, 0.10]),  "quat": picker_quat, "gripper": 0},
            {"t": t0 + 320, "xyz": goal_xyz   + np.array([0, 0, 0.06]),  "quat": picker_quat, "gripper": 0},
            {"t": t0 + 355, "xyz": goal_xyz   + np.array([0, 0, 0.025]), "quat": picker_quat, "gripper": 0},
            {"t": t0 + 370, "xyz": goal_xyz   + np.array([0, 0, 0.01]),  "quat": picker_quat, "gripper": 1},
            {"t": t0 + 400, "xyz": goal_xyz,                              "quat": picker_quat, "gripper": 1},
            {"t": t0 + 425, "xyz": goal_xyz,                              "quat": picker_quat, "gripper": 1},
            {"t": t0 + 440, "xyz": goal_xyz   + np.array([0, 0, 0.10]),  "quat": picker_quat, "gripper": 1},
        ])

    def _add_handover_segment(self, t0, picker, target_xyz, picker_quat, dropper,
                              dropper_quat, dropper_carry_quat, goal_xyz):
        """Picker-to-dropper handover plus final placement, length T_HANDOVER_SEGMENT (760 steps).

        Picker timing (relative to t0):
          100/150/170 : above / descend / close
          260         : grasp dwell complete (90 step dwell stabilises grip)
          310         : lift to z+0.10
          350         : approach side of meet (offset in x)
          390         : at meet center, holding cube
          510         : release (after dropper has closed and settled)
          560         : retract to picker side
          760         : hold (pad until segment end)

        Dropper timing (relative to t0):
          240/390     : align at approach_off, wait for picker arrival
          420/440     : approach finger_meet, close gripper
          500         : close-settle complete (60 step joint settle)
          535         : slerp midpoint of wrist rotation (sideways -> top-down)
          570         : rotation complete (top-down carry quat)
          605         : (goal_x, goal_y, 0.20)
          635         : (goal_x, goal_y, 0.12)
          660         : (goal_x, goal_y, goal_z+0.025), gripper OPEN (release while descending)
          690         : at goal_xyz, gripper open (continued slight descent)
          730         : at goal_xyz, gripper fully open + cube settled (40 step dwell)
          760         : retract to goal+0.10
        """
        ptraj = self._traj(picker)
        dtraj = self._traj(dropper)
        meet  = self.MEET_XYZ
        pick_off_x = +0.06 if picker == 'right' else -0.06
        drop_off_x = -0.06 if dropper == 'left' else +0.06
        grasp_x   = 0.01 if picker == 'right' else -0.01
        grasp_off = np.array([grasp_x, 0, -0.03])
        approach_off = np.array([drop_off_x * 1.33, 0, 0.01])
        finger_meet  = np.array([-drop_off_x * 0.42, 0, 0.01])

        ptraj.extend([
            {"t": t0 + 100, "xyz": target_xyz + np.array([0, 0, 0.08]),     "quat": picker_quat, "gripper": 1},
            {"t": t0 + 150, "xyz": target_xyz + grasp_off,                   "quat": picker_quat, "gripper": 1},
            {"t": t0 + 170, "xyz": target_xyz + grasp_off,                   "quat": picker_quat, "gripper": 0},
            {"t": t0 + 260, "xyz": target_xyz + grasp_off,                   "quat": picker_quat, "gripper": 0},
            {"t": t0 + 310, "xyz": target_xyz + np.array([0, 0, 0.10]),     "quat": picker_quat, "gripper": 0},
            {"t": t0 + 350, "xyz": meet + np.array([pick_off_x, 0, 0]),      "quat": picker_quat, "gripper": 0},
            {"t": t0 + 390, "xyz": meet,                                      "quat": picker_quat, "gripper": 0},
            {"t": t0 + 510, "xyz": meet,                                      "quat": picker_quat, "gripper": 1},
            {"t": t0 + 560, "xyz": meet + np.array([pick_off_x * 2, 0, 0]),  "quat": picker_quat, "gripper": 1},
            {"t": t0 + 760, "xyz": meet + np.array([pick_off_x * 2, 0, 0]),  "quat": picker_quat, "gripper": 1},
        ])

        dtraj.extend([
            {"t": t0 + 240, "xyz": meet + approach_off, "quat": dropper_quat, "gripper": 1},
            {"t": t0 + 390, "xyz": meet + approach_off, "quat": dropper_quat, "gripper": 1},
            {"t": t0 + 420, "xyz": meet + finger_meet,  "quat": dropper_quat, "gripper": 1},
            {"t": t0 + 440, "xyz": meet + finger_meet,  "quat": dropper_quat, "gripper": 0},
            {"t": t0 + 500, "xyz": meet + finger_meet,  "quat": dropper_quat, "gripper": 0},
            # Wrist rotation 70 step with slerp midpoint -> per-segment ~52 deg.
            {"t": t0 + 535, "xyz": meet + finger_meet,  "quat": _slerp_mid(dropper_quat, dropper_carry_quat), "gripper": 0},
            {"t": t0 + 570, "xyz": meet + finger_meet,  "quat": dropper_carry_quat, "gripper": 0},
            # Descend toward goal area while holding the carry quat. Release happens
            # mid-descent (gripper opens at z = goal+0.025); the arm then dwells
            # at goal_xyz long enough for the gripper to fully open (50 step) and
            # the cube to fall and settle before retracting.
            {"t": t0 + 605, "xyz": np.array([goal_xyz[0], goal_xyz[1], 0.20]), "quat": dropper_carry_quat, "gripper": 0},
            {"t": t0 + 635, "xyz": np.array([goal_xyz[0], goal_xyz[1], 0.12]), "quat": dropper_carry_quat, "gripper": 0},
            {"t": t0 + 660, "xyz": goal_xyz + np.array([0, 0, 0.025]),          "quat": dropper_carry_quat, "gripper": 1},
            {"t": t0 + 690, "xyz": goal_xyz,                                     "quat": dropper_carry_quat, "gripper": 1},
            {"t": t0 + 730, "xyz": goal_xyz,                                     "quat": dropper_carry_quat, "gripper": 1},
            {"t": t0 + 760, "xyz": goal_xyz + np.array([0, 0, 0.10]),           "quat": dropper_carry_quat, "gripper": 1},
        ])


def _safe_grasp_z_offset(obj_z, obj_half_height, default_offset=-0.03):
    """Compute grasp Z-offset that avoids platform collision.
    For elevated objects, the gripper can't go below the platform top.
    For table-level objects, use the default offset."""
    elevation = obj_z - 0.05
    if elevation > 0.005:
        # Object on platform: gripper goes to just at object bottom
        return -(obj_half_height - 0.003)
    return default_offset


class PickAndTransferPolicy3D(PickAndTransferPolicy):
    """Transfer policy with Z-rotation compensation for 3D spawned cube.
    Pickup: gripper matches cube's Z-rotation + safe grasp offset.
    Transfer: gripper uses standard orientation (realigned).

    v2 (2026-04-29): meet_xyz 에 per-episode 노이즈 주입.
        클래스 attr 로 amplitude/스위치 default 설정, 외부에서 override 가능.
    """

    # Meet point 노이즈 설정 (default = 작은 amp, ON, fixed offset 없음)
    MEET_NOISE_AMP = (0.03, 0.02, 0.01)   # v2: transfer noise 축소 (1차 SR 80→95% 목표)
    MEET_NOISE_ON = True                  # False → 노이즈 OFF (deterministic)
    MEET_FIXED_OFFSET = None              # (dx, dy, dz) 지정 시 그 값 고정

    def generate_trajectory(self, ts_first):
        from utils.utils import sample_meet_xyz_noise
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        box_info = np.array(ts_first.observation['env_state'])
        box_xyz = box_info[:3]
        box_quat = box_info[3:]

        # Extract Z-rotation from box quaternion
        box_z_angle = 2 * np.arctan2(box_quat[3], box_quat[0])

        # Standard gripper quat (same as 2D)
        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        # Rotated gripper quat for pickup (matches box Z-rotation, world frame)
        gripper_pick_quat_rotated = Quaternion(axis=[0.0, 0.0, 1.0], radians=box_z_angle) * gripper_pick_quat

        # Safe grasp offset (box half-height = 0.02)
        grasp_z = _safe_grasp_z_offset(box_xyz[2], 0.02, default_offset=-0.025)

        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)
        # v2: 큐브 우측 스폰(x [0.05, 0.30]) → 전달은 중심에서 5cm 좌측 (좌완 receive 자연 위치)
        meet_xyz_base = np.array([-0.05, 0.5, 0.25])
        meet_xyz = meet_xyz_base + sample_meet_xyz_noise(
            amp_xyz=self.MEET_NOISE_AMP,
            enabled=self.MEET_NOISE_ON,
            fixed_offset=self.MEET_FIXED_OFFSET,
        )

        # Left: receives at meet point (no rotation needed, same as 2D)
        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1},
            {"t": 100, "xyz": meet_xyz + np.array([-0.08, 0, 0.01]), "quat": meet_left_quat.elements, "gripper": 1},
            {"t": 260, "xyz": meet_xyz + np.array([0.025, 0, 0.01]), "quat": meet_left_quat.elements, "gripper": 0},
            {"t": 310, "xyz": meet_xyz + np.array([0.025, 0, 0.01]), "quat": meet_left_quat.elements, "gripper": 0},
            {"t": 360, "xyz": meet_xyz + np.array([-0.1, 0, 0.01]), "quat": np.array([1, 0, 0, 0]), "gripper": 0},
            {"t": 400, "xyz": meet_xyz + np.array([-0.1, 0, 0.01]), "quat": np.array([1, 0, 0, 0]), "gripper": 0},
        ]

        # Right: rotated quat for pickup, standard quat for transfer
        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1},
            # Pickup phase: rotated quat (match box Z-rotation) + safe grasp offset
            {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_rotated.elements, "gripper": 1},
            {"t": 130, "xyz": box_xyz + np.array([0.005, 0, grasp_z + 0.0]), "quat": gripper_pick_quat_rotated.elements, "gripper": 0},
            {"t": 170, "xyz": box_xyz + np.array([0.005, 0, grasp_z + 0.0]), "quat": gripper_pick_quat_rotated.elements, "gripper": 0},
            # Transfer phase: standard quat (realigned via interpolation)
            {"t": 200, "xyz": meet_xyz + np.array([0.06, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 0},
            {"t": 220, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 1},
            {"t": 310, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 1},
            {"t": 360, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1},
            {"t": 400, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1},
        ]


class InsertionPolicy3D(InsertionPolicy):
    """Insertion policy with Z-rotation compensation for 3D spawned objects.
    Pickup: gripper matches object's Z-rotation + safe grasp offset.
    Insertion: gripper uses standard orientation (realigned).

    v2 (2026-04-29): meet_xyz 에 per-episode 노이즈 주입.
        평면(z=0.05) + 회전 0° 환경에서도 동작 (회전 보정 코드는 자동 identity).
    """

    MEET_NOISE_AMP = (0.05, 0.03, 0.01)   # v2: insertion z 노이즈 0.02→0.01 (96%→100% 목표)
    MEET_NOISE_ON = True
    MEET_FIXED_OFFSET = None

    def generate_trajectory(self, ts_first):
        from utils.utils import sample_meet_xyz_noise
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        peg_info = np.array(ts_first.observation['env_state'])[:7]
        peg_xyz = peg_info[:3]
        peg_quat = peg_info[3:]

        socket_info = np.array(ts_first.observation['env_state'])[7:]
        socket_xyz = socket_info[:3]
        socket_quat = socket_info[3:]

        # Extract Z-rotation angles from object quaternions
        peg_z_angle = 2 * np.arctan2(peg_quat[3], peg_quat[0])
        socket_z_angle = 2 * np.arctan2(socket_quat[3], socket_quat[0])

        # Standard gripper quats (same as 2D)
        gripper_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_quat_right = gripper_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        gripper_quat_left = Quaternion(init_mocap_pose_right[3:])
        gripper_quat_left = gripper_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)

        # Rotated gripper quats for pickup (match object Z-rotation)
        gripper_quat_right_rot = Quaternion(axis=[0.0, 0.0, 1.0], radians=peg_z_angle) * gripper_quat_right
        gripper_quat_left_rot = Quaternion(axis=[0.0, 0.0, 1.0], radians=socket_z_angle) * gripper_quat_left

        # Safe grasp offsets (avoid platform collision)
        peg_grasp_z = _safe_grasp_z_offset(peg_xyz[2], 0.01, default_offset=-0.03)
        socket_grasp_z = _safe_grasp_z_offset(socket_xyz[2], 0.018, default_offset=-0.03)

        meet_xyz_base = np.array([0, 0.5, 0.25])
        meet_xyz = meet_xyz_base + sample_meet_xyz_noise(
            amp_xyz=self.MEET_NOISE_AMP,
            enabled=self.MEET_NOISE_ON,
            fixed_offset=self.MEET_FIXED_OFFSET,
        )
        lift_right = 0.01215

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1},
            # Pickup phase: rotated quat + safe grasp offset
            {"t": 120, "xyz": socket_xyz + np.array([0, 0, 0.08]), "quat": gripper_quat_left_rot.elements, "gripper": 1},
            {"t": 160, "xyz": socket_xyz + np.array([-0.01, 0, socket_grasp_z + 0.005]), "quat": gripper_quat_left_rot.elements, "gripper": 0},
            {"t": 220, "xyz": socket_xyz + np.array([-0.01, 0, socket_grasp_z + 0.005]), "quat": gripper_quat_left_rot.elements, "gripper": 0},
            # Insertion phase: standard quat (realigned via interpolation)
            {"t": 285, "xyz": meet_xyz + np.array([-0.1, 0, 0]), "quat": gripper_quat_left.elements, "gripper": 0},
            {"t": 340, "xyz": meet_xyz + np.array([-0.035, 0, 0]), "quat": gripper_quat_left.elements, "gripper": 0},
            {"t": 400, "xyz": meet_xyz + np.array([-0.035, 0, 0]), "quat": gripper_quat_left.elements, "gripper": 0},
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1},
            # Pickup phase: rotated quat + safe grasp offset
            {"t": 120, "xyz": peg_xyz + np.array([0, 0, 0.08]), "quat": gripper_quat_right_rot.elements, "gripper": 1},
            {"t": 160, "xyz": peg_xyz + np.array([0.01, 0, peg_grasp_z + 0.005]), "quat": gripper_quat_right_rot.elements, "gripper": 0},
            {"t": 220, "xyz": peg_xyz + np.array([0.01, 0, peg_grasp_z + 0.005]), "quat": gripper_quat_right_rot.elements, "gripper": 0},
            # Insertion phase: standard quat (realigned via interpolation)
            {"t": 285, "xyz": meet_xyz + np.array([0.1, 0, lift_right]), "quat": gripper_quat_right.elements, "gripper": 0},
            {"t": 340, "xyz": meet_xyz + np.array([0.035, 0, lift_right]), "quat": gripper_quat_right.elements, "gripper": 0},
            {"t": 400, "xyz": meet_xyz + np.array([0.035, 0, lift_right]), "quat": gripper_quat_right.elements, "gripper": 0},
        ]


def test_policy(task_name):
    # example rolling out pick_and_transfer policy
    onscreen_render = True
    inject_noise = False

    # setup the environment
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    if 'sim_transfer_cube' in task_name:
        env = make_ee_sim_env('sim_transfer_cube')
    elif 'sim_insertion' in task_name:
        env = make_ee_sim_env('sim_insertion')
    else:
        raise NotImplementedError

    for episode_idx in range(2):
        ts = env.reset()
        episode = [ts]
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images']['angle'])
            plt.ion()

        policy = PickAndTransferPolicy(inject_noise)
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images']['angle'])
                plt.pause(0.02)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        if episode_return > 0:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")


if __name__ == '__main__':
    test_task_name = 'sim_transfer_cube_scripted'
    test_policy(test_task_name)

