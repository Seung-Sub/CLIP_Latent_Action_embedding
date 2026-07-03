import numpy as np
import collections
import os

from utils.constants import DT, XML_DIR, START_ARM_POSE
from utils.constants import PUPPET_GRIPPER_POSITION_CLOSE
from utils.constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from utils.constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from utils.constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

from utils.utils import sample_box_pose, sample_insertion_pose
from utils.utils import sample_box_pose_3d, sample_insertion_pose_3d, position_platform
from utils.utils import sample_cube_classification_pose
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

import IPython
e = IPython.embed


def make_ee_sim_env(task_name):
    """
    Environment for simulated robot bi-manual manipulation, with end-effector control.
    Action space:      [left_arm_pose (7),             # position and quaternion for end effector
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_pose (7),            # position and quaternion for end effector
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
    """
    is_3d = '3d' in task_name
    if 'sim_transfer_cube' in task_name:
        suffix = '_3d' if is_3d else ''
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_ee_transfer_cube{suffix}.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeEETask3D(random=False) if is_3d else TransferCubeEETask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_insertion' in task_name:
        suffix = '_3d' if is_3d else ''
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_ee_insertion{suffix}.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = InsertionEETask3D(random=False) if is_3d else InsertionEETask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'cube_classification' in task_name:
        xml_path = os.path.join(XML_DIR, 'bimanual_viperx_ee_cube_classification.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = CubeClassificationEETask(random=False)
        env = control.Environment(physics, task, time_limit=36, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env

class BimanualViperXEETask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)
        # ✨ NEW: 외부에서 설정되는 이진 그리퍼 명령값 저장
        self.binary_gripper_cmd_left = 1.0
        self.binary_gripper_cmd_right = 1.0
    
    def set_binary_gripper_commands(self, left_cmd, right_cmd):
        """
        ✨ NEW: scripted_policy에서 현재 이진 그리퍼 명령 설정
        
        Args:
            left_cmd (float): 좌측 이진 명령 (0.0 또는 1.0)
            right_cmd (float): 우측 이진 명령 (0.0 또는 1.0)
        """
        self.binary_gripper_cmd_left = left_cmd
        self.binary_gripper_cmd_right = right_cmd

    def before_step(self, action, physics):
        a_len = len(action) // 2
        action_left = action[:a_len]
        action_right = action[a_len:]

        # set mocap position and quat
        # left
        np.copyto(physics.data.mocap_pos[0], action_left[:3])
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        # right
        np.copyto(physics.data.mocap_pos[1], action_right[:3])
        np.copyto(physics.data.mocap_quat[1], action_right[3:7])

        # set gripper
        g_left_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_left[7])
        g_right_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_right[7])
        np.copyto(physics.data.ctrl, np.array([g_left_ctrl, -g_left_ctrl, g_right_ctrl, -g_right_ctrl]))

    def initialize_robots(self, physics):
        # reset joint position
        physics.named.data.qpos[:16] = START_ARM_POSE

        # reset mocap to align with end effector
        # to obtain these numbers:
        # (1) make an ee_sim env and reset to the same start_pose
        # (2) get env._physics.named.data.xpos['vx300s_left/gripper_link']
        #     get env._physics.named.data.xquat['vx300s_left/gripper_link']
        #     repeat the same for right side
        np.copyto(physics.data.mocap_pos[0], [-0.31718881, 0.5, 0.29525084])
        np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])
        # right
        np.copyto(physics.data.mocap_pos[1], np.array([0.31718881, 0.49999888, 0.29525084]))
        np.copyto(physics.data.mocap_quat[1],  [1, 0, 0, 0])

        # reset gripper control
        close_gripper_control = np.array([
            PUPPET_GRIPPER_POSITION_CLOSE,
            -PUPPET_GRIPPER_POSITION_CLOSE,
            PUPPET_GRIPPER_POSITION_CLOSE,
            -PUPPET_GRIPPER_POSITION_CLOSE,
        ])
        np.copyto(physics.data.ctrl, close_gripper_control)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        qpos_raw = self.get_qpos(physics)
        
        # ✨ CHANGED: gripper는 이진 명령값(0 또는 1)으로 반환 (지연 없음!)
        qpos = qpos_raw.copy()
        qpos[6] = self.binary_gripper_cmd_left   # ← 외부에서 설정된 이진 명령
        qpos[13] = self.binary_gripper_cmd_right # ← 외부에서 설정된 이진 명령
        
        obs['qpos'] = qpos
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        obs['images']['angle_down'] = physics.render(height=480, width=640, camera_id='angle_down')
        obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')

        obs['images']['left_pillar'] = physics.render(height=480, width=640, camera_id='left_pillar')
        obs['images']['right_pillar'] = physics.render(height=480, width=640, camera_id='right_pillar')
        obs['images']['left_wrist'] = physics.render(height=480, width=640, camera_id='left_wrist')
        obs['images']['right_wrist'] = physics.render(height=480, width=640, camera_id='right_wrist')


        # used in scripted policy to obtain starting pose
        obs['mocap_pose_left'] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
        obs['mocap_pose_right'] = np.concatenate([physics.data.mocap_pos[1], physics.data.mocap_quat[1]]).copy()

        # used when replaying joint trajectory
        obs['gripper_ctrl'] = physics.data.ctrl.copy()
        return obs

    def get_reward(self, physics):
        raise NotImplementedError


class TransferCubeEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        cube_pose = sample_box_pose()
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)
        # print(f"randomized cube position to {cube_position}")

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table: # lifted
            reward = 2
        if touch_left_gripper: # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table: # successful transfer
            reward = 4
        return reward


class InsertionEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize peg and socket position
        peg_pose, socket_pose = sample_insertion_pose()
        id2index = lambda j_id: 16 + (j_id - 16) * 7 # first 16 is robot qpos, 7 is pose dim # hacky

        peg_start_id = physics.model.name2id('red_peg_joint', 'joint')
        peg_start_idx = id2index(peg_start_id)
        np.copyto(physics.data.qpos[peg_start_idx : peg_start_idx + 7], peg_pose)
        # print(f"randomized cube position to {cube_position}")

        socket_start_id = physics.model.name2id('blue_socket_joint', 'joint')
        socket_start_idx = id2index(socket_start_id)
        np.copyto(physics.data.qpos[socket_start_idx : socket_start_idx + 7], socket_pose)
        # print(f"randomized cube position to {cube_position}")

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether peg touches the pin
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_left_gripper = ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = ("socket-1", "table") in all_contact_pairs or \
                             ("socket-2", "table") in all_contact_pairs or \
                             ("socket-3", "table") in all_contact_pairs or \
                             ("socket-4", "table") in all_contact_pairs
        peg_touch_socket = ("red_peg", "socket-1") in all_contact_pairs or \
                           ("red_peg", "socket-2") in all_contact_pairs or \
                           ("red_peg", "socket-3") in all_contact_pairs or \
                           ("red_peg", "socket-4") in all_contact_pairs
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper: # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table): # grasp both
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table): # peg and socket touching
            reward = 3
        if pin_touched: # successful insertion
            reward = 4
        return reward


class TransferCubeEETask3D(TransferCubeEETask):
    """EE transfer cube with 3D spawning (Z elevation + Z rotation + platform).
    qpos layout is identical to 2D. Platform is a static body."""
    def initialize_episode(self, physics):
        self.initialize_robots(physics)
        box_pose = sample_box_pose_3d()
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], box_pose)
        # Position static platform under the box
        position_platform(physics, 'cube_platform', box_pose[:3], object_half_height=0.02)
        super(TransferCubeEETask, self).initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state


class CubeClassificationEETask(BimanualViperXEETask):
    """EE control for cube_classification task.

    qpos layout: [robot(16)] [red_box(7)] [blue_box(7)] = 30.
    Spawn config can be forced by setting `task.force_config = 'A'/'B'/'C'/'D'` before reset.

    Reward 단계 (max=4, sticky 진행도):
        0 = 어느 그리퍼도 자기 색 cube를 터치한 적 없음.
        1 = 한 cube만 터치됨.
        2 = 두 cube 모두 터치됨 (아직 plate 안 됨).
        3 = 한 cube가 자기 색 plate에 안착한 적 있음.
        4 = 두 cube 모두 자기 색 plate에 안착한 적 있음.
    """
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4
        self.force_config = None
        self.last_config = None
        self._red_touched = False
        self._blue_touched = False
        self._red_placed = False
        self._blue_placed = False

    def initialize_episode(self, physics):
        self.initialize_robots(physics)
        red_pose, blue_pose, cfg = sample_cube_classification_pose(force_config=self.force_config)
        self.last_config = cfg

        red_start = physics.model.name2id('red_box_joint', 'joint')
        id2index = lambda j_id: 16 + (j_id - 16) * 7
        red_idx = id2index(red_start)
        blue_start = physics.model.name2id('blue_box_joint', 'joint')
        blue_idx = id2index(blue_start)
        np.copyto(physics.data.qpos[red_idx : red_idx + 7], red_pose)
        np.copyto(physics.data.qpos[blue_idx : blue_idx + 7], blue_pose)

        self._red_touched = False
        self._blue_touched = False
        self._red_placed = False
        self._blue_placed = False
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id1 = physics.data.contact[i_contact].geom1
            id2 = physics.data.contact[i_contact].geom2
            n1 = physics.model.id2name(id1, 'geom')
            n2 = physics.model.id2name(id2, 'geom')
            all_contact_pairs.append((n1, n2))
            all_contact_pairs.append((n2, n1))

        if ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs:
            self._red_touched = True
        if ("blue_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs:
            self._blue_touched = True
        if ("red_box", "red_goal_geom") in all_contact_pairs:
            self._red_placed = True
        if ("blue_box", "blue_goal_geom") in all_contact_pairs:
            self._blue_placed = True

        placed_count  = int(self._red_placed)  + int(self._blue_placed)
        touched_count = int(self._red_touched) + int(self._blue_touched)

        if placed_count == 2:
            return 4
        if placed_count == 1:
            return 3
        if touched_count == 2:
            return 2
        if touched_count == 1:
            return 1
        return 0


class InsertionEETask3D(InsertionEETask):
    """EE insertion with 3D spawning (Z elevation + Z rotation + platforms).
    qpos layout is identical to 2D. Platforms are static bodies."""
    def initialize_episode(self, physics):
        self.initialize_robots(physics)
        peg_pose, socket_pose = sample_insertion_pose_3d()

        id2index = lambda j_id: 16 + (j_id - 16) * 7

        peg_start_id = physics.model.name2id('red_peg_joint', 'joint')
        peg_start_idx = id2index(peg_start_id)
        np.copyto(physics.data.qpos[peg_start_idx : peg_start_idx + 7], peg_pose)

        socket_start_id = physics.model.name2id('blue_socket_joint', 'joint')
        socket_start_idx = id2index(socket_start_id)
        np.copyto(physics.data.qpos[socket_start_idx : socket_start_idx + 7], socket_pose)

        # Position static platforms under objects
        position_platform(physics, 'peg_platform', peg_pose[:3], object_half_height=0.01)
        position_platform(physics, 'socket_platform', socket_pose[:3], object_half_height=0.018)

        super(InsertionEETask, self).initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state
