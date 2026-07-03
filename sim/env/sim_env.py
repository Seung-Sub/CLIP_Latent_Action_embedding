import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

from utils.constants import DT, XML_DIR, START_ARM_POSE
from utils.constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from utils.constants import MASTER_GRIPPER_POSITION_NORMALIZE_FN
from utils.constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from utils.constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

import IPython
e = IPython.embed

BOX_POSE = [None] # to be changed from outside

def make_sim_env(task_name, xml_path_override=None):
    """
    Environment for simulated robot bi-manual manipulation, with joint position control
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
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
        xml_path = xml_path_override or os.path.join(XML_DIR, f'bimanual_viperx_transfer_cube{suffix}.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeTask3D(random=False) if is_3d else TransferCubeTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_insertion' in task_name:
        suffix = '_3d' if is_3d else ''
        xml_path = xml_path_override or os.path.join(XML_DIR, f'bimanual_viperx_insertion{suffix}.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = InsertionTask3D(random=False) if is_3d else InsertionTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'cube_classification' in task_name:
        xml_path = xml_path_override or os.path.join(XML_DIR, 'bimanual_viperx_cube_classification.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = CubeClassificationTask(random=False)
        env = control.Environment(physics, task, time_limit=36, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env

class BimanualViperXTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        left_arm_action = action[:6]
        right_arm_action = action[7:7+6]
        normalized_left_gripper_action = action[6]
        normalized_right_gripper_action = action[7+6]

        left_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_left_gripper_action)
        right_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_right_gripper_action)

        full_left_gripper_action = [left_gripper_action, -left_gripper_action]
        full_right_gripper_action = [right_gripper_action, -right_gripper_action]

        env_action = np.concatenate([left_arm_action, full_left_gripper_action, right_arm_action, full_right_gripper_action])
        super().before_step(env_action, physics)
        return

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
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
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

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError


class TransferCubeTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
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


class InsertionTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7*2:] = BOX_POSE[0] # two objects
            # print(f"{BOX_POSE=}")
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


class CubeClassificationTask(BimanualViperXTask):
    """Color-sort with adaptive handover.

    qpos layout: [robot(16)] [red(7)] [blue(7)] = 30.

    Reward 단계 (max=4, sticky 진행도):
        0 = 어느 그리퍼도 자기 색깔 cube를 터치한 적 없음.
        1 = 한 cube만 올바른 그리퍼가 터치한 적 있음.
        2 = 두 cube 모두 올바른 그리퍼가 터치한 적 있음 (아직 plate에 안 올림).
        3 = 한 cube가 자기 색깔 plate에 올라간 적 있음 (다른 cube는 placed 안 됨).
        4 = 두 cube 모두 자기 색깔 plate에 올라간 적 있음.

    "터치"는 약속된 그리퍼 ↔ cube contact (red ↔ left_gripper_finger, blue ↔ right). sticky.
    "올라감"은 cube ↔ goal_geom contact. sticky (한 번 안착하면 상위 단계 유지).
    """

    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4
        self._red_touched = False
        self._blue_touched = False
        self._red_placed = False
        self._blue_placed = False

    def initialize_episode(self, physics):
        # Reset robot + qpos[-14:] from BOX_POSE (set by record_sim_episodes from EE phase).
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-14:] = BOX_POSE[0]
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


class TransferCubeTask3D(TransferCubeTask):
    """Transfer cube task with 3D spawning (Z elevation + Z rotation + platform).
    qpos layout is identical to 2D: [robot(16)] [box(7)] = 23.
    Platform is a static body positioned via physics.model.body_pos."""
    def initialize_episode(self, physics):
        # Parent sets robot qpos + BOX_POSE[0] into qpos[-7:] — works as-is
        super().initialize_episode(physics)
        # Position static platform under the box
        from utils.utils import position_platform
        box_xyz = physics.data.qpos[-7:-4]
        position_platform(physics, 'cube_platform', box_xyz, object_half_height=0.02)


class InsertionTask3D(InsertionTask):
    """Insertion task with 3D spawning (Z elevation + Z rotation + platforms).
    qpos layout is identical to 2D: [robot(16)] [peg(7)] [socket(7)] = 30.
    Platforms are static bodies positioned via physics.model.body_pos."""
    def initialize_episode(self, physics):
        # Parent sets robot qpos + BOX_POSE[0] into qpos[-14:] — works as-is
        super().initialize_episode(physics)
        # Position static platforms under objects
        from utils.utils import position_platform
        peg_xyz = physics.data.qpos[-14:-11]
        socket_xyz = physics.data.qpos[-7:-4]
        position_platform(physics, 'peg_platform', peg_xyz, object_half_height=0.01)
        position_platform(physics, 'socket_platform', socket_xyz, object_half_height=0.018)


def get_action(master_bot_left, master_bot_right):
    action = np.zeros(14)
    # arm action
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    action[7:7+6] = master_bot_right.dxl.joint_states.position[:6]
    # gripper action
    left_gripper_pos = master_bot_left.dxl.joint_states.position[7]
    right_gripper_pos = master_bot_right.dxl.joint_states.position[7]
    normalized_left_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(left_gripper_pos)
    normalized_right_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(right_gripper_pos)
    action[6] = normalized_left_pos
    action[7+6] = normalized_right_pos
    return action

def test_sim_teleop():
    """ Testing teleoperation in sim with ALOHA. Requires hardware and ALOHA repo to work. """
    from interbotix_xs_modules.arm import InterbotixManipulatorXS

    BOX_POSE[0] = [0.2, 0.5, 0.05, 1, 0, 0, 0]

    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_right', init_node=False)

    # setup the environment
    env = make_sim_env('sim_transfer_cube')
    ts = env.reset()
    episode = [ts]
    # setup plotting
    ax = plt.subplot()
    plt_img = ax.imshow(ts.observation['images']['angle'])
    plt.ion()

    for t in range(1000):
        action = get_action(master_bot_left, master_bot_right)
        ts = env.step(action)
        episode.append(ts)

        plt_img.set_data(ts.observation['images']['angle'])
        plt.pause(0.02)


if __name__ == '__main__':
    test_sim_teleop()

