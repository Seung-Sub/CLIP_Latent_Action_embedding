import numpy as np
import torch
import torch.nn.functional as F  # VIT resize용
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

# keyframe 패딩 길이 (실제 선별/저장은 data_process/add_keyframes.py 에서만 수행)
MAX_KEYFRAMES = 10

GRIPPER_INDICES_BY_ARMS = {
    1: [7],       # Franka single: 7 joints + gripper
    2: [6, 13],   # ALOHA bimanual
    3: [6, 13],   # av-aloha 3-arm (middle arm gripper 없음)
    4: [7, 15],   # dual Franka (예약)
    5: [7, 8],    # IGRIS-C single arm 9-D: right arm 7 + thumb + index
}


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, use_vit=False, augment: str = None, image_noise: float = None, num_arms=2, chunk_size=None):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.use_vit = use_vit  # VIT면 224×224로 리사이즈
        self.augment = augment
        self.image_noise = image_noise
        self.num_arms = num_arms
        self.pad_len = int(chunk_size * 1.5) if chunk_size else None

        # VIT transform 초기화
        if self.use_vit:
            from vit.vit_transforms import VITImageTransform
            self.vit_transform = VITImageTransform(
                augment=augment,
                image_noise=image_noise
            )
        else:
            self.vit_transform = None

        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            # qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            action = root['/action'][start_ts:]
            action_len = episode_len - start_ts
            # read precomputed keyframe indices (absolute timesteps)
            keyframe_abs = root['/keyframe'][()]

        self.is_sim = is_sim
        action_dim = original_action_shape[1]
        pad_len = self.pad_len if self.pad_len else episode_len
        actual_len = min(action_len, pad_len)
        padded_action = np.zeros((pad_len, action_dim), dtype=np.float32)
        padded_action[:actual_len] = action[:actual_len]
        is_pad = np.zeros(pad_len)
        is_pad[actual_len:] = 1

        # keyframe indices: absolute → window-relative (start_ts 기준)
        relative = keyframe_abs - start_ts
        valid = (keyframe_abs >= 0) & (relative >= 0) & (relative < pad_len)
        valid_relative = np.sort(relative[valid])
        keyframe_indices = np.full(MAX_KEYFRAMES, -1, dtype=np.int64)
        n = min(len(valid_relative), MAX_KEYFRAMES)
        if n > 0:
            keyframe_indices[:n] = valid_relative[:n]

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # VIT 처리
        if self.use_vit:
            # Step 1: 640 → 480 center crop
            image_data = self.vit_transform.crop_640_to_480(image_data)
            
            # Step 2: 480 → 240 → 224 (crop augmentation 적용)
            image_data = self.vit_transform.resize_for_vit(
                image_data, 
                is_training=True
            )
            
            # Step 3: 정규화 (중요: apply_image_noise 전)
            image_data = image_data / 255.0
            
            # Step 4: Image augmentation + ImageNet Normalize
            # apply_image_noise 내부에서:
            # - ColorJitter 적용 (is_training=True일 때)
            # - ImageNet Normalize 적용 (항상)
            image_data = self.vit_transform.apply_image_noise(
                image_data,
                is_training=True
            )
        else:
            # ResNet18 (기존 경로 유지)
            image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        # keyframe 인덱스 텐서 변환
        keyframe_indices_tensor = torch.from_numpy(keyframe_indices).long()

        return image_data, qpos_data, action_data, is_pad, keyframe_indices_tensor


def get_norm_stats(dataset_dir, num_episodes):
    # /keyframe 존재 여부 사전 체크 (첫 파일만)
    first_path = os.path.join(dataset_dir, 'episode_0.hdf5')
    with h5py.File(first_path, 'r') as f:
        if '/keyframe' not in f:
            raise RuntimeError(
                f"'/keyframe' 데이터셋이 없습니다: {first_path}\n"
                f"  실행: python data_process/add_keyframes.py "
                f"--dataset_dir {dataset_dir} --num_arms <N>")

    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            # qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))

    all_qpos_data = torch.cat(all_qpos_data, dim=0)       # (sum_T, D)
    all_action_data = torch.cat(all_action_data, dim=0)    # (sum_T, D)

    # normalize action data
    # TODO: [-1, 1] min-max 정규화 시도 (Diffusion Policy 방식)
    #   normalized = (action - joint_min) / (joint_max - joint_min) * 2 - 1
    #   그리퍼도 0~1 → -1~1로 매핑 가능
    action_mean = all_action_data.mean(dim=0, keepdim=True)
    action_std = all_action_data.std(dim=0, keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std = all_qpos_data.std(dim=0, keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def _worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, use_vit=False, augment: str = None, image_noise: float = None, num_arms=2, chunk_size=None, seed=1):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, use_vit=use_vit, augment=augment, image_noise=image_noise, num_arms=num_arms, chunk_size=chunk_size)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, use_vit=use_vit, augment=None, image_noise=None, num_arms=num_arms, chunk_size=chunk_size)
    train_g = torch.Generator()
    train_g.manual_seed(seed)
    val_g = torch.Generator()
    val_g.manual_seed(seed)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=0, generator=train_g)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=0, generator=val_g)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_cube_classification_pose(force_config=None, min_x_sep=0.06, max_retries=200):
    """Spawn 2 boxes (red, blue) inside a 40×20cm workspace for cube_classification task.

    Workspace per arm: 20×20cm 정사각형 2개 (좌/우), 총 40×20cm.
        x_left  ∈ [-0.20, 0.00]   x_right ∈ [0.00, +0.20]
        y       ∈ [0.40, 0.60]    (transfer_cube와 동일 범위 — IK 안정)
        z       = 0.05 (테이블 위 안착)
        rot     = ±15° (Z축)

    force_config: None → 무작위. 'A' = red 좌측 + blue 우측 (정배치).
                                  'B' = red 우측 + blue 좌측 (교차).
                                  'C' = 둘 다 좌측 (좌몰림).
                                  'D' = 둘 다 우측 (우몰림).
    min_x_sep: 두 박스 x 좌표 최소 간격 (붙어서 스폰 방지).

    Returns: (red_pose(7), blue_pose(7), config_str).
    """
    workspace_y = (0.40, 0.60)
    z = 0.05

    def _x_range(side):
        return (-0.20, 0.00) if side == 'left' else (0.00, 0.20)

    def _one(side):
        x = np.random.uniform(*_x_range(side))
        y = np.random.uniform(*workspace_y)
        angle = np.random.uniform(-15, 15) * np.pi / 180
        quat = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
        return np.concatenate([[x, y, z], quat])

    side_map = {
        'A': ('left', 'right'),
        'B': ('right', 'left'),
        'C': ('left', 'left'),
        'D': ('right', 'right'),
    }
    if force_config is None:
        force_config = np.random.choice(['A', 'B', 'C', 'D'])
    if force_config not in side_map:
        raise ValueError(f"force_config must be one of {list(side_map)}, got {force_config!r}")

    red_side, blue_side = side_map[force_config]
    for _ in range(max_retries):
        red, blue = _one(red_side), _one(blue_side)
        if abs(red[0] - blue[0]) >= min_x_sep:
            return red, blue, force_config
    raise RuntimeError(
        f"Failed to satisfy min_x_sep={min_x_sep} for config={force_config} "
        f"after {max_retries} retries"
    )


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

def _compute_platform_pos(x, y, z, object_half_height):
    """Compute platform XYZ position to support object at given z height.
    If elevation is negligible, hide platform far away.
    Returns np.array([x, y, z]) — 3D position only (static body, no quat needed)."""
    elevation = z - 0.05  # above table surface
    if elevation > 0.005:
        platform_top_z = z - object_half_height
        platform_center_z = platform_top_z - 0.05  # platform half-height
        return np.array([x, y, platform_center_z])
    else:
        return np.array([0.0, 0.6, -0.5])

def _z_rotation_quat(rot_range_deg):
    """Return quaternion for random Z-axis rotation within ±rot_range_deg."""
    angle = np.random.uniform(-rot_range_deg, rot_range_deg) * np.pi / 180
    return np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])

def position_platform(physics, body_name, obj_xyz, object_half_height):
    """Position a static platform body under an object.
    Uses physics.model.body_pos (works for static bodies without joints).

    v2: insertion XML 에서 platform body 가 제거된 경우 silent no-op.
    """
    try:
        body_id = physics.model.name2id(body_name, 'body')
    except Exception:
        return
    pos = _compute_platform_pos(obj_xyz[0], obj_xyz[1], obj_xyz[2], object_half_height)
    physics.model.body_pos[body_id] = pos

def sample_box_pose_3d(x_range=[0.05, 0.25], y_range=[0.4, 0.6],
                       z_range=[0.05, 0.15], rot_range=15):
    """Sample 3D box pose (v3.2: x 상한 축소 — 로봇 베이스와의 마진 확보).

    v3   (2026-05-06): transfer cube 진짜 3D 복원 — z=[0.05, 0.20], rot=±15°.
    v3.1 (2026-05-07): z 변동을 2/3로 축소 + 평가/녹화 환경 그림자 복원
                       (bimanual_viperx_transfer_cube_3d.xml → scene_3d.xml).
    v3.2 (2026-05-07): x 상한 0.30 → 0.25 (right 로봇 베이스 x=0.469와의 거리 확보).
        x:   [0.05, 0.25]   (상한 0.30 → 0.25)
        y:   [0.4, 0.6]     (유지)
        z:   [0.05, 0.15]   (테이블 표면 위 0~10cm 변동 → 플랫폼이 박스 받침)
        rot: ±15°           (유지)
    Insertion 은 별도 함수(sample_insertion_pose_3d)에서 평면 유지.
    """
    x = np.random.uniform(*x_range)
    y = np.random.uniform(*y_range)
    z = np.random.uniform(*z_range)
    quat = _z_rotation_quat(rot_range)
    return np.concatenate([[x, y, z], quat])

def sample_insertion_pose_3d(x_range_peg=[0.1, 0.2], x_range_socket=[-0.2, -0.1],
                             y_range=[0.4, 0.6],
                             z_range=[0.05, 0.05], rot_range=0):
    """Sample peg / socket poses for *flat* insertion (z 고정 + 회전 없음).

    v2 (2026-04-29): insertion 평면화 — z=0.05 고정, rot=0°.
        z_range:  [0.05, 0.15] → [0.05, 0.05]
        rot_range: 10            → 0
    플랫폼은 elevation≈0 → `_compute_platform_pos` 가 자동 hide.
    """
    px = np.random.uniform(*x_range_peg)
    py = np.random.uniform(*y_range)
    pz = np.random.uniform(*z_range)
    peg_quat = _z_rotation_quat(rot_range)
    peg_pose = np.concatenate([[px, py, pz], peg_quat])

    sx = np.random.uniform(*x_range_socket)
    sy = np.random.uniform(*y_range)
    sz = np.random.uniform(*z_range)
    socket_quat = _z_rotation_quat(rot_range)
    socket_pose = np.concatenate([[sx, sy, sz], socket_quat])

    return peg_pose, socket_pose


def sample_meet_xyz_noise(amp_xyz=(0.05, 0.03, 0.04),
                          enabled=True,
                          fixed_offset=None):
    """Per-episode random offset to add to a base meet_xyz.

    Args:
        amp_xyz       : (ax, ay, az) — uniform on [-ax, ax].
        enabled       : False → returns zeros (노이즈 OFF).
        fixed_offset  : (dx, dy, dz) — given → 그 값 그대로 (결정론적 오프셋).

    우선순위: fixed_offset > (not enabled → 0) > amp_xyz uniform.
    """
    if fixed_offset is not None:
        return np.asarray(fixed_offset, dtype=float)
    if not enabled:
        return np.zeros(3)
    a = np.asarray(amp_xyz, dtype=float)
    return np.random.uniform(-a, a)


def _sample_from_margin(outer_range, inner_range):
    """Sample uniformly from outer_range EXCLUDING inner_range.
    E.g. outer=[−0.01, 0.21], inner=[0.0, 0.2] → sample from [−0.01, 0.0) ∪ (0.2, 0.21]."""
    left  = max(0.0, inner_range[0] - outer_range[0])
    right = max(0.0, outer_range[1] - inner_range[1])
    total = left + right
    if total <= 0:
        raise ValueError(f"No OOD margin: outer={outer_range}, inner={inner_range}")
    if np.random.uniform() < left / total:
        return np.random.uniform(outer_range[0], inner_range[0])
    else:
        return np.random.uniform(inner_range[1], outer_range[1])

def _sample_rot_from_margin(rot_range, rot_range_in):
    """Sample rotation (degrees) from OOD margins: [−rot, −rot_in) ∪ (rot_in, rot]."""
    margin = rot_range - rot_range_in
    if margin <= 0:
        raise ValueError(f"No OOD rotation margin: outer=±{rot_range}, inner=±{rot_range_in}")
    if np.random.uniform() < 0.5:
        return np.random.uniform(-rot_range, -rot_range_in)
    else:
        return np.random.uniform(rot_range_in, rot_range)

def sample_box_pose_3d_ood(
        # Outer (evaluation) bounds — XY only
        x_range=[0.04, 0.26], y_range=[0.39, 0.61],
        # Inner (training) bounds — XY excluded
        x_range_in=[0.05, 0.25], y_range_in=[0.4, 0.6],
        # v3.1 — z/rot in-distribution은 transfer cube의 새 분포에 맞춤 (z 2/3로 축소)
        z_range_in=[0.05, 0.15], rot_range_in=15):
    """v3.2 (2026-05-07): OOD = XY-only 고정. x 상한 0.30→0.25 (in), 0.31→0.26 (outer). z/rot은 transfer cube의 in-distribution(z=[0.05,0.15], rot=±15°)."""
    x = _sample_from_margin(x_range, x_range_in)
    y = _sample_from_margin(y_range, y_range_in)
    z = np.random.uniform(*z_range_in)
    rot_deg = np.random.uniform(-rot_range_in, rot_range_in)
    angle = rot_deg * np.pi / 180
    quat = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
    return np.concatenate([[x, y, z], quat])

def sample_insertion_pose_3d_ood(
        x_range_peg=[0.09, 0.21], x_range_socket=[-0.21, -0.09],
        y_range=[0.39, 0.61],
        x_range_peg_in=[0.1, 0.2], x_range_socket_in=[-0.2, -0.1],
        y_range_in=[0.4, 0.6],
        # insertion v2: z/rot in-distribution = 평면 + 회전 0
        z_range_in=[0.05, 0.05], rot_range_in=0):
    """v2 (2026-04-29): OOD = XY-only 고정. insertion 평면 (z=0.05, rot=0)."""
    px = _sample_from_margin(x_range_peg, x_range_peg_in)
    py = _sample_from_margin(y_range, y_range_in)
    pz = np.random.uniform(*z_range_in)
    if rot_range_in > 0:
        p_rot = np.random.uniform(-rot_range_in, rot_range_in)
    else:
        p_rot = 0.0
    angle = p_rot * np.pi / 180
    peg_quat = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
    peg_pose = np.concatenate([[px, py, pz], peg_quat])

    sx = _sample_from_margin(x_range_socket, x_range_socket_in)
    sy = _sample_from_margin(y_range, y_range_in)
    sz = np.random.uniform(*z_range_in)
    if rot_range_in > 0:
        s_rot = np.random.uniform(-rot_range_in, rot_range_in)
    else:
        s_rot = 0.0
    angle = s_rot * np.pi / 180
    socket_quat = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
    socket_pose = np.concatenate([[sx, sy, sz], socket_quat])

    return peg_pose, socket_pose


### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
