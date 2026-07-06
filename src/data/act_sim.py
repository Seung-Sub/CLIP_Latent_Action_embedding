"""ALOHA 시뮬(act_sim) 데이터 로더 — StackcupDataset과 동일 인터페이스.

에피소드 구조 (third_party/act_sim/record_sim_episodes.py 산출):
  observations/images/<cam>  : (T, 480, 640, 3) uint8, 50Hz (DT=0.02)
  observations/qpos          : (T, 14)  — proprioception
  action                     : (T, 14)  — 양팔 관절 목표 (6+1그리퍼 ×2)

stackcup과의 차이:
  - 이미지가 JPEG bytes가 아닌 raw 배열, 타임스탬프 없이 등간격 50Hz
  - 액션이 delta EEF가 아니라 절대 관절공간 14D (변환 없이 정규화만)
학습 샘플 = (z_t, z_{t+chunk}, 액션청크 n_chunk×14)
"""
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

HZ = 50.0   # DT = 0.02


class ActSimDataset:
    def __init__(self, cfg):
        d = cfg["data"]
        roots = d["root"] if isinstance(d["root"], list) else [d["root"]]
        self.roots = [Path(os.path.expanduser(r)) for r in roots]
        cam = d["camera"]
        self.cameras = cam if isinstance(cam, list) else [cam]
        self.chunk_sec = float(d["chunk_sec"])
        self.n_chunk = int(d["n_chunk"])
        self.cache_dir = Path(os.path.expanduser(d["cache_dir"]))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.span = max(2, int(round(self.chunk_sec * HZ)))   # 프레임 간격(스텝)
        self.stride = max(1, self.span // 8)                  # 시작 시점 간격

    def episode_files(self):
        files = []
        for r in self.roots:
            files += sorted(r.glob("episode_*.hdf5"))
        return files

    # ---------- 원시 접근 ----------

    def load_actions(self, path):
        with h5py.File(path, "r") as f:
            return f["action"][:].astype(np.float64)

    def load_frames(self, path, camera):
        with h5py.File(path, "r") as f:
            return f[f"observations/images/{camera}"][:]

    # ---------- 임베딩 캐시 (앵커별 분리: {anchor_id}/{projection}/{normalize}) ----------

    def _cache_path(self, encoder, filename):
        key = getattr(encoder, "cache_key", None)
        if key is None:                                   # 구형 ClipWrapper 직접 사용
            return self.cache_dir / filename
        p = self.cache_dir / key / filename
        legacy = self.cache_dir / filename
        # 하위호환: 기본 앵커(기존 ClipWrapper와 동일 출력)는 기존 평면 캐시 재사용
        if not p.exists() and key == "clip-vit-l-14/joint/norm" and legacy.exists():
            return legacy
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def embeddings(self, clip, path, camera=None):
        camera = camera or self.cameras[0]
        cache = self._cache_path(clip, path.parent.name + "_" + path.stem
                                 + f"_{camera}.npz")
        if cache.exists():
            return np.load(cache)["Z"]
        imgs = self.load_frames(path, camera)
        frames = [Image.fromarray(im) for im in imgs]
        Z = []
        for i in range(0, len(frames), 64):
            Z.append(clip.encode_images(frames[i:i + 64])["embeds"])
        Z = np.concatenate(Z)
        np.savez_compressed(cache, Z=Z)
        return Z

    # ---------- 학습 쌍 생성 ----------

    def pair_starts(self, T):
        return list(range(0, T - self.span, self.stride))

    def resample_chunk(self, seg):
        src = np.linspace(0, len(seg) - 1, self.n_chunk)
        lo = np.floor(src).astype(int)
        hi = np.minimum(lo + 1, len(seg) - 1)
        w = (src - lo)[:, None]
        return seg[lo] * (1 - w) + seg[hi] * w

    def build(self, clip, files=None, verbose=True):
        files = files or self.episode_files()
        out = []
        for p in files:
            acts = self.load_actions(p)
            T = len(acts)
            starts = self.pair_starts(T)
            per_cam = []
            for cam in self.cameras:
                Z = self.embeddings(clip, p, cam)
                Zt = np.stack([Z[i] for i in starts])
                Ztn = np.stack([Z[i + self.span] for i in starts])
                A = np.stack([self.resample_chunk(acts[i:i + self.span]).ravel()
                              for i in starts])
                per_cam.append((Zt.astype(np.float32), Ztn.astype(np.float32),
                                A.astype(np.float32)))
            pairs = tuple(np.concatenate([pc[k] for pc in per_cam])
                          for k in range(3))
            out.append(pairs)
            if verbose:
                print(f"  {p.stem}: T={T}, pairs {len(pairs[0])}")
        return out

    def build_policy_samples(self, clip, files=None, stride=2):
        """정책 학습용 연속 윈도우 삼중쌍 (에피소드별).

        returns per-episode (Zprev, Zcur, Znext, Apast, Afut).
        경계 포함: t=0부터 시작해 롤아웃 부트스트랩 문맥
        (정지 과거, z_prev≈z_cur)을 학습 분포에 포함한다.
        """
        files = files or self.episode_files()
        out = []
        for p in files:
            acts = self.load_actions(p)
            T = len(acts)
            starts = list(range(0, T - self.span, stride))
            per_cam = []
            for cam in self.cameras:
                Z = self.embeddings(clip, p, cam)
                def past_seg(t):
                    if t == 0:
                        return np.repeat(acts[0:1], 2, axis=0)   # 정지 과거
                    return acts[max(t - self.span, 0):t]
                Zp = np.stack([Z[max(t - self.span, 0)] for t in starts])
                Zc = np.stack([Z[t] for t in starts])
                Zn = np.stack([Z[t + self.span] for t in starts])
                Ap = np.stack([self.resample_chunk(past_seg(t)).ravel()
                               for t in starts])
                Af = np.stack([self.resample_chunk(acts[t:t + self.span]).ravel()
                               for t in starts])
                per_cam.append((Zp, Zc, Zn, Ap, Af))
            out.append(tuple(
                np.concatenate([pc[k] for pc in per_cam]).astype(np.float32)
                for k in range(5)))
        return out

    def build_proprio(self, files=None):
        files = files or self.episode_files()
        out = []
        for p in files:
            with h5py.File(p, "r") as f:
                qpos = f["observations/qpos"][:].astype(np.float32)
            starts = self.pair_starts(len(qpos))
            s = np.stack([qpos[i] for i in starts])
            out.append(np.concatenate([s] * len(self.cameras)))
        return out


if __name__ == "__main__":
    # 로더 단독 점검: python src/data/act_sim.py
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import yaml
    from core.clip_wrapper import ClipWrapper

    cfg = yaml.safe_load(open(Path(__file__).resolve().parents[2]
                              / "configs" / "phase1.yaml"))
    ds = ActSimDataset(cfg)
    files = ds.episode_files()
    print(f"episodes: {len(files)}")
    clip = ClipWrapper()
    eps = ds.build(clip, files[:2])
    Zt, Ztn, A = eps[0]
    print(f"pair: z {Zt.shape}, chunk {A.shape} (span {ds.span} steps @ {HZ}Hz)")
