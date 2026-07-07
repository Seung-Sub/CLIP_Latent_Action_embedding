"""LIBERO 데모 로더 — ActSimDataset과 동일 인터페이스 (임베딩 규격으로 통일).

데이터 형식 (robomimic HDF5, 태스크당 1파일 × 데모 50개):
  <suite>/<task>_demo.hdf5
    data/demo_K/obs/agentview_rgb : (T, H, W, 3) uint8, 20Hz
    data/demo_K/actions           : (T, 7)  — OSC 델타 (Δpos 3 + Δrot 3 + 그리퍼 1)

에피소드 단위 = (hdf5 경로, demo 키) 튜플. 언어 지시문은 태스크 파일명 기준으로
CLIP 텍스트 임베딩을 캐시한다 (저장만 — 정책 사용은 이후 단계).
"""
import os
import re
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

HZ = 20.0


class LiberoDataset:
    def __init__(self, cfg):
        d = cfg["data"]
        roots = d["root"] if isinstance(d["root"], list) else [d["root"]]
        self.roots = [Path(os.path.expanduser(r)) for r in roots]
        self.camera = d.get("camera", "agentview_rgb")
        self.wrist_camera = d.get("wrist_camera")    # 예: eye_in_hand_rgb (없으면 미사용)
        self.chunk_sec = float(d["chunk_sec"])
        self.n_chunk = int(d["n_chunk"])
        self.cache_dir = Path(os.path.expanduser(d["cache_dir"]))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.span = max(2, int(round(self.chunk_sec * HZ)))
        self.stride = max(1, self.span // 8)
        # D1: OpenVLA식 no-op 필터 (‖a[:-1]‖<ε AND 그리퍼 명령 불변 스텝 제거, None=off)
        # float 강제: yaml/--set 경유 시 "1e-4"가 str로 들어오는 경우 방지
        eps = d.get("noop_filter_eps", None)
        self.noop_eps = None if eps is None else float(eps)

    # ---------- 에피소드 열거: (파일, demo키) ----------

    def episode_files(self):
        eps = []
        for r in self.roots:
            for f in sorted(r.glob("*.hdf5")):
                with h5py.File(f, "r") as h:
                    demos = sorted(h["data"].keys(),
                                   key=lambda k: int(k.split("_")[-1]))
                eps += [(f, k) for k in demos]
        return eps

    @staticmethod
    def _key(ep):
        path, demo = ep
        return f"{path.stem}_{demo}"

    # ---------- 원시 접근 ----------

    def load_actions(self, ep):
        path, demo = ep
        with h5py.File(path, "r") as h:
            return h[f"data/{demo}/actions"][:].astype(np.float64)

    def load_frames(self, ep, camera=None):
        path, demo = ep
        with h5py.File(path, "r") as h:
            return h[f"data/{demo}/obs/{camera or self.camera}"][:]

    def instruction(self, ep):
        """태스크 파일명 → 자연어 지시문 (예: pick_up_the_..._demo.hdf5)."""
        path, _ = ep
        name = re.sub(r"_demo$", "", path.stem)
        # SCENE 접두어 제거 (예: LIVING_ROOM_SCENE1_)
        name = re.sub(r"^[A-Z0-9_]+SCENE\d+_", "", name)
        return name.replace("_", " ")

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

    def embeddings(self, clip, ep, camera=None):
        camera = camera or self.camera
        cache = self._cache_path(clip, self._key(ep) + f"_{camera}.npz")
        if cache.exists():
            return np.load(cache)["Z"]
        frames = [Image.fromarray(im) for im in self.load_frames(ep, camera)]
        Z = []
        for i in range(0, len(frames), 64):
            Z.append(clip.encode_images(frames[i:i + 64])["embeds"])
        Z = np.concatenate(Z)
        np.savez_compressed(cache, Z=Z)
        return Z

    def embeddings_meanpatch(self, encoder, ep, camera=None):
        """C7: 패치 토큰 평균 임베딩 (encoder.encode_images의 tokens 사용, CLS 제외)."""
        camera = camera or self.camera
        cache = self._cache_path(encoder, self._key(ep) + f"_{camera}_mp.npz")
        if cache.exists():
            return np.load(cache)["Z"]
        frames = [Image.fromarray(im) for im in self.load_frames(ep, camera)]
        Z = []
        for i in range(0, len(frames), 32):
            out = encoder.encode_images(frames[i:i + 32])
            assert out.get("tokens") is not None, "encoder가 패치 토큰 미반환"
            Z.append(out["tokens"][:, 1:].mean(axis=1))   # CLS 제외 평균
        Z = np.concatenate(Z).astype(np.float32)
        np.savez_compressed(cache, Z=Z)
        return Z

    def instruction_embedding(self, clip, ep):
        path, _ = ep
        cache = self._cache_path(clip, path.stem + "_lang.npz")
        if cache.exists():
            return np.load(cache)["L"]
        L = clip.encode_texts([self.instruction(ep)])["embeds"][0]
        np.savez_compressed(cache, L=L)
        return L

    # ---------- no-op 필터 (검증 노트 §1: OpenVLA regenerate 스크립트 규칙) ----------

    def keep_indices(self, acts):
        """no-op이 아닌 스텝 인덱스. self.noop_eps=None이면 전체."""
        if self.noop_eps is None:
            return np.arange(len(acts))
        norm = np.linalg.norm(acts[:, :-1], axis=1)
        same_grip = np.zeros(len(acts), bool)
        same_grip[1:] = acts[1:, -1] == acts[:-1, -1]    # 첫 스텝은 norm 기준만
        noop = (norm < self.noop_eps) & same_grip
        noop[0] = norm[0] < self.noop_eps
        return np.where(~noop)[0]

    def _filtered(self, clip, ep):
        """(acts, Z) 동기 필터 — 프레임·액션 병렬 배열에서 no-op 스텝 동시 제거."""
        acts = self.load_actions(ep)
        Z = self.embeddings(clip, ep)
        keep = self.keep_indices(acts)
        if len(keep) < len(acts):
            acts, Z = acts[keep], Z[keep]
        return acts, Z

    # ---------- 학습 쌍 생성 (act_sim과 동일 수식) ----------

    def resample_chunk(self, seg):
        src = np.linspace(0, len(seg) - 1, self.n_chunk)
        lo = np.floor(src).astype(int)
        hi = np.minimum(lo + 1, len(seg) - 1)
        w = (src - lo)[:, None]
        return seg[lo] * (1 - w) + seg[hi] * w

    def build(self, clip, files=None, verbose=True):
        files = files or self.episode_files()
        out = []
        for ep in files:
            acts, Z = self._filtered(clip, ep)
            T = len(acts)
            starts = list(range(0, T - self.span, self.stride))
            Zt = np.stack([Z[t] for t in starts])
            Ztn = np.stack([Z[t + self.span] for t in starts])
            A = np.stack([self.resample_chunk(acts[t:t + self.span]).ravel()
                          for t in starts])
            out.append((Zt.astype(np.float32), Ztn.astype(np.float32),
                        A.astype(np.float32)))
            if verbose:
                print(f"  {self._key(ep)}: T={T}, pairs {len(starts)}")
        return out

    def build_policy_samples(self, clip, files=None, stride=2):
        """연속 윈도우 삼중쌍 (경계 포함 — 롤아웃 부트스트랩 분포 커버)."""
        files = files or self.episode_files()
        out = []
        for ep in files:
            acts, Z = self._filtered(clip, ep)
            T = len(acts)
            starts = list(range(0, T - self.span, stride))

            def past_seg(t):
                if t == 0:
                    return np.repeat(acts[0:1], 2, axis=0)
                return acts[max(t - self.span, 0):t]

            Zp = np.stack([Z[max(t - self.span, 0)] for t in starts])
            Zc = np.stack([Z[t] for t in starts])
            Zn = np.stack([Z[t + self.span] for t in starts])
            Ap = np.stack([self.resample_chunk(past_seg(t)).ravel()
                           for t in starts])
            Af = np.stack([self.resample_chunk(acts[t:t + self.span]).ravel()
                           for t in starts])
            arrs = [Zp, Zc, Zn, Ap, Af]
            if self.wrist_camera:                    # 6번째: 손목캠 z_t (정책 토큰용)
                Zw = self.embeddings(clip, ep, self.wrist_camera)
                arrs.append(np.stack([Zw[t] for t in starts]))
            out.append(tuple(x.astype(np.float32) for x in arrs))
        return out

    def build_proprio(self, files=None, stride=2, fields=None):
        """정책 proprio 토큰용 (S1.v2 §4): 기본 joint(7)+gripper(2), no-op 필터·starts를
        build_policy_samples와 동일하게 적용해 샘플 정렬 보장.
        fields: obs 키 리스트로 구성 선택 (예: ["gripper_states"] = (a2) 2D 변형)."""
        files = files or self.episode_files()
        fields = fields or ["joint_states", "gripper_states"]
        out = []
        for ep in files:
            path, demo = ep
            with h5py.File(path, "r") as h:
                P = np.concatenate([h[f"data/{demo}/obs/{f}"][:] for f in fields],
                                   axis=1)
            P = P[self.keep_indices(self.load_actions(ep))]
            starts = list(range(0, len(P) - self.span, stride))
            out.append(np.stack([P[t] for t in starts]).astype(np.float32))
        return out


if __name__ == "__main__":
    # 로더 단독 점검: python src/data/libero.py
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import yaml
    from core.clip_wrapper import ClipWrapper

    cfg = yaml.safe_load(open(Path(__file__).resolve().parents[2]
                              / "configs" / "phase1_libero.yaml"))
    ds = LiberoDataset(cfg)
    eps = ds.episode_files()
    print(f"episodes: {len(eps)} (파일 {len(set(p for p, _ in eps))}개 태스크)")
    clip = ClipWrapper()
    print("지시문 예:", ds.instruction(eps[0]))
    pairs = ds.build(clip, eps[:2])
    Zt, Ztn, A = pairs[0]
    print(f"pair: z {Zt.shape}, chunk {A.shape} (span {ds.span} steps @ {HZ}Hz)")
