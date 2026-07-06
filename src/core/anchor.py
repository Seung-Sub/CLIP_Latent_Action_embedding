"""앵커(관측/텍스트 임베딩 백본) 추상화 — 계획서 Phase 0.1.

인터페이스 (모든 구현체 공통):
  encode_images(pil_images) -> {"embeds": (N, dim) float32, "tokens": (N, P, patch_dim)|None}
  encode_texts(texts)       -> {"embeds": (N, dim_text) float32}  (has_text=False면 예외)
  속성: dim / patch_dim / has_text / id
  옵션: projection {joint, pre} / normalize {true, false}
  cache_key: "{id}/{projection}/{norm|raw}" — 임베딩 캐시 디렉터리 분리 키

선택은 config의 anchor 섹션 (없으면 기본 = 기존과 동일):
  anchor: {name: clip|siglip2|dinov2, projection: joint, normalize: true, model_dir: ...}

기본값(ClipAnchor, joint, normalize=True)은 기존 ClipWrapper와 출력이 완전 동일
→ 기존 평면 캐시(libero_emb/*.npz 등) 재사용 가능 (로더의 하위호환 폴백 참조).
ImageBindAnchor는 비상업 라이선스로 구현 보류 (계획서 리스크 (iv) — SigLIP2로 대체).
"""
import numpy as np
import torch

from core.config import load_config


class BaseAnchor:
    id = "base"
    dim = None
    patch_dim = None
    has_text = False

    def __init__(self, projection="joint", normalize=True):
        assert projection in ("joint", "pre"), projection
        self.projection = projection
        self.normalize = normalize

    @property
    def cache_key(self):
        return f"{self.id}/{self.projection}/{'norm' if self.normalize else 'raw'}"

    def _post(self, x):
        x = x.float()
        if self.normalize:
            x = torch.nn.functional.normalize(x, dim=-1)
        return x.cpu().numpy()

    def encode_texts(self, texts):
        raise RuntimeError(f"{self.id}: 텍스트 인코더 없음 (has_text=False) — "
                           "lang_token 조건은 언어 정렬 앵커에서만 가능")


class ClipAnchor(BaseAnchor):
    """frozen CLIP ViT-L/14. joint: projection 후 768 / pre: vision pooler 1024 (텍스트 768)."""
    id = "clip-vit-l-14"
    has_text = True
    patch_dim = 1024

    def __init__(self, projection="joint", normalize=True, cfg=None):
        super().__init__(projection, normalize)
        from core.clip_wrapper import ClipWrapper
        self._w = ClipWrapper(cfg if (cfg and "clip" in cfg) else None)
        self.dim = 768 if projection == "joint" else 1024
        self.dim_text = 768
        self.device = self._w.device

    @torch.no_grad()
    def encode_images(self, pil_images):
        if self.projection == "joint" and self.normalize:
            return self._w.encode_images(pil_images)      # 기존 경로와 완전 동일
        m, proc = self._w.model, self._w.processor
        inputs = proc(images=pil_images, return_tensors="pt").to(self.device)
        vout = m.vision_model(pixel_values=inputs["pixel_values"].to(m.dtype))
        pooled = (vout.pooler_output if self.projection == "pre"
                  else m.visual_projection(vout.pooler_output))
        tokens = (vout.last_hidden_state.float().cpu().numpy()
                  if self._w.save_tokens else None)
        return {"embeds": self._post(pooled), "tokens": tokens}

    @torch.no_grad()
    def encode_texts(self, texts):
        if self.projection == "joint" and self.normalize:
            return self._w.encode_texts(texts)
        m, proc = self._w.model, self._w.processor
        inputs = proc(text=texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=77).to(self.device)
        tout = m.text_model(input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"])
        pooled = (tout.pooler_output if self.projection == "pre"
                  else m.text_projection(tout.pooler_output))
        return {"embeds": self._post(pooled), "tokens": None}


class Siglip2Anchor(BaseAnchor):
    """SigLIP2 so400m — 언어 정렬 앵커 후보 A군. joint(공유 공간)만 지원."""
    id = "siglip2-so400m"
    has_text = True
    patch_dim = 1152

    def __init__(self, projection="joint", normalize=True, model_dir=None):
        if projection != "joint":
            raise ValueError("siglip2: projection=pre 미지원 (공유 공간 헤드 일체형)")
        super().__init__(projection, normalize)
        from transformers import AutoModel, AutoProcessor
        src = model_dir or "google/siglip2-so400m-patch14-384"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(
            src, dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(src)
        self.dim = self.model.config.vision_config.hidden_size    # 1152
        self.dim_text = self.dim

    @torch.no_grad()
    def encode_images(self, pil_images):
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        emb = self.model.get_image_features(
            pixel_values=inputs["pixel_values"].to(self.model.dtype))
        return {"embeds": self._post(emb), "tokens": None}

    @torch.no_grad()
    def encode_texts(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding="max_length",
                                truncation=True).to(self.device)
        emb = self.model.get_text_features(input_ids=inputs["input_ids"])
        return {"embeds": self._post(emb), "tokens": None}


class Dinov2Anchor(BaseAnchor):
    """DINOv2-L — 무언어 대조 앵커 (H2). 임베딩 = CLS(pooler), projection 구분 없음(pre 고정)."""
    id = "dinov2-large"
    has_text = False
    patch_dim = 1024

    def __init__(self, projection="pre", normalize=True, model_dir=None):
        super().__init__("pre", normalize)     # joint 공간 없음 → pre 고정
        from transformers import AutoImageProcessor, AutoModel
        src = model_dir or "facebook/dinov2-large"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(
            src, dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(src)
        self.dim = self.model.config.hidden_size                  # 1024

    @torch.no_grad()
    def encode_images(self, pil_images):
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        out = self.model(pixel_values=inputs["pixel_values"].to(self.model.dtype))
        return {"embeds": self._post(out.pooler_output),
                "tokens": out.last_hidden_state.float().cpu().numpy()}


_REGISTRY = {"clip": ClipAnchor, "siglip2": Siglip2Anchor, "dinov2": Dinov2Anchor}


def get_anchor(cfg=None):
    """config의 anchor 섹션으로 앵커 선택. 섹션 없으면 기존과 동일한 ClipAnchor."""
    cfg = cfg or load_config()
    a = cfg.get("anchor") or {}
    name = a.get("name", "clip")
    if name not in _REGISTRY:
        raise KeyError(f"unknown anchor '{name}' (지원: {sorted(_REGISTRY)})")
    kwargs = {"projection": a.get("projection", "joint" if name != "dinov2" else "pre"),
              "normalize": a.get("normalize", True)}
    if name == "clip":
        return ClipAnchor(**kwargs, cfg=cfg)
    return _REGISTRY[name](**kwargs, model_dir=a.get("model_dir"))
