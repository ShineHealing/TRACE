"""Frozen ResNet50, UNI, and UNI2-h feature encoders."""

import os
import logging
import torch
import torch.nn as nn
from torchvision import transforms, models
from typing import Optional

logger = logging.getLogger('trace')


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


UNI_CONFIGS = {
    'uni': {
        'repo_id': 'MahmoodLab/UNI',
        'feature_dim': 1024,
        'model_name': 'vit_large_patch16_224',
        'description': 'UNI ViT-Large (1024-dim)',
    },
    'uni2-h': {
        'repo_id': 'MahmoodLab/UNI2-h',
        'feature_dim': 1536,
        'model_name': 'vit_giant_patch14_dinov2',
        'description': 'UNI2-h ViT-Huge (1536-dim)',
    },
}


def get_uni_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class UNIEncoder(nn.Module):
    """Frozen pathology image encoder."""
    
    def __init__(
        self,
        backbone: str = 'uni',
        pretrained: bool = True,
        output_dim: Optional[int] = None,
        strict_token_check: bool = True,
    ):
        super(UNIEncoder, self).__init__()
        self.backbone_name = backbone
        self.loaded_pretrained = False
        self.strict_token_check = bool(strict_token_check)
        self._token_layout_checked = False
        self._expected_total_tokens: Optional[int] = None
        self._expected_prefix_tokens: Optional[int] = None
        self._expected_patch_tokens: Optional[int] = None
        
        if backbone == 'resnet50':
            self._init_resnet(pretrained, output_dim)
        elif backbone in ['uni', 'uni2-h']:
            self._init_uni_official(backbone, pretrained, output_dim)
        else:
            raise ValueError(f"Unknown backbone: {backbone}. Choose from: resnet50, uni, uni2-h")

        
        self.transform = get_uni_transform()
        
        self.output_dim = self.feature_dim

    @torch.no_grad()
    def _strict_check_uni2h_token_layout(self):
        if str(self.backbone_name).lower().strip() != 'uni2-h':
            return
        if not hasattr(self, 'model') or self.model is None:
            return
        if not hasattr(self.model, 'forward_features'):
            raise RuntimeError("UNI2-h strict token check requires a timm ViT with forward_features().")

        m = self.model
        no_embed_class = bool(getattr(m, 'no_embed_class', False))
        has_cls = bool(getattr(m, 'cls_token', None) is not None)
        cls = 1 if has_cls else 0

        reg_param = getattr(m, 'reg_token', None)
        reg = 0
        if torch.is_tensor(reg_param) and reg_param.dim() == 3:
            reg = int(reg_param.shape[1])

        npt = getattr(m, 'num_prefix_tokens', None)
        try:
            npt = int(npt) if npt is not None else int(reg + cls)
        except Exception:
            npt = int(reg + cls)

        num_patches = None
        pe = getattr(m, 'patch_embed', None)
        if pe is not None:
            num_patches = getattr(pe, 'num_patches', None)
            if num_patches is None:
                gs = getattr(pe, 'grid_size', None)
                if gs is not None and len(gs) == 2:
                    try:
                        num_patches = int(gs[0]) * int(gs[1])
                    except Exception:
                        num_patches = None
        if num_patches is None:
            raise RuntimeError("Cannot infer num_patches for UNI2-h; timm model structure changed.")
        try:
            num_patches = int(num_patches)
        except Exception:
            raise RuntimeError(f"Cannot cast num_patches={num_patches} to int for UNI2-h")

        expected_total = int(num_patches + npt)
        expected_prefix = int(npt)

        device = next(m.parameters()).device if any(True for _ in m.parameters()) else torch.device('cpu')
        dummy = torch.zeros(1, 3, 224, 224, device=device, dtype=torch.float32)
        m.eval()
        out = m.forward_features(dummy)
        if isinstance(out, (tuple, list)):
            out = out[0]
        elif isinstance(out, dict):
            for key in ['x', 'last_hidden_state', 'features', 'tokens']:
                if key in out:
                    out = out[key]
                    break

        if not torch.is_tensor(out) or out.dim() != 3:
            raise RuntimeError(
                f"UNI2-h token check failed: forward_features returned {type(out)} with shape="
                f"{tuple(out.shape) if torch.is_tensor(out) else 'N/A'}."
            )

        L = int(out.shape[1])
        D = int(out.shape[2])

        if num_patches != 256:
            raise RuntimeError(
                f"UNI2-h token layout mismatch: expected num_patches=256 (224/14)^2, got num_patches={num_patches}."
            )
        if D != 1536:
            raise RuntimeError(
                f"UNI2-h embed dim mismatch: expected D=1536, got D={D}."
            )
        if not has_cls or cls != 1:
            raise RuntimeError("UNI2-h token layout mismatch: expected a cls_token to exist.")
        if reg != 8:
            raise RuntimeError(
                f"UNI2-h token layout mismatch: expected reg_token count=8, got {reg}."
            )
        if expected_prefix != 9:
            raise RuntimeError(
                f"UNI2-h prefix token count mismatch: expected prefix=9 (1 cls + 8 reg), got prefix={expected_prefix}."
            )
        if L != expected_total:
            raise RuntimeError(
                f"UNI2-h token length mismatch (internal). Expected {expected_total}, got {L}. "
                f"(num_patches={num_patches}, prefix={expected_prefix}, reg={reg}, cls={cls}, no_embed_class={no_embed_class})"
            )

        pe_pos = getattr(m, 'pos_embed', None)
        if not torch.is_tensor(pe_pos) or pe_pos.dim() != 3:
            raise RuntimeError("UNI2-h pos_embed missing or has unexpected shape.")
        if no_embed_class:
            if int(pe_pos.shape[1]) != int(num_patches):
                raise RuntimeError(
                    f"UNI2-h pos_embed length mismatch: expected {num_patches} (patch-only when no_embed_class=True), got {int(pe_pos.shape[1])}."
                )

        self._token_layout_checked = True
        self._expected_total_tokens = int(expected_total)
        self._expected_prefix_tokens = int(expected_prefix)
        self._expected_patch_tokens = int(num_patches)
        logger.info(
            "[UNIEncoder][StrictCheck] UNI2-h token layout OK: total=%d, prefix=%d, patch=%d (reg=%d, cls=%d, no_embed_class=%s).",
            L,
            expected_prefix,
            num_patches,
            reg,
            cls,
            str(no_embed_class),
        )
    
    def _init_resnet(self, pretrained: bool, output_dim: Optional[int]):
        native_dim = 2048
        if output_dim is not None and int(output_dim) != native_dim:
            raise ValueError(f'ResNet50 uses its native {native_dim}-dimensional features')
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.model = models.resnet50(weights=weights)
        self.model.fc = nn.Identity()
        self.projection = None
        self.feature_dim = native_dim
        self.backbone_name = 'resnet50'
    
    def _init_uni_official(self, backbone: str, pretrained: bool, output_dim: Optional[int]):
        import timm
        
        config = UNI_CONFIGS[backbone]
        self.feature_dim = config['feature_dim']
        if output_dim is not None and int(output_dim) != self.feature_dim:
            raise ValueError(
                f'{backbone} uses its native {self.feature_dim}-dimensional features'
            )
        
        if pretrained:
            try:
                
                
                if backbone == 'uni':
                    def _resolve_local_uni_weights_path() -> str | None:
                        p = os.getenv('GENAR_UNI_WEIGHTS_PATH') or os.getenv('UNI_WEIGHTS_PATH')
                        if p:
                            p = os.path.expanduser(str(p))
                            if os.path.isdir(p):
                                p2 = os.path.join(p, 'pytorch_model.bin')
                                if os.path.exists(p2):
                                    return p2
                            if os.path.exists(p):
                                return p
                            raise FileNotFoundError(f"UNI weights not found at path={p}")
                        try:
                            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
                            cand = os.path.join(repo_root, 'weights', 'UNI', 'pytorch_model.bin')
                            if os.path.exists(cand):
                                return cand
                        except Exception:
                            pass
                        try:
                            cand = os.path.join(os.getcwd(), 'weights', 'UNI', 'pytorch_model.bin')
                            if os.path.exists(cand):
                                return cand
                        except Exception:
                            pass
                        return None

                    weights_path = _resolve_local_uni_weights_path()
                    if weights_path is not None:
                        print(f"Loading {config['description']} from local weights...")
                        print(f"[UNI] Using weights: {weights_path}")
                        self.model = timm.create_model(
                            'vit_large_patch16_224',
                            pretrained=False,
                            num_classes=0,
                            init_values=1e-5,
                            dynamic_img_size=True,
                        )
                        state_dict = torch.load(weights_path, map_location='cpu')
                        self.model.load_state_dict(state_dict, strict=True)
                    else:
                        print(f"Loading {config['description']} from HuggingFace...")
                        
                        self.model = timm.create_model(
                            "hf-hub:MahmoodLab/UNI",
                            pretrained=True,
                            init_values=1e-5,
                            dynamic_img_size=True,
                        )
                else:
                    def _resolve_local_uni2h_weights_path() -> str | None:
                        
                        p = os.getenv('GENAR_UNI2H_WEIGHTS_PATH') or os.getenv('UNI2H_WEIGHTS_PATH')
                        if p:
                            p = os.path.expanduser(str(p))
                            if os.path.isdir(p):
                                p2 = os.path.join(p, 'pytorch_model.bin')
                                if os.path.exists(p2):
                                    return p2
                            if os.path.exists(p):
                                return p
                            raise FileNotFoundError(f"UNI2-h weights not found at path={p}")

                        try:
                            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
                            cand = os.path.join(repo_root, 'weights', 'UNI2-h', 'pytorch_model.bin')
                            if os.path.exists(cand):
                                return cand
                        except Exception:
                            pass

                        
                        try:
                            cand = os.path.join(os.getcwd(), 'weights', 'UNI2-h', 'pytorch_model.bin')
                            if os.path.exists(cand):
                                return cand
                        except Exception:
                            pass
                        return None

                    weights_path = _resolve_local_uni2h_weights_path()
                    if weights_path is None:
                        from huggingface_hub import hf_hub_download

                        
                        
                        
                        local_only = bool(int(os.getenv('HF_HUB_OFFLINE', '0') or '0'))
                        weights_path = hf_hub_download(
                            repo_id="MahmoodLab/UNI2-h",
                            filename="pytorch_model.bin",
                            local_files_only=local_only,
                        )

                        print(f"Loading {config['description']} from HuggingFace...")
                    else:
                        print(f"Loading {config['description']} from local weights...")

                    print(f"[UNI2-h] Using weights: {weights_path}")

                    from timm.models.vision_transformer import VisionTransformer
                    from timm.layers import SwiGLUPacked

                    self.model = VisionTransformer(
                        img_size=224,
                        patch_size=14,
                        in_chans=3,
                        num_classes=0,
                        global_pool='',
                        embed_dim=1536,
                        depth=24,
                        num_heads=24,
                        
                        mlp_ratio=16 / 3,
                        qkv_bias=True,
                        qk_norm=False,
                        init_values=1e-5,
                        class_token=True,
                        no_embed_class=True,
                        reg_tokens=8,
                        mlp_layer=SwiGLUPacked,
                        dynamic_img_size=True,
                    )

                    state_dict = torch.load(weights_path, map_location='cpu')
                    self.model.load_state_dict(state_dict, strict=True)

                    if self.strict_token_check:
                        self._strict_check_uni2h_token_layout()
                
                print(f"Successfully loaded {backbone} pretrained weights")
                self.loaded_pretrained = True
                
            except Exception as e:
                raise RuntimeError(
                    f"Could not load pretrained {backbone}. Accept the official "
                    "Hugging Face license, authenticate, or provide the documented local weights path."
                ) from e
        else:
            raise ValueError("TRACE requires pretrained UNI/UNI2-h weights")

        if backbone == 'uni2-h' and self.strict_token_check:
            self._strict_check_uni2h_token_layout()
        
        self.projection = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._extract_backbone_features(x)

        if self.projection is not None:
            features = self.projection(features)

        return features

    def _extract_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        def _infer_num_prefix_tokens_from_model() -> int:
            m = getattr(self, 'model', None)
            if m is None:
                return 0

            npt = getattr(m, 'num_prefix_tokens', None)
            try:
                if npt is not None:
                    return int(npt)
            except Exception:
                pass

            npt = getattr(m, 'num_prefix_tokens', None)
            try:
                if npt is not None:
                    return int(npt)
            except Exception:
                pass

            reg_param = getattr(m, 'reg_token', None)
            reg = int(reg_param.shape[1]) if (torch.is_tensor(reg_param) and reg_param.dim() == 3) else 0
            cls = 1 if (getattr(m, 'cls_token', None) is not None) else 0
            return int(reg + cls)

        
        if hasattr(self.model, 'forward_features') and self.backbone_name in ['uni', 'uni2-h']:
            out = self.model.forward_features(x)

            if isinstance(out, (tuple, list)):
                out = out[0]
            elif isinstance(out, dict):
                for key in ['x', 'last_hidden_state', 'features', 'tokens']:
                    if key in out:
                        out = out[key]
                        break

            if not torch.is_tensor(out):
                raise TypeError(f"Unexpected forward_features output type: {type(out)}")

            if out.dim() == 3:
                if self.backbone_name == 'uni2-h':
                    prefix = (
                        int(self._expected_prefix_tokens)
                        if (self.strict_token_check and self._token_layout_checked and self._expected_prefix_tokens is not None)
                        else _infer_num_prefix_tokens_from_model()
                    )
                    if self.strict_token_check and self._token_layout_checked and self._expected_total_tokens is not None:
                        if int(out.shape[1]) != int(self._expected_total_tokens):
                            raise RuntimeError(
                                f"UNI2-h token length drift detected at runtime: expected {int(self._expected_total_tokens)}, got {int(out.shape[1])}. "
                                "Refuse to proceed to avoid corrupting offline features."
                            )
                    patch_tokens = out[:, prefix:, :]
                    if patch_tokens.numel() == 0:
                        return out[:, 0, :]
                    return patch_tokens.mean(dim=1)
                
                return out[:, 0, :]
            if out.dim() == 2:
                return out

            raise ValueError(f"Unexpected forward_features tensor shape: {tuple(out.shape)}")

        
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        elif isinstance(out, dict):
            for key in ['pooler_output', 'last_hidden_state', 'logits', 'x', 'features']:
                if key in out:
                    out = out[key]
                    break

        if not torch.is_tensor(out):
            raise TypeError(f"Unexpected model output type: {type(out)}")

        if out.dim() == 3:
            if self.backbone_name == 'uni2-h':
                prefix = (
                    int(self._expected_prefix_tokens)
                    if (self.strict_token_check and self._token_layout_checked and self._expected_prefix_tokens is not None)
                    else _infer_num_prefix_tokens_from_model()
                )
                if self.strict_token_check and self._token_layout_checked and self._expected_total_tokens is not None:
                    if int(out.shape[1]) != int(self._expected_total_tokens):
                        raise RuntimeError(
                            f"UNI2-h token length drift detected at runtime: expected {int(self._expected_total_tokens)}, got {int(out.shape[1])}. "
                            "Refuse to proceed to avoid corrupting offline features."
                        )
                patch_tokens = out[:, prefix:, :]
                if patch_tokens.numel() == 0:
                    return out[:, 0, :]
                return patch_tokens.mean(dim=1)
            return out[:, 0, :]
        if out.dim() == 2:
            return out

        raise ValueError(f"Unexpected model output shape: {tuple(out.shape)}")
    
    def get_feature_dim(self) -> int:
        return self.feature_dim
