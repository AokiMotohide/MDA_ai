from collections import OrderedDict
from typing import Any, NamedTuple

import torch
import torchvision.transforms as tvf
from safetensors.torch import load_file

from add_ckpt_path import add_path_to_da3  # noqa: F401  side-effect: extends sys.path

from src.training.da3_wrapper import DA3Wrapper
from src.training.vggt_wrapper import VGGTWrapper


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_NORM = tvf.Compose(
    [
        tvf.ToTensor(),
        tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# model_name -> how to build it. Keys: wrapper, ckpt, (optional) config, kwargs, model_size.
MODELS = {
    "pretrained": dict(
        wrapper=DA3Wrapper,
        config="src/depth_anything_3/configs/da3-giant.yaml",
        ckpt="checkpoints/DA3-GIANT/model.safetensors",
    ),
    "mda_mog_sky_l2": dict(
        wrapper=DA3Wrapper,
        config="src/depth_anything_3/configs/da3-giant-cam-mog-sky.yaml",
        ckpt="checkpoints/MDA/DA3_MOG_Sky_LogL2.ckpt",
        kwargs={"loss_type": "logl2"},
    ),
    "vggt": dict(
        wrapper=VGGTWrapper,
        ckpt="checkpoints/vggt_1b_model.pt",
        model_size=518,
    ),
    "vggt_mog_l2": dict(
        wrapper=VGGTWrapper,
        ckpt="checkpoints/MDA/VGGT_MOG_LogL2.ckpt",
        kwargs={"is_mog": True, "loss_type": "logl2"},
        model_size=518,
    ),
}


class LoadedModel(NamedTuple):
    model: Any
    checkpoint_path: str
    patch_size: int
    img_norm: Any
    model_size: int


def _load_state_dict(path):
    if path.endswith(".safetensors"):
        raw = load_file(path, device="cpu")
    else:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    raw = raw.get("state_dict", raw)
    # strip wrapper prefixes (Lightning etc.) so keys match model.net
    out = {}
    for k, v in raw.items():
        for p in ("model.", "net.net.", "net."):
            if k.startswith(p):
                k = k[len(p):]
                break
        out[k] = v
    return out


def choose_model(model_name):
    if model_name not in MODELS:
        raise NotImplementedError(f"Unknown model {model_name!r}. Available: {list(MODELS)}")
    spec = MODELS[model_name]

    args = (spec["config"],) if "config" in spec else ()
    model = spec["wrapper"](*args, **spec.get("kwargs", {}))

    missing, unexpected = model.net.load_state_dict(_load_state_dict(spec["ckpt"]), strict=False)
    print("missing_keys", missing)
    print("unexpected_keys", unexpected)
    model = model.eval().to(device)

    return LoadedModel(model, spec["ckpt"], 14, IMAGENET_NORM, spec.get("model_size", 504))


def available_models():
    return list(MODELS)


CONFIGS = OrderedDict(
    crop_center_112=True,
    cam_inp=False,
    gt_cam_output=True,
    output_normalize=False,
)
