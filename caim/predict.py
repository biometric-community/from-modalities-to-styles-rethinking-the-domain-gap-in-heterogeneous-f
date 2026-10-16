"""Print HR–LR cosine after CAIM adaptation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from caim.data.celeba import resolve_celeba_root
from caim.data.celeba_lrhr import CelebALRHR
from caim.models.caim_net import CAIMFaceNet


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=16)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = resolve_celeba_root(cfg)
    max_n = cfg["data"].get("max_images")
    ds = CelebALRHR(
        root,
        image_size=int(cfg["model"].get("image_size", 112)),
        lr_scale=int(cfg["model"].get("lr_scale", 4)),
        max_n=int(max_n) if max_n is not None else None,
        seed=int(cfg["train"].get("seed", 42)),
        split="test",
    )
    ckpt = torch.load(Path(cfg["paths"]["checkpoint_dir"]) / "caim.pt", map_location=device, weights_only=False)
    model = CAIMFaceNet(emb_dim=ckpt.get("emb_dim", 128), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    rows = []
    for i in range(min(args.limit, len(ds))):
        lr, hr, y = ds[i]
        eh = model.embed(hr.unsqueeze(0).to(device), target_modality=False)
        el = model.embed(lr.unsqueeze(0).to(device), target_modality=True)
        rows.append({"idx": i, "y": int(y), "cos_hr_lr": float((eh * el).sum().item())})
    out = Path(cfg["paths"]["pred_dir"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / "pair_cosines.json"
    path.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
