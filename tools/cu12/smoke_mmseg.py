#!/usr/bin/env python3
"""mmseg cu12 GPU smoke: forward a minimal head module on CUDA tensors."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _smoke_common import run_smoke, standard_argv  # noqa: E402


def runner() -> dict[str, Any]:
    import torch
    from mmseg.models.decode_heads.fcn_head import FCNHead

    head = FCNHead(
        in_channels=8,
        channels=4,
        num_convs=1,
        num_classes=2,
        in_index=0,
        kernel_size=3,
        concat_input=False,
        dropout_ratio=0.0,
        norm_cfg=dict(type="BN"),
    ).cuda()
    head.eval()
    feat = torch.randn(1, 8, 16, 16, device="cuda")
    with torch.no_grad():
        out = head([feat])
    torch.cuda.synchronize()
    return {
        "op": "mmseg.models.decode_heads.FCNHead.forward",
        "cuda_op_device": out.device.type,
        "output_shape": list(out.shape),
        "output_dtype": str(out.dtype),
    }


def main() -> int:
    args = standard_argv("mmseg cu12 GPU smoke")
    return run_smoke(module_name="mmseg", runner=runner, output_json=args.output_json)


if __name__ == "__main__":
    sys.exit(main())
