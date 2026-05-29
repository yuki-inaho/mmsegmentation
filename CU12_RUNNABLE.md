# Running this mmsegmentation fork on cu12 (torch 2.1.0 + cu121)

Part of the OpenMMLab cu12 migration (end goal: train
`tomato_pipe_rgbd_rtmdet_obb_training` on cu12). Validated stack:
**Python 3.10 / torch 2.1.0+cu121 / mmcv 2.2.0 / mmengine 0.10.7 / mmseg 1.2.2**.

## Quick start (uv)

```bash
just sync          # uv sync (cu12 deps incl. mmcv 2.2.0 cu121 wheel) + this source on path (.pth)
just smoke         # import mmseg + FCNHead.forward on GPU -> cuda_op_device: cuda
just env-doctor    # GPU + versions
```

`pyproject.toml` is a **virtual** uv project. mmcv comes from the prebuilt cu121
wheel; `MMCV_MAX='2.3.0'` / `MMENGINE_MAX='1.0.0'` (mmseg `__init__.py` guards) are
satisfied by mmcv 2.2.0 + mmengine 0.10.7. `ftfy`/`regex` are included because
mmseg's CLIP-derived text imports need them. mmseg is pure Python, installed via
`.pth` rather than a setuptools build.
