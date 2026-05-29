# mmsegmentation cu12 (torch 2.1.0 + cu121) task runner.
#
#   just sync        # provision .venv (cu12 deps) + put this mmseg source on the path (.pth)
#   just smoke       # import mmseg + FCNHead.forward on GPU
#   just env-doctor  # print python/torch/mmcv/mmengine/mmseg/gpu state

VENV := ".venv"
PY := justfile_directory() + "/" + VENV + "/bin/python"

default:
    @just --list

list:
    @just --list

# Provision the cu12 env (deps) and make this mmseg source importable (.pth).
sync:
    uv sync
    printf '%s\n' "{{ justfile_directory() }}" > "{{ VENV }}/lib/python3.10/site-packages/_cu12_src.pth"
    @echo "Synced: cu12 deps + mmseg source on path (.pth)."

# Read-only environment triage (imports from /tmp to avoid source shadowing).
env-doctor:
    @echo "=== gpu ==="
    @nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
    @echo ""
    @echo "=== python / packages ==="
    @if [ -x "{{ PY }}" ]; then \
        "{{ PY }}" --version; \
        for pkg in torch mmcv mmengine mmseg numpy; do \
            (cd /tmp && "{{ PY }}" -c "import importlib; m=importlib.import_module('$pkg'); print('$pkg', getattr(m,'__version__','?'))") 2>/dev/null || echo "$pkg IMPORT_ERROR"; \
        done; \
        (cd /tmp && "{{ PY }}" -c "import torch; print('cuda_available', torch.cuda.is_available(), '| cuda', torch.version.cuda)"); \
    else \
        echo "{{ PY }} missing — run 'just sync' first."; \
    fi

# Run the mmseg GPU smoke (FCNHead forward on cuda).
smoke OUT="/tmp/smoke_mmseg_cu12.json": sync
    "{{ PY }}" tools/cu12/smoke_mmseg.py --output-json "{{ OUT }}"
