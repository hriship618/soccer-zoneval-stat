from __future__ import annotations

from pathlib import Path
import numpy as np


class CudaUnavailable(RuntimeError):
    pass


def is_available() -> bool:
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def leave_one_out_cuda(positions: np.ndarray, velocities: np.ndarray, teams: np.ndarray, zone_values: np.ndarray, grid_x: int = 32, grid_y: int = 21) -> np.ndarray:
    """Execute the in-kernel-reduction CUDA implementation through CuPy."""
    try:
        import cupy as cp
    except ImportError as exc:
        raise CudaUnavailable("install the 'cuda' extra and use a CUDA-capable host") from exc
    if not is_available():
        raise CudaUnavailable("no CUDA device is available")
    source = (Path(__file__).with_name("cuda") / "pitch_control.cu").read_text(encoding="utf-8")
    kernel = cp.RawKernel(source, "zcpv_leave_one_out", options=("--std=c++14",))
    pos = cp.asarray(positions, dtype=cp.float32)
    vel = cp.asarray(velocities, dtype=cp.float32)
    team = cp.asarray(teams, dtype=cp.int32)
    values = cp.asarray(zone_values, dtype=cp.float32)
    output = cp.zeros((len(positions), 22, 12), dtype=cp.float32)
    cells = grid_x * grid_y
    threads = 256
    kernel(((len(positions) * cells + threads - 1) // threads,), (threads,), (pos, vel, team, values, output, np.int32(len(positions)), np.int32(grid_x), np.int32(grid_y)))
    cp.cuda.runtime.deviceSynchronize()
    ix = np.arange(grid_x)[None, :]
    iy = np.arange(grid_y)[:, None]
    zone_ids = np.minimum(3, ix * 4 // grid_x) * 3 + np.minimum(2, iy * 3 // grid_y)
    zone_cell_counts = np.bincount(zone_ids.ravel(), minlength=12)
    return cp.asnumpy(output) / np.maximum(zone_cell_counts[None, None, :], 1)
