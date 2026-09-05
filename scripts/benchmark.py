from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np

from zcpv.cuda_backend import CudaUnavailable, is_available, leave_one_out_cuda
from zcpv.pitch_control import PitchControlConfig, leave_one_out_zone_values, naive_leave_one_out_zone_values


def synthetic_tracking(frames: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    base_x = np.r_[np.linspace(7, 49, 11), np.linspace(98, 56, 11)]
    base_y = np.tile(np.linspace(5, 63, 11), 2)
    velocity = rng.normal(0, 1.7, (frames, 22, 2)).astype(np.float32)
    position = np.empty_like(velocity)
    position[0] = np.column_stack((base_x, base_y))
    for f in range(1, frames):
        position[f] = np.clip(position[f - 1] + velocity[f] * .1, (0, 0), (105, 68))
    return position, velocity, np.r_[np.zeros(11, dtype=np.int8), np.ones(11, dtype=np.int8)]


def timed(callable_, repeats: int = 3):
    samples=[]; result=None
    for _ in range(repeats):
        start=perf_counter(); result=callable_(); samples.append(perf_counter()-start)
    return min(samples), result


def main():
    parser=argparse.ArgumentParser(description="Benchmark ZCPV pitch control on deterministic synthetic tracking data.")
    parser.add_argument('--frames', nargs='+', type=int, default=[10,50,100])
    parser.add_argument('--output', type=Path, default=Path('benchmark-results.json'))
    parser.add_argument('--skip-naive', action='store_true')
    args=parser.parse_args(); values=np.linspace(.005,.18,12,dtype=np.float32); rows=[]
    for frames in args.frames:
        pos,vel,teams=synthetic_tracking(frames); cfg=PitchControlConfig()
        optimized,_=timed(lambda:leave_one_out_zone_values(pos,vel,teams,values,cfg))
        naive=None
        if not args.skip_naive:
            naive,_=timed(lambda:naive_leave_one_out_zone_values(pos,vel,teams,values,cfg),1)
        gpu=None
        if is_available():
            gpu,_=timed(lambda:leave_one_out_cuda(pos,vel,teams,values,cfg.grid_x,cfg.grid_y))
        rows.append({'frames':frames,'cpu_optimized_s':optimized,'cpu_naive_s':naive,'cuda_s':gpu,'cuda_speedup':optimized/gpu if gpu else None})
        print(json.dumps(rows[-1]))
    args.output.write_text(json.dumps({'hardware_note':'Times are measurements from the machine that ran this script. Null CUDA means unavailable.','results':rows},indent=2),encoding='utf-8')

if __name__ == '__main__': main()
