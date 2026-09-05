from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from benchmark import synthetic_tracking
from zcpv.pitch_control import PitchControlConfig, leave_one_out_zone_values


def main():
    positions, velocities, teams = synthetic_tracking(40)
    zone_values = np.linspace(.006, .19, 12, dtype=np.float32)
    spatial = leave_one_out_zone_values(positions, velocities, teams, zone_values, PitchControlConfig()).sum(axis=(0,2))
    payload={'provenance':{'kind':'deterministic synthetic tracking','frames':40,'seed':7},'zone_values':zone_values.tolist(),'spatial_raw':spatial.tolist()}
    target=Path('public/data/demo-engine-output.json'); target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print(target)

if __name__ == '__main__': main()
