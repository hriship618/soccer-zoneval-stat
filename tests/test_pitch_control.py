import numpy as np
from zcpv.pitch_control import PitchControlConfig, leave_one_out_zone_values, naive_leave_one_out_zone_values


def test_optimized_matches_naive():
    rng=np.random.default_rng(3); pos=rng.uniform((0,0),(105,68),(2,22,2)).astype('f4'); vel=rng.normal(0,1,(2,22,2)).astype('f4')
    teams=np.r_[np.zeros(11,dtype='i1'),np.ones(11,dtype='i1')]; values=np.linspace(.01,.2,12,dtype='f4'); cfg=PitchControlConfig(grid_x=12,grid_y=9)
    fast=leave_one_out_zone_values(pos,vel,teams,values,cfg); slow=naive_leave_one_out_zone_values(pos,vel,teams,values,cfg)
    np.testing.assert_allclose(fast,slow,rtol=2e-5,atol=2e-7)
    assert np.all(fast>=0)
