import numpy as np
from zcpv.zones import ZoneGrid, fit_zone_values


def test_zone_grid_boundaries():
    grid=ZoneGrid()
    assert grid.index(0,0)==0
    assert grid.index(104.9,67.9)==11
    assert grid.index(52.5,34)==7


def test_zone_value_chain_propagates_goal_probability():
    transitions=np.zeros((12,12)); transitions[0,1]=80; transitions[1,2]=50
    shots=np.zeros(12); shots[2]=20; goals=np.zeros(12); goals[2]=5
    values=fit_zone_values(transitions,shots,goals)
    assert np.allclose(values[:3],[.25,.25,.25])
