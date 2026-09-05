import numpy as np
import pytest
from zcpv.actions import Action, value_action


def test_pass_and_failed_pass_values():
    values=np.linspace(0,.22,12)
    assert value_action(Action('p','pass',2,8),values)==pytest.approx(values[8]-values[2])
    assert value_action(Action('p','pass',8,10,False),values)==pytest.approx(-values[8])


def test_same_zone_dribble_uses_retention_risk():
    values=np.linspace(.01,.23,12)
    action=Action('p','take_on',10,10,True,loss_risk_before=.55,loss_risk_after=.2)
    assert value_action(action,values)==pytest.approx(values[10]*.35)
