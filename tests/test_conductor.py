import numpy as np
import pytest
from shapely.geometry import box, mapping

from mom25d.base import MU0
from mom25d.conductor import surface_impedance
from mom25d.mqs import MQS


@pytest.mark.parametrize('faces', [1, 2])
def test_slab_limits_and_passivity(faces):
    sigma, t = 5.8e7, 35e-6
    f = np.geomspace(1e-3, 1e10, 41)
    z = surface_impedance(f, t, sigma, faces)
    assert surface_impedance(0, t, sigma, faces) == pytest.approx(1/(sigma*t))
    assert z[0].real == pytest.approx(1/(sigma*t), rel=1e-12)
    assert z[0].imag/(2*np.pi*f[0]) == pytest.approx(MU0*t/(3*faces**2), rel=1e-10)
    delta = np.sqrt(1/(np.pi*f[-1]*MU0*sigma))
    assert z[-1] == pytest.approx((1+1j)/(faces*sigma*delta), rel=1e-8)
    assert (z.real > 0).all() and (z.imag >= 0).all()
    assert np.diff(z.real).min() >= -1e-15


def test_sheet_loss_is_used_by_mqs_and_peec():
    board = dict(stack=dict(layers=[], copper=[dict(name='F', z=0, t=.035)]),
                 copper={'F': [dict(net='N', polygon=mapping(box(0, 0, 5, 2)))]}, vias=[])
    terms = {'a': [('F', box(0, 0, 1, 2))], 'b': [('F', box(4, 0, 5, 2))]}
    answers = []
    for cap, loss in [(False, 'dc'), (False, 'slab'), (True, 'slab')]:
        q = MQS(board, 'N', terminals=terms, dmax=.7, capacitance=cap, conductor_loss=loss,
                conductor_faces=1, log=lambda *_: None)
        q.source('I', 'a', 'b', part_h=0)
        answers.append(q.solve([1e3, 30e6]).z['I'])
    assert answers[1][1].real > 1.5*answers[0][1].real
    assert answers[2] == pytest.approx(answers[1], rel=.001)
