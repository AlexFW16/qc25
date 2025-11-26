from sympy import Matrix as M
from sympy import pprint
from .simulator import Simulator
from .my_math import kronecker


def AND_circ(size: int, q0: int = 0, q1: int = 1, q2: int = 2)-> M:
    """
    Uses a Toffoli gate to simulate a lgocial AND.
    Returns ``q0 & q1 = q2``
    ``q2`` must always be initialised with 0
    """
    s = Simulator()
    return  s.TOFFOLI(q0, q1,q2, size)

#TODO:
def OR_circ(size: int, q0: int = 0, q1: int = 1, q2: int = 2)-> M:
    """
    Uses a Toffoli gate to simulate a lgocial OR.
    Returns ``q0 | q1 = q2``
    ``q2`` must always be initialised with 0
    """

    s = Simulator()
    return s.distribute_gates([(q0, s.X), (q1, s.X)], size) * \
              s.TOFFOLI(q0, q1,q2, size) * \
              s.distribute_gates([(q0, s.X), (q1, s.X), (q2, s.X)], size)

