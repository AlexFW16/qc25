import unittest
import random
from sympy import Matrix as M
from sim.simulator import Simulator  # replace with actual import path

def equal_up_to_global_phase(v1: M, v2: M, tol=1e-10) -> bool:
    """
    Returns True if v1 and v2 differ only by a global phase.
    Both must be column vectors.
    """
    c1 = [complex(x.evalf()) for x in v1.iter_values()]
    c2 = [complex(x.evalf()) for x in v2.iter_values()]

    ratios = []
    for a, b in zip(c1, c2):
        if abs(a) < tol and abs(b) < tol:
            continue
        if abs(a) < tol or abs(b) < tol:
            return False
        ratios.append(a / b)

    if not ratios:
        return True

    first = ratios[0]
    return all(abs(r - first) < tol for r in ratios)

class TestTeleportation(unittest.TestCase):

    def setUp(self):
        self.sim = Simulator(True)  # initialize simulator

    def test_teleportation_random_trials(self):
        trials = 40
        for k in range(trials):
            # Random normalized 1-qubit state
            a = random.random() + random.random() * 1j
            b = random.random() + random.random() * 1j
            norm = (abs(a)**2 + abs(b)**2)**0.5
            a /= norm
            b /= norm

            psi = M([[a], [b]])

            # Teleport using your simulator
            out = self.sim.teleport(psi)

            # Reduce final state to Bob’s qubit (assumes Bob is qubit 0 after teleport)
            if out.rows == 2:
                bob = out
            else:
                bob = M([[out[0, 0]], [out[1, 0]]])

            # Check equality up to global phase
            self.assertTrue(
                equal_up_to_global_phase(psi, bob),
                msg=f"Teleportation FAILED on trial {k+1}\nInput: {psi}\nBob got: {bob}"
            )

if __name__ == "__main__":
    unittest.main()
