import unittest
import random
from sympy import Matrix as M, sqrt
from sim.simulator import Simulator  # adjust import
from math import log2

def equal_up_to_global_phase(v1: M, v2: M, tol=1e-10) -> bool:
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


class TestPartialMeasurement(unittest.TestCase):

    def setUp(self):
        self.sim = Simulator(True)

    # -------------------------------------------------------------------------
    # 1) Single-qubit measurement tests
    # -------------------------------------------------------------------------

    def test_measure_single_qubit_zero(self):
        """Measure |0> — must always return 0 and leave state |0>."""
        psi = M([[1], [0]])
        outcome, state = self.sim.partial_measure_and_collapse_single(psi, 0)
        self.assertEqual(outcome, 0)
        self.assertTrue(equal_up_to_global_phase(state, psi))

    def test_measure_single_qubit_one(self):
        """Measure |1> — must always return 1 and leave state |1>."""
        psi = M([[0], [1]])
        outcome, state = self.sim.partial_measure_and_collapse_single(psi, 0)
        self.assertEqual(outcome, 1)
        self.assertTrue(equal_up_to_global_phase(state, psi))

    # -------------------------------------------------------------------------
    # 2) Multi-qubit: check correct collapse behavior
    # -------------------------------------------------------------------------

    def test_measure_first_qubit_of_bell(self):
        """Bell state (|00> + |11>)/√2 → measuring qubit 0 collapses to |00> or |11>."""
        psi = M([
            [1/sqrt(2)],
            [0],
            [0],
            [1/sqrt(2)]
        ])

        outcome, collapsed = self.sim.partial_measure_and_collapse_single(psi, 0)

        if outcome == 0:
            expected = M([[1], [0], [0], [0]])
        else:
            expected = M([[0], [0], [0], [1]])

        self.assertTrue(equal_up_to_global_phase(collapsed, expected),
            msg=f"Expected collapse to |{outcome}{outcome}> but got {collapsed}")

    def test_measure_second_qubit_of_bell(self):
        """Check correct bit-indexing (important!)."""
        psi = M([
            [1/sqrt(2)],
            [0],
            [0],
            [1/sqrt(2)]
        ])

        # Measure qubit index 1
        outcome, collapsed = self.sim.partial_measure_and_collapse_single(psi, 1)

        if outcome == 0:
            expected = M([[1], [0], [0], [0]])
        else:
            expected = M([[0], [0], [0], [1]])

        self.assertTrue(equal_up_to_global_phase(collapsed, expected))

    # -------------------------------------------------------------------------
    # 3) Test multi-qubit partial_measure_and_collapse
    # -------------------------------------------------------------------------

    def test_multiple_index_measurement(self):
        """Measure both qubits of a random 2-qubit state and check consistency."""
        # random normalized 2-qubit state
        vec = [random.random() + random.random()*1j for _ in range(4)]
        norm = sum(abs(x)**2 for x in vec)**0.5
        vec = [x / norm for x in vec]
        psi = M([[vec[i]] for i in range(4)])

        outcomes, collapsed = self.sim.partial_measure_and_collapse(psi, [0, 1])

        # outcomes define a single computational basis state |ab>
        idx = (outcomes[0] << 1) | outcomes[1]
        expected = M([[1 if i == idx else 0] for i in range(4)])

        self.assertTrue(equal_up_to_global_phase(collapsed, expected),
            msg=f"Expected collapse to |{outcomes[0]}{outcomes[1]}>")

    # -------------------------------------------------------------------------
    # 4) Test collapse_to vs. repeated collapse_to_single
    # -------------------------------------------------------------------------

    def test_collapse_to_matches_single(self):
        """collapse_to(indices, outcomes) should match repeated single collapses."""
        psi = M([
            [1/sqrt(2)],
            [0],
            [0],
            [1/sqrt(2)]
        ])  # Bell state

        ind_and_out = [(0, 1), (1, 1)]
        collapsed_multi = self.sim.collapse_to(psi, ind_and_out)

        # Apply collapse individually
        tmp = psi
        ind_and_out_s = sorted(ind_and_out, reverse=True, key = lambda x : x[0])
        for i, o in ind_and_out_s:
            tmp = self.sim.collapse_to_single(tmp, i, o)

        self.assertTrue(equal_up_to_global_phase(tmp, collapsed_multi))

    # -------------------------------------------------------------------------
    # 5) Check normalization
    # -------------------------------------------------------------------------

    def test_normalization_after_measurement(self):
        psi = M([
            [1/sqrt(2)],
            [0],
            [0],
            [1/sqrt(2)]
        ])  # Bell

        _, collapsed = self.sim.partial_measure_and_collapse_single(psi, 0)

        norm = sum(abs(complex(x))**2 for x in collapsed)
        self.assertAlmostEqual(norm, 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
