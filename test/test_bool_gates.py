import unittest
from sympy import Matrix as M, pprint
from sim.bool_logic import AND_circ, OR_circ
from sim.simulator import Simulator, kronecker

class TestLogicCircuits(unittest.TestCase):
    def setUp(self):
        self.sim = Simulator(numerically=True)
        self.ZERO = self.sim.ZERO
        self.ONE = self.sim.ONE

    def test_and_truth_table(self):
        """Test AND_circ for all possible 2-bit inputs."""
        n = 3
        for a in [0, 1]:
            for b in [0, 1]:
                input_state = kronecker([
                    self.ONE if a else self.ZERO,
                    self.ONE if b else self.ZERO,
                    self.ZERO  # output qubit initialized to 0
                ])
                circ = AND_circ(n)
                output, _ = self.sim.partial_measure_and_collapse_single(circ * input_state, 2)
                expected = a & b
                self.assertEqual(output, expected, f"AND({a},{b}) failed")

    def test_or_truth_table(self):
        """Test OR_circ for all possible 2-bit inputs."""
        n = 3
        for a in [0, 1]:
            for b in [0, 1]:
                input_state = kronecker([
                    self.ONE if a else self.ZERO,
                    self.ONE if b else self.ZERO,
                    self.ZERO  # output qubit initialized to 0
                ])
                circ = OR_circ(n)
                output, _ = self.sim.partial_measure_and_collapse_single(circ * input_state, 2)
                expected = a | b
                self.assertEqual(output, expected, f"OR({a},{b}) failed")

if __name__ == "__main__":
    unittest.main()
