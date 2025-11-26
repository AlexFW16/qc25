import unittest
from sim.simulator import Simulator, kronecker  # replace with actual module name
from sympy import Matrix as M

class TestDistributeGates(unittest.TestCase):

    def setUp(self):
        self.sim = Simulator()

    def test_single_qubit_gates(self):
        gates = [(0, self.sim.X), (2, self.sim.H)]
        result = self.sim.distribute_gates(gates, 3)
        expected = kronecker([self.sim.X, self.sim.ID, self.sim.H])
        self.assertEqual(result, expected)

    def test_two_qubit_gate(self):
        gates = [([0, 1], self.sim.CNOT())]
        result = self.sim.distribute_gates(gates, 2)
        expected = self.sim.CNOT()
        self.assertEqual(result, expected)

    def test_mixed_gates(self):
        gates = [(0, self.sim.H), ([1, 2], self.sim.CNOT())]
        result = self.sim.distribute_gates(gates, 3)
        expected = kronecker([self.sim.H, self.sim.CNOT()])
        self.assertEqual(result, expected)

    def test_single_qubit_middle(self):
        gates: list[tuple[int, M]] = [(1, self.sim.Z)]
        result = self.sim.distribute_gates(gates, 3)
        expected = kronecker([self.sim.ID, self.sim.Z, self.sim.ID])
        self.assertEqual(result, expected)

    def test_multiple_single_qubit_gates(self):
        gates = [(0, self.sim.X), (1, self.sim.Y), (2, self.sim.Z)]
        result = self.sim.distribute_gates(gates, 3)
        expected = kronecker([self.sim.X, self.sim.Y, self.sim.Z])
        self.assertEqual(result, expected)

if __name__ == "__main__":
    unittest.main()
