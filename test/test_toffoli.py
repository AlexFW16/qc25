import unittest
from sympy import Matrix as M
from sim.simulator import Simulator  # adjust to your actual module path


class TestToffoli(unittest.TestCase):

    def setUp(self):
        """Create a fresh simulator instance."""
        self.sim = Simulator()

    def ket(self, n, i):
        """Return computational basis |i> as sympy Matrix of size 2^n × 1."""
        v = M.zeros(2**n, 1)
        v[i, 0] = 1
        return v

    def test_toffoli_matrix_size(self):
        """Check that TOFFOLI returns a correct 8x8 matrix for 3 qubits."""
        M_toff = self.sim.TOFFOLI(0, 1, 2, size=3)
        self.assertIsInstance(M_toff, M, "TOFFOLI should return sympy.Matrix")
        self.assertEqual(M_toff.shape, (8, 8), "TOFFOLI matrix must be 8×8 for 3 qubits")

    def test_toffoli_truth_table(self):
        """Check that TOFFOLI flips target qubit only when both controls are 1."""
        M_toff = self.sim.TOFFOLI(0, 1, 2, size=3)

        # |110> → |111>
        out = M_toff * self.ket(3, 6)
        expected = self.ket(3, 7)
        self.assertEqual(out, expected, "|110> did not flip to |111>")

        # |111> → |110>
        out = M_toff * self.ket(3, 7)
        expected = self.ket(3, 6)
        self.assertEqual(out, expected, "|111> did not flip to |110>")

        # |100> unchanged
        out = M_toff * self.ket(3, 4)
        expected = self.ket(3, 4)
        self.assertEqual(out, expected, "|100> changed unexpectedly")

    def test_toffoli_only_flips_on_11(self):
        """Check that all other states remain unchanged."""
        M_toff = self.sim.TOFFOLI(0, 1, 2, size=3)
        for i in range(6):  # |000> to |101>
            ket_i = self.ket(3, i)
            out = M_toff * ket_i
            self.assertEqual(out, ket_i, f"State |{i:03b}> changed unexpectedly")


if __name__ == "__main__":
    unittest.main()
