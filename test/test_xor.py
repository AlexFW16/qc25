import unittest
from unittest import mock
from sympy import Matrix as M
from sim.simulator import Simulator


class TestXOR(unittest.TestCase):

    def setUp(self):
        self.qc = Simulator()

    def test_xor_output_type_and_size(self):
        """XOR should return a sympy Matrix of correct dimension."""
        size = 4
        out_bit = 3
        input_bits = [0, 1, 2]

        result = self.qc.XOR(input_bits, out_bit, size)

        self.assertIsInstance(result, M)
        self.assertEqual(result.shape, (2**size, 2**size))

    def test_xor_identity_for_size_1(self):
        """XOR with size 1 should return identity."""
        size = 1
        result = self.qc.XOR([], 0, size)
        expected = self.qc.ID  # assuming kronecker([ID]) returns self.ID
        self.assertTrue(result == expected)

    def test_xor_composition_of_cnot(self):
        """Check that XOR is composed of CNOTs as expected."""
        size = 3
        out_bit = 2
        input_bits = [0, 1]

        with mock.patch.object(self.qc, "CNOT", wraps=self.qc.CNOT) as mock_cnot:
            self.qc.XOR(input_bits, out_bit, size)
            # size-1 CNOTs are called
            self.assertEqual(mock_cnot.call_count, size-1)
            for i in range(size-1):
                # Check each CNOT call has correct control-target-size args
                args, _ = mock_cnot.call_args_list[i]
                self.assertEqual(args[0], i)
                self.assertEqual(args[1], i+1)
                self.assertEqual(args[2], size)

    def test_xor_consistency_with_manual_multiplication(self):
        """Check that XOR produces same matrix as manual CNOT multiplication."""
        size = 3
        out_bit = 2
        input_bits = [0, 1]

        manual = self.qc.CNOT(0,1,size) * self.qc.CNOT(1,2,size)
        result = self.qc.XOR(input_bits, out_bit, size)

        self.assertTrue(result.equals(manual), "XOR matrix does not match manual CNOT multiplication")
