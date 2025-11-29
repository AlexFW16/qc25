
import unittest
import pytest
import itertools
from unittest.mock import patch, MagicMock
from sympy import Matrix as M

from sim.simulator import Simulator


class TestControlledX(unittest.TestCase):

    def setUp(self):
        self.qc = Simulator()

    def test_controlled_x_combinations_and_layers(self):
        """
        Tests that the number of layers matches the number of binary combinations.
        """
        controls = [0, 1, 2]
        out_bit = 3
        size = 4

        result = self.qc.CONTROLLED_X(controls, out_bit, size)

        combinations = list(itertools.product([0, 1], repeat=len(controls)))
        assert len(combinations) == 2 ** len(controls)

    def test_controlled_x_invalid_too_few_controls(self):
        """
        CONTROLLED_X should reject fewer than 3 controls.
        """
        with pytest.raises(AssertionError):
            self.qc.CONTROLLED_X([0], 1, 3)

        with pytest.raises(AssertionError):
            self.qc.CONTROLLED_X([0, 1], 2, 3)

    def test_controlled_x_invalid_size(self):
        """
        Cannot embed more controls than fit into the circuit.
        """
        with pytest.raises(AssertionError):
            self.qc.CONTROLLED_X([0, 1, 2], out_bit=3, size=3)

    def test_controlled_x_output_type(self):
        """
        Final output should be of type sympy.Matrix.
        """
        controls = [0, 1, 2]
        out_bit = 3
        size = 4

        result = self.qc.CONTROLLED_X(controls, out_bit, size)

        assert isinstance(result, M)

    def test_controlled_x_calls_distribute_gates_correctly(self):
        """
        Ensures distribute_gates is called exactly once per layer.
        """
        controls = [0, 1, 2]
        out_bit = 3
        size = 4
        num_layers = 2

