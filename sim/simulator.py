import itertools
import sympy as sp
from sympy import N, Matrix as M, sstr
from sympy import cos, exp, pprint, I, sin, sqrt 
from sympy import pi as PI
import numpy as np
from typing import cast
from math import log2
import random

from .my_math import kronecker, kronecker2

sqrt2 = sp.sqrt(2)

class Simulator:
    
    def __init__(self, numerically: int = False) -> None:
        self.NUMERICALLY = numerically


    # Basis 1-qbuit states
    ZERO:M  = M([1, 0])  # Zero-State
    ONE: M = M([0, 1])  # One-State
    PLUS: M = M([1, 1]) / sqrt2  # Superposition Plus
    MINUS: M = M([1, -1]) / sqrt2  # Superposition Minus
    IPLUS: M = M([1, I]) / sqrt2  # Superposition Plus i 
    IMINUS: M = M([1, -I]) / sqrt2 # Superposition Minus i 

    # Basis 2-qbuit states
    BELL00: M = M(1/sqrt2 * (kronecker2(ZERO, ZERO) + kronecker2(ONE, ONE))) # 00 Bell State
    BELL01: M = M(1/sqrt2 * (kronecker2(ZERO, ONE) + kronecker2(ONE, ZERO))) # 01 Bell State
    BELL10: M= M(1/sqrt2 * (kronecker2(ZERO, ZERO) - kronecker2(ONE, ONE))) # 10 Bell State
    BELL11: M = M(1/sqrt2 * (kronecker2(ZERO, ONE) - kronecker2(ONE, ZERO))) # 11 Bell State


    # Basic 1-qubit Gates
    ID = sp.eye(2)  # Identity
    Z = M([[1, 0], [0, -1]])  # signflip
    X = M([[0, 1], [1, 0]])  # NOT / bitflip
    H = M([[1, 1], [1, -1]]) / sqrt2  # Hadamard
    S  = M([[1, 0], [0, I]]) # sqrt of Z
    T  = M([[1, 0], [0, exp( I * PI / 4)]]) # sqrt of S
    Y = M([[0, - I], [I, 0]])

    # Basic 2-qubit Gates 
    #NOTE: moved to methods below
    CNOT_VAR = M([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]) # first bit control
    CNOT2_VAR = M([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
    
    # TOFFOLI = M([])
    BELLM = kronecker2(H, ID) * CNOT_VAR

    # T-State for teleportation
    T_STATE0: M  = T * H * ZERO
    T_STATE1: M  = T * H * ONE

    # Basic projector matrices
    P0: M = ZERO * ZERO.H # |0> <0| (= projector)
    P1: M  = ONE * ONE.H # |1> <1| (= projector)
    P00 = kronecker2(P0, P0)
    P01 = kronecker2(P0, P1)
    P10 = kronecker2(P1, P0)
    P11 = kronecker2(P1, P1)
 

    # X-Axis Pauli Rotation
    def R_X(self, theta: sp.Expr | sp.Rational | float) -> M:
        return M([[cos(theta/2), - I * sin(theta/2)], [- I * sin(theta/2), cos(theta/2)]])

    # Y-Axis Pauli Rotation
    def R_Y(self, theta: sp.Expr | sp.Rational | float) -> M:
        return M([[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]])

    # Z-Axis Pauli Rotation
    def R_Z(self, theta: sp.Expr | sp.Rational | float) -> M:
        return M([[exp(-I * theta / 2), 0], [0, exp(I * theta/2)]])

    def CNOT2(self):
        return M([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
    def CNOT(self, q1: int = -1, q2: int = -1, n: int = -1) -> M:
        """
        Takes as input indices of control bit ``q1`` and negated bit ``q2`` as well
        as the number of qubits in the circuit and connects them via CNOT.
        For emtpy input, it returns standard CNOT
        """
        if q1 == -1 and q2 == -1 and n == -1:
            return M([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]) # first bit control

        assert (q1 != -1 and q2 != -1 and n != -1), "Either all arguments for CNOT() must be empty, or none."

        l1: list[M] = [self.ID] * n
        l2: list[M] = [self.ID] * n
        l1[q1] = self.P0
        l2[q1] = self.P1
        l2[q2] = self.X
        return kronecker(l1) + kronecker(l2)


    def TOFFOLI(self, q0: int = -1, q1: int = -1, q2:int = -1, size: int = -1) -> M:
        """
        Implements the TOFFOLI gate like ``CNOT()`` is implemented. ``q0, q1`` are control bits
        and ``q3`` is affected bit.
        """

        if q0 == -1 and q1 == -1 and q2 == -1 and size == -1:
            return M([]) #TODO:
        assert q0 != -1 and q1 != -1 and q2 != -1 and size != -1,  "Either all arguments for TOFFOLI() must be empty, or none."
        
        l1: list[M] = [self.ID] * size
        l2: list[M] = [self.ID ] * size

        layers = [
            [(q0, self.P0), (q1, self.P0),  (q2, self.ID)],
            [(q0, self.P0), (q1, self.P1),  (q2, self.ID)],
            [(q0, self.P1), (q1, self.P0),  (q2, self.ID)],
            [(q0, self.P1), (q1, self.P1),  (q2, self.X)],
        ]

        out = sum([self.distribute_gates(layer, size) for layer in layers], M.zeros(2**size, 2**size))

        assert isinstance(out, M), "TOFFOLI gate construction returned zero, no layers?"
        return out

    def CONTROLLED_X(self, controls: list[int], out_bit: int, size: int) -> M:
        """
        Returns a controlled bitflip gated (generalisation of CNOT/TOFFOLI).
        Needs a list of control bits, an output bit that is flipped and
        the size of the circuit it is embedded into.

        """
        assert len(controls) > 2, "If using less than 2 control bits, use TOFFOLI/CNOT"
        assert len(controls) + 1 <= size, f"Cannot have {len(controls)} control bits in circuit with {size} qubits"

        layers = [[] for _ in range(2**len(controls))]
        combinations = list(itertools.product([0, 1], repeat=len(controls)))

        assert len(combinations) == len(layers), f"There must be the same amount of layers [{len(layers)}] as combinations of 0-1 [{len(combinations)}]"
        for layer, comb in zip(layers, combinations):
            for c_bit, proj in zip(controls, comb):
                if proj == 0:
                    layer.append((c_bit, self.P0))
                else:
                    layer.append((c_bit, self.P1))

            layer.append((out_bit, self.X))
        out = sum([self.distribute_gates(layer, size) for layer in layers], M.zeros(2**size, 2**size))
        assert isinstance(out, M), "CONTROLLED_X gate construction returned zero, no layers?"
        return out

    def XOR(self, input_bits: list[int], out_bit: int, size: int) -> M:
        out = kronecker([self.ID] * size)
        for i in range(size-1):
            out *= self.CNOT(i, i+1, size)
        return out



    def all_bitstrings(self, n:int):
        """
        Returns a list of labels for all binary numbers up to n.
        """
        return [format(i, f'0{n}b') for i in range(2 ** n)]


    #NOTE: just use Hermite conjugate directly
    # def bra_to_ket(A: M) -> M:
    #     return A.H
        
    def is_normalised(self, v: M) -> bool:
        norm = v.norm()
        if self.NUMERICALLY:
            norm = float(norm.evalf())
            return bool(np.isclose(norm, 1.0))
        else:
            return sp.simplify(norm) == 1

    def is_normalised_prob(self, v: M) -> float:
        """
        Returns a probablity of being normalised, to manually disregard small
        imperfections when using concrete numbers.
        """
        return float(1 - v.norm())


    def eval(self, expr: sp.Expr) -> float:
        """
        Evaluates a sympy expression to a float.
        """
        return float(sp.Float(expr.evalf()))

    def get_probs(self, state: M, as_cdf: bool = False) -> list[sp.Expr]:
        assert self.is_normalised(state), f"[Warn] {self.is_normalised_prob(state)}"
        
        out: list[sp.Expr] = []
        for i in range(state.rows):
            entry = state[i, 0]
            assert isinstance(entry, sp.Expr), f"Excpected expression and not {type(entry)}"

            if as_cdf:
                prev = out[i-1] if i > 0 else 0
                out.append(prev + sp.Pow(abs(entry) , 2))
            else:
                out.append(sp.Pow(abs(entry), 2))
        return out

    def get_prob_single(self, state: M, value: int) -> sp.Expr:
        if not self.is_normalised(state):
            print(f"[Warn] {self.is_normalised_prob(state)}")
        if value not in (0, 1):
            raise ValueError("Value must be 0 or 1")
        entry = sp.sympify(state[value, 0])
        # entry = sp.sympify(state[0, 0]) if value == 0 else sp.sympify(state[0, 1])

        if not isinstance(entry, sp.Expr):
            raise TypeError(f"Excpected expression and not {type(entry)}")
        return sp.Pow(abs(entry), 2)


    def compose_gates(self, gates: list[M]):
        """
        The gates are applied as if you write out the list from left [0] to right [-1]
        """
        result = sp.eye(2)
        for i in range(len(gates)):
            result = gates[i] * result
        return result


    def distribute_gates(self, gates: list[tuple[list[int] | int, M]], n: int) -> M:
        """
        Given as input a list of tuples ``(location, gate)`` that specify on which qubit
        the given gate should act. If the location has more than one entry, it is assumed
        that a gate like CNOT or CCNOT is given. MUST START WITH THE SMALLEST INDEX TO START THE GATE.
        ``n`` tells the function how many qubits the circuit has.
        """
        extended_gates: list[M] = [self.ID for _ in range(n)] # gates list that will be fed into kron prod
        to_remove: list[int] = [] # stores indices where to remove elements if CNOT etc. are used
        for positions, gate in gates:
            if isinstance(positions, int):
                extended_gates[positions] = gate
            else:
                assert len(positions) > 0, "Cannot insert a gate at position < 0 for distribution."
                # single qubit gate
                if len(positions) == 1:
                    extended_gates[positions[0]] = gate
                else:
                    min_pos = min(positions) # find minimal position
                    extended_gates[min_pos] = gate # add the gate

                    # Removes the used spot and schedules other indices for removal in the end
                    positions.remove(min_pos)
                    to_remove += positions
                    
        for i in to_remove:
            assert i < len(extended_gates), "Cannot remove element of index larger than size of list"
        adjusted_extended_gates = [g for i, g in enumerate(extended_gates) if i not in to_remove]
        return kronecker(adjusted_extended_gates)



    def measure(self, state: M) -> int:
        size = state.shape[0]
        assert size % 2 == 0, "Invalid state given: Length not a power of 2"

        cdf = self.get_probs(state, as_cdf=True)
        if self.NUMERICALLY:
            cdf = [float(N(x)) for x in cdf]
            assert abs(cdf[-1] -1) < 1e-12, f"Sum over CDF is not 1: {cdf[-1]}"
        else:
            assert cdf[-1] == 1, f"Sum over CDF is not 1: {cdf[-1]}"

        r = random.random()
        for i in range(size):
            if r <= cdf[i]:
                return i
        return -1

    def partial_measure_prob(self, state: M, index: int, outcome: int) -> sp.Expr:

        """
        Compute the probability of obtaining a given measurement ``outcome`` (0 or 1)
        on a single qubit at the specified ``index`` of a quantum state.
        """


        shape = state.shape
        assert outcome in {0, 1}, f"Asking for partial measurement outcome of {outcome} not in {{0, 1}}"
        assert shape[0] == 1 or shape[1] == 1, f"Given state {state} is not a vector"

        vec_len =  shape[0] if shape[1] == 1 else shape[1]
        n = int(log2(vec_len))

        assert index < n, f"index {index} for partial measure oob"
        assert self.is_normalised(state), f"Given state is not normalised\n{sstr(state)}"

        #NOTE: adjust ordering of LSB/MSB to be coherent with kronecker product
        lsb_index = n - 1- index
        # shift by index s.t the interesting bit is now the LSB, then bitwise and with 1 to extract it
        indices = [i for i in range(vec_len) if ((i >> lsb_index) & 1) == outcome]

        # Sum probabilities
        if shape[0] == 1:
            return cast(sp.Expr, sum(abs(cast(sp.Expr, state[0, i]))**2 for i in indices))
        else:
            return cast(sp.Expr, sum(abs(cast(sp.Expr, state[i, 0]))**2 for i in indices))

    def partial_measure_and_collapse(self, state: M, indices: list[int]) -> tuple[list[int], M]:
        outcomes = []
        assert len(indices) > 0, "No indices given for which to partial measure and collapse"

        new_state = state
        for i in sorted(indices, reverse=True):
            outcome, new_state = self.partial_measure_and_collapse_single(new_state, i)
            outcomes.append(outcome)
        return outcomes, new_state
        

    def partial_measure_and_collapse_single(self, state: M, index: int) -> tuple[int, M]:
        """
        Perform a partial measurement on a single qubit at the given index,
        collapse the state vector accordingly, and return the new normalized state.
        """
        prob_0 = self.partial_measure_prob(state, index, 0)
        outcome = 0 if random.random() < prob_0.evalf() else 1
        new_state = self.collapse_to_single(state, index, outcome)
        return (outcome, new_state)

    def collapse_to(self, state: M, indices_and_outcomes: list[tuple[int, int]]):

        assert len(indices_and_outcomes) > 0, "No indices/outcomes given for which to collapse"

        new_state = state
        indices_and_outcomes_sorted = sorted(indices_and_outcomes, reverse=True, key=lambda x: x[0])

        for i, o in indices_and_outcomes_sorted:
            new_state = self.collapse_to_single(new_state, i, o)
        assert new_state is not None, "State collapsed cannot be None"
        return new_state


    def collapse_to_single(self, state: M, index: int, outcome: int) -> M:
        """
        Collapses a state into a given outcome, disregarding probabilities.
        """
        shape = state.shape
        assert outcome in {0, 1}, f"Asking for collapsing with outcome {outcome} not in {{0, 1}}"
        assert shape[0] == 1 or shape[1] == 1, f"Given state {state} is not a vector"

        vec_len =  shape[0] if shape[1] == 1 else shape[1]
        n = int(log2(vec_len))
        lsb_index = n -1 - index

        assert index < n, f"Cannot collapse for state at index {index} >= {n}-quibts"

        if state.shape[0] == 1:
            vec_len = state.shape[1]
            new_state = M([[state[0, i] for i in range(vec_len) if ((i >> lsb_index) & 1) == outcome]])
        elif state.shape[1] == 1:
            vec_len = state.shape[0]
            new_state = M([[state[i, 0]] for i in range(vec_len) if ((i >> lsb_index) & 1) == outcome])
        else:
            raise ValueError(f"Given state {state} is not a vector")

        normalisation_correction = sqrt(self.partial_measure_prob(state, index, outcome))
        # if prob of events of which we are marginalising is 0, no need to change normalisation
        normalisation_correction = sp.S.One if normalisation_correction == 0 else normalisation_correction
        new_state = new_state / normalisation_correction 

        return new_state

       


    def t_gate_teleport(self, state: M) -> M:
        state2 = self.CNOT * self.CNOT2 * kronecker2(self.T_STATE0, state)
        outcome, collapsed_state =  self.partial_measure_and_collapse_single(state2, 0)
        return self.S * self.X * collapsed_state if outcome == 1 else collapsed_state

    def teleport(self, state: M) -> M:
        circuit = kronecker([self.H, self.ID, self.ID]) * kronecker([self.CNOT(), self.ID]) * kronecker([self.ID, self.CNOT()]) * kronecker([self.ID, self.H, self.ID])


        state = circuit * kronecker([state, self.ZERO, self.ZERO])

        use_z, state = self.partial_measure_and_collapse_single(state, 0)
        use_x, state = self.partial_measure_and_collapse_single(state, 0)


        corr = self.Z if use_z else self.ID
        corr *= self.X if use_x else self.ID

        # return kronecker([self.ID, self.ID, corr]) * state

        return self.Z ** use_z * self.X ** use_x * state
        # return self.Z ** use_z * self.X ** use_x * state



