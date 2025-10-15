# import numpy as np
import random
import sympy as sp
from sympy import pprint

sqrt2 = sp.sqrt(2)

# Basis States
ZERO = sp.Matrix([1, 0])  # Zero-State
ONE = sp.Matrix([0, 1])  # One-State
SP = sp.Matrix([1, 1]) / sqrt2  # Superposition Plus
SM = sp.Matrix([1, -1]) / sqrt2  # Superposition Minus


# Basic Quantum Gates

I = sp.eye(2)  # Identity
Z = sp.Matrix([[1, 0], [0, -1]])  # signflip
X = sp.Matrix([[0, 1], [1, 0]])  # NOT / bitflip
H = sp.Matrix([[1, 1], [1, -1]]) / sqrt2  # Hadamard


def is_normalised(v: sp.Matrix):
    return v.norm() == 1


def eval(expr: sp.Expr) -> float:
    """
    Evaluates a sympy expression to a float.
    """
    return float(sp.Float(expr.evalf()))


def get_prob(state: sp.Matrix, value: int) -> sp.Expr:
    assert is_normalised(state), "State is not normalised"
    if value not in (0, 1):
        raise ValueError("Value must be 0 or 1")

    return sp.Pow(state[0], 2) if value == 0 else sp.Pow(state[1], 2)


def compose_gates(gates: list[sp.Matrix]):
    """
    The gates are applied as if you write out the list from left [0] to right [-1]
    """
    result = sp.eye(2)
    for i in range(len(gates)):
        result = gates[i] * result
    return result


def measure(state: sp.Matrix):
    assert is_normalised(state), "State is not normalised"

    p0 = get_prob(state, 0)
    return 0 if random.random() < eval(p0) else 1


def main():
    G = compose_gates([H, H])
    # pprint(measure(G * ZERO))
    print(measure(H * ZERO))


if __name__ == "__main__":
    main()
