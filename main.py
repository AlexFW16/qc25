# import numpy as np
import sympy as sp

sqrt2 = sp.sqrt(2)

ZERO = sp.Matrix([1, 0])  # Zero-State
ONE = sp.Matrix([0, 1])  # One-State
SP = sp.Matrix([1, 1]) / sqrt2  # Superposition Plus
SM = sp.Matrix([1, -1]) / sqrt2  # Superposition Minus

# Hadamard
H = sp.Matrix([[1, 1], [1, -1]]) / sqrt2


def is_normalised(v: sp.Matrix):
    return v.norm() == 1


def get_prob(value: int, state: sp.Matrix):
    return sp.Pow(state[0], 2) if value == 1 else sp.Pow(state[1], 2)


def main():
    print(get_prob(0, ZERO))
    print(get_prob(1, ZERO))
    print(get_prob(0, ONE))
    print(get_prob(1, ONE))
    print(get_prob(0, SP))
    print(get_prob(1, SP))
    print(get_prob(0, SM))
    print(get_prob(1, SM))


if __name__ == "__main__":
    main()
