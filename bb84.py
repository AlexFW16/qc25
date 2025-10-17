from random import random, seed, randint
import sympy as sp

from main import H, ONE, ZERO, measure


class Person:

    def __init__(self) -> None:
        self.bits: list[int] = []  # stores the bits randomly created or received
        self.ops: list[int] = []  # stores the type of operation (0 = I, 1 = H)

    def encode(self) -> sp.Matrix:
        """
        Encodes and sends the bit and returns it (sends it)
        """
        # random 0/1 bit
        bit = randint(0, 1)
        self.bits.append(bit)
        bit = ONE if bit else ZERO

        # apply either H or I
        rand = randint(0, 1)
        self.ops.append(rand)
        # return = send
        return H * bit if rand else bit

    def decode(self, state: sp.Matrix):
        # receive quantum state
        rand = randint(0, 1)
        self.ops.append(rand)

        # apply I or H randomly
        # measure outcome and store
        result = measure(H * state) if rand else measure(state)
        self.bits.append(result)


def bb84(length: int, p1: Person, p2: Person) -> float:

    n = 20
    seed(10)
    for i in range(n):
        p2.decode(p1.encode())

    check_list: list[tuple[int, int]] = []
    for i, (o1, o2) in enumerate(zip(p1.ops, p2.ops)):
        if o1 == o2:
            check_list.append((p1.bits[i], p2.bits[i]))

    sum = 0
    for entry in check_list:
        sum += 1 if entry[0] == entry[1] else 0
    return sum / len(check_list)

    alice = Person()
    bob = Person()


alice, bob = Person(), Person()

print("alice")
print("bits:", alice.bits)
print("ops  ", alice.ops)

print("bob")
print("bits:", bob.bits)
print("ops  ", bob.ops)

print(bb84(100, alice, bob))
