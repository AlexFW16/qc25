from random import Random, random

def draw_short_straw(prob: float) -> int:
    return random() < prob

def draw_straws(n: int) -> tuple[list[int], int]:
    out = []
    loser = -1
    for i in range(n):
        prob = 1/ (n - i)
        out.append(prob)
        if loser == -1 and draw_short_straw(prob):
            loser = i
    return out, loser


def main():
    data: list[tuple[list[int], int]] = []

    for i in range(100):
        data.append(draw_straws(10))
main()
