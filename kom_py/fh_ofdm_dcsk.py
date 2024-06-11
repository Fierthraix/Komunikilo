#!/usr/bin/env python3
import random
from typing import Any, Generator, Iterable, List, T

RNG_SEED: int = 1


def to_nrz(bit: bool) -> int:
    return 1 if bit else -1


def _len(lst: List[Any]) -> int:
    return sum(_len(i) if isinstance(i, list) else 1 for i in lst)


def _flatten(inp: List[Any]) -> List[Any]:
    return sum(map(_flatten, inp), []) if isinstance(inp, list) else [inp]


def _take(x: int, gen: Generator[T, None, None]) -> Generator[T, None, None]:
    for _ in range(x):
        yield next(gen)


def _chebyshev(xn: float) -> float:
    # return 2 * xn**2 - 1 # First is just the sample, raw.
    return 1 - 2 * xn**2


def chebyshev(x0: float) -> Generator[float, None, None]:
    assert -1 <= x0 <= 1
    xn: float = x0
    while True:
        xn = _chebyshev(xn)
        yield xn


def tx_fh_dcsk(data: Iterable[bool], b: int = 8) -> Iterable[float]:
    # Make a `b * b` matrix of input data.
    dat_array: List[List[bool]] = [
        [to_nrz(datum) for _ in range(b)] for datum in _take(b, iter(data))
    ]

    x0s: List[float] = [0.001 + (i / 10) for i in range(b)]

    # Multiply each bit (row) by different chaos sequences.
    # for x0, row in zip(x0s, dat_array):
    for row, x0 in enumerate(x0s):
        xn = x0
        for col in range(b):
            dat_array[row][col] *= xn  # First is just the sample, raw.
            xn = _chebyshev(xn)

    rng = random.Random(RNG_SEED)

    # Re-arrange each column.
    for col in range(b):
        # Shuffle the column.
        new = [dat_array[i][col] for i in range(b)]
        rng.shuffle(new)

        # Assign the column.
        for row, new_dat in enumerate(new):
            dat_array[row][col] = new_dat

    return dat_array


def rx_fh_dcsk(data: Iterable[bool], b: int = 8) -> Iterable[float]:
    # rng = random.Random(RNG_SEED)
    ...


if __name__ == "__main__":
    b = 8
    data: List[bool] = [True if i % 2 == 0 else False for i in range(b)]
    assert _len(data) == b

    result = tx_fh_dcsk(data)
    assert _len(data) == _len(result) / b
    # assert _flatten(data) == _flatten(result)
    print(result)
