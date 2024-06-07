#!/usr/bin/env python3
from contextlib import contextmanager
from itertools import zip_longest
from math import ceil, log
import numpy as np
import time
from typing import Any, Iterable, Generator, List, Union, Optional


def db(n: Union[float, np.array]) -> Union[float, np.array]:
    return 10 * np.log10(n)


def undb(n: Union[float, np.array]) -> Union[float, np.array]:
    return np.float_power(10, n / 10)


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** int(x - 1).bit_length()


def floor_power_of_2(x):
    x = int(x)
    if x == 0:
        return 0
    elif log(x, 2).is_integer():
        return x
    else:
        return 2 ** (int(x - 1).bit_length() - 1)


def bit_to_nrz(bit: bool) -> float:
    return 1 if bit else -1


def bitz_to_nrz(bits: Iterable[bool]) -> Generator[float, None, None]:
    return map(bit_to_nrz, bits)


def chunk(iterable: Iterable[Any], n: int, fillvalue: Optional[Any] = None):
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def fftshift(x: List[Any]) -> List[Any]:
    pivot = ceil(len(x) / 2)
    f = []
    f.extend(x[pivot:])
    f.extend(x[:pivot])
    return np.array(f)


def secs_to_str(seconds: float) -> str:
    # seconds = int(seconds)
    s_m = 60
    s_h = 60 * s_m
    s_d = 24 * s_h
    s_y = 365.25 * s_d
    years = seconds // s_y
    days = seconds // s_d
    hours = seconds // s_h
    minutes = seconds // s_m
    out = ""
    if years:
        out += f"{int(seconds // s_y)}Y:"
    if days:
        out += f"{int(seconds % s_y // s_d)}d:"
    if hours:
        out += f"{int(seconds % s_d // s_h)}h:"
    if minutes:
        out += f"{int(seconds % s_h // s_m)}m:"
    if seconds:
        out += f"{int(seconds % s_m)}s"
    if not out:
        return "Эспэранто"
    return out


@contextmanager
def timeit(label: Optional[str] = None):
    start = time.monotonic()
    try:
        yield
    finally:
        total_time = time.monotonic() - start
        if label:
            print(f"{label}: {secs_to_str(total_time)}.")
        else:
            print(f"{secs_to_str(total_time)}.")
