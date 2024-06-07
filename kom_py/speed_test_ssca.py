#!/usr/bin/env python3
from komunikilo import ssca, ssca_mapped
from ssca import max_cut, ssca as ssca_np
from util import timeit

import functools
import numpy as np
import multiprocessing
from typing import List

if __name__ == "__main__":

    # Unmapped SSCA vs SSCA_PY
    N = 4096
    Np = 64

    num_iters = 150

    datas: List[np.array] = [np.random.randn(N + Np) for _ in range(num_iters)]

    def ssca_rs(data):
        return ssca(data, N, Np)
    with multiprocessing.Pool(8) as p, timeit("SSCA (Unmapped) - Rust") as _:
        results = p.map(ssca_rs, datas)
        assert len(results) == num_iters

    def ssca_py(data):
        return ssca_np(data, N, Np, map_output=False)
    with multiprocessing.Pool(8) as p, timeit("SSCA (Unmapped) - Python") as _:
        results = p.map(ssca_py, datas)
        assert len(results) == num_iters

    # Mapped SSCA vs SSCA_PY
    def ssca_rs(data):
        return ssca_mapped(data, N, Np)
    with multiprocessing.Pool(8) as p, timeit("SSCA (Mapped) - Rust") as _:
        results = p.map(ssca_rs, datas)
        assert len(results) == num_iters

    def ssca_py(data):
        return ssca_np(data, N, Np, map_output=True)
    with multiprocessing.Pool(8) as p, timeit("SSCA (Mapped) - Python") as _:
        results = p.map(ssca_py, datas)
        assert len(results) == num_iters
