#!/usr/bin/env python3
from komunikilo import awgn, ssca as ssca_rs
from ssca import ssca as ssca_py, plot_ssca_triangle, plot_ssca_diamond
from util import timeit

import numpy as np
import multiprocessing
from typing import List

if __name__ == "__main__":
    from psk import tx_bpsk, tx_qpsk
    from cdma import tx_bpsk_cdma
    import random

    SAMPLE_RATE = 48_000
    SYMBOL_RATE = 1000
    CARRIER_FREQ = 2500
    NUM_BITS = 9002

    def rand_data(num_bits: int) -> List[bool]:
        return [random.choice((True, False)) for _ in range(num_bits)]

    N0 = 0.5
    data: List[bool] = rand_data(NUM_BITS)
    bpsk: List[float] = tx_bpsk(data, SAMPLE_RATE, SYMBOL_RATE, CARRIER_FREQ)
    # bpsk_awgn: List[float] = awgn(bpsk, N0)
    bpsk_awgn: List[float] = bpsk

    # cdma_bpsk_awgn = awgn(
    #     tx_bpsk_cdma(data, SAMPLE_RATE, SYMBOL_RATE, CARRIER_FREQ), N0
    # )
    cdma_bpsk_awgn = tx_bpsk_cdma(data, SAMPLE_RATE, SYMBOL_RATE, CARRIER_FREQ)

    qpsk: List[float] = tx_qpsk(data, SAMPLE_RATE, SYMBOL_RATE, CARRIER_FREQ)
    qpsk_awgn: List[float] = awgn(qpsk, N0)
    N = 4096
    # N = 32768 # 4096
    Np = 64

    # for sig in (awgn(np.zeros(len(bpsk)), N0), bpsk, bpsk_awgn, qpsk, qpsk_awgn):
    # for sig in (awgn(np.zeros(len(bpsk)), N0), bpsk, bpsk_awgn):
    signals = (awgn(np.zeros(len(bpsk)), N0), bpsk_awgn, cdma_bpsk_awgn)

    def do_signal(sig: List[float]):
        # Np = 256
        Np = 64
        # N = floor_power_of_2(len(sig) - Np)
        N = 4096
        with timeit("SSCA") as _:
            sx = ssca_rs(sig, N, Np, map_output=True)
        plot_ssca_triangle(sx)

    results = [do_signal(signal) for signal in signals]
    # with Pool(8) as p:
    #     results = p.map(do_signal, signals)

    import matplotlib.pyplot as plt

    plt.show()
