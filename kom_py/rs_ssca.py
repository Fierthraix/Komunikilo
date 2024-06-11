#!/usr/bin/env python3
from willie import awgn
from psk import tx_bpsk, tx_qpsk
from cdma import tx_bpsk_cdma
from util import timeit
from ssca import (
    ssca as ssca_py,
    # plot_ssca_diamond,
    plot_ssca_triangle,
    max_cut,
    dcs,
)
from komunikilo import ssca_mapped

import random
from typing import List


if __name__ == "__main__":

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

    # signals = (awgn(np.zeros(len(bpsk)), N0), bpsk_awgn, cdma_bpsk_awgn)
    # signals = (awgn(np.zeros(len(bpsk)), N0),)
    signals = (bpsk_awgn,)
    with timeit("Rust") as _:
        for sig in signals:
            # Np = 256
            Np = 64
            # N = floor_power_of_2(len(sig) - Np)
            N = 4096
            # sx = ssca(sig, N, Np)
            sx = ssca_mapped(sig, N, Np)
            mc = max_cut(sx)
            dc = dcs(sx)
            # plot_ssca_diamond(sx)
            plot_ssca_triangle(sx)

    with timeit("Python") as _:
        for sig in signals:
            # Np = 256
            Np = 64
            # N = floor_power_of_2(len(sig) - Np)
            N = 4096
            # sx = ssca_py(sig, N, Np, map_output=False)
            sx = ssca_py(sig, N, Np)
            mc = max_cut(sx)
            dc = dcs(sx)
            # plot_ssca_diamond(sx)
            plot_ssca_triangle(sx)

    import matplotlib.pyplot as plt

    plt.show()
