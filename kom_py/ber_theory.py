#!/usr/bin/env python3
from komunikilo import (
    awgn,
    random_data,
    tx_bpsk,
    rx_bpsk,
    tx_qpsk,
    rx_qpsk,
)
from util import undb, timeit

import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
from random import gauss
from scipy.special import erfc
from typing import List


def ber(tx: List[bool], rx: List[bool]) -> float:
    return sum(0 if tx_i == rx_i else 1 for tx_i, rx_i in zip(tx, rx)) / min(
        len(tx),
        len(rx),
    )


def ber_bpsk(eb_n0: float) -> float:
    return 0.5 * erfc(np.sqrt(eb_n0))


def ber_qpsk(eb_n0: float) -> float:
    return 0.5 * erfc(np.sqrt(eb_n0)) - 0.25 * erfc(np.sqrt(eb_n0)) ** 2


def get_ber(data, tx_sig, rx_func, n0) -> float:
    noisy_sig = awgn(tx_sig, n0)
    sig_hat = rx_func(noisy_sig)
    return ber(data, sig_hat)


def noise_sig(tx_sig, n0, konst) -> float:
    return [s_i + gauss(sigma=n0) * konst for s_i in signal]


##
# The goal of this file is to check the BER matches theory.
##

if __name__ == "__main__":
    SAMPLE_RATE: float = 2**16  # 65536
    SYMBOL_RATE: float = 2**8  # 256
    CARRIER_FREQ: float = 4000
    CARRIER_FREQ: float = 2121

    NUM_BITS: int = 9001

    snrs = undb(np.linspace(-25, 6, 25))
    # snrs = np.linspace(1e-100, 15, 100)

    def bpsk_signal(data: List[bool]):
        return tx_bpsk(data, SAMPLE_RATE, SYMBOL_RATE, CARRIER_FREQ)

    def bpsk_signal_rx(signal: List[float]):
        return rx_bpsk(signal, SAMPLE_RATE, SYMBOL_RATE, CARRIER_FREQ)

    def qpsk_signal(data: List[bool]):
        return tx_qpsk(data, SAMPLE_RATE, SYMBOL_RATE, CARRIER_FREQ)

    def qpsk_signal_rx(signal: List[float]):
        return rx_qpsk(signal, SAMPLE_RATE, SYMBOL_RATE, CARRIER_FREQ)

    data: List[bool] = random_data(NUM_BITS)

    with multiprocessing.Pool() as p, timeit("BPSK") as _:
        signal: List[float] = bpsk_signal(data)
        signal = 0.5 * np.array(signal)
        eb = sum(s_i**2 for s_i in signal) / NUM_BITS
        n0s: np.array = np.nan_to_num(
            np.sqrt(SAMPLE_RATE * eb / (2 * snrs)) / SYMBOL_RATE
        )  # perfect...

        bers = p.starmap(get_ber, [(data, signal, bpsk_signal_rx, n0) for n0 in n0s])

        fig, ax = plt.subplots()
        ax.plot(snrs, bers)
        ax.plot(snrs, [ber_bpsk(snr) for snr in snrs])
        ax.set_yscale("log")

    with multiprocessing.Pool() as p, timeit("QPSK") as _:
        signal: List[float] = qpsk_signal(data)
        eb = sum(s_i**2 for s_i in signal) / NUM_BITS
        n0s: np.array = np.nan_to_num(np.sqrt(SAMPLE_RATE * eb / (2 * snrs)) / SYMBOL_RATE)
        bers = p.starmap(
            get_ber, [(data, signal, qpsk_signal_rx, n0) for n0 in n0s]
        )

        fig, ax = plt.subplots()
        ax.plot(snrs, bers)
        ax.plot(snrs, [ber_qpsk(snr) for snr in snrs])
        ax.plot(snrs, [ber_bpsk(snr) for snr in snrs])
        ax.set_yscale("log")

    plt.show()

    # for n0 in n0s:
    #     data = random_data(NUM_BITS)
    #     sig = awgn(bpsk_signal(data), n0)
    #     rx = rx_bpsk(sig, SAMPLE_RATE, SYMBOL_RATE, CARRIER_FREQ)
    #     bers.append(ber(data, rx))
