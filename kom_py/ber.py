#!/usr/bin/env python3
from komunikilo import (
    awgn,
    random_data,
    tx_bpsk,
    rx_bpsk,
    tx_qpsk,
    rx_qpsk,
    tx_ofdm,
    rx_ofdm,
)
from util import undb

from dataclasses import dataclass
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
from random import gauss
from typing import Callable, List, Optional


@dataclass
class BitErrorTestResults:
    name: str
    data: List[bool]
    tx_sig: List[float]
    rx_fn: Callable[[List[float]], List[bool]]
    snrs: List[float]
    bers: Optional[List[float]] = None


def ber(tx: List[bool], rx: List[bool]) -> float:
    return sum(0 if tx_i == rx_i else 1 for tx_i, rx_i in zip(tx, rx)) / min(
        len(tx),
        len(rx),
    )


def get_ber(data, tx_sig, rx_func, n0) -> float:
    noisy_sig = awgn(tx_sig, n0)
    sig_hat = rx_func(noisy_sig)
    return ber(data, sig_hat)


def noise_sig(tx_sig, n0, konst) -> float:
    return [s_i + gauss(sigma=n0) * konst for s_i in signal]


def calc_ber(ber_test: BitErrorTestResults) -> BitErrorTestResults:

    eb = sum(s_i**2 for s_i in ber_test.tx_sig) / len(ber_test.data)
    n0s: np.array = np.nan_to_num(
        np.sqrt(SAMPLE_RATE * eb / (2 * ber_test.snrs)) / SYMBOL_RATE
    )

    with multiprocessing.Pool() as p:
        ber_test.bers = p.starmap(
            get_ber,
            [(ber_test.data, ber_test.tx_sig, ber_test.rx_fn, n0) for n0 in n0s],
        )

    return ber_test


if __name__ == "__main__":
    SAMPLE_RATE: float = 2**16  # 65536
    SYMBOL_RATE: float = 2**8  # 256
    CARRIER_FREQ: float = 4000
    CARRIER_FREQ: float = 2121

    NUM_BITS: int = 9001

    snrs = undb(np.linspace(-25, 6, 25))
    # snrs = np.linspace(1e-100, 15, 100)

    data: List[bool] = random_data(NUM_BITS)

    def bpsk_rx(signal: List[float]) -> List[bool]:
        return rx_bpsk(signal, SAMPLE_RATE, SYMBOL_RATE, CARRIER_FREQ)

    def qpsk_rx(signal: List[float]) -> List[bool]:
        return rx_qpsk(signal, SAMPLE_RATE, SYMBOL_RATE, CARRIER_FREQ)

    def ofdm_signal(data: List[bool]) -> np.array:
        subcarriers = 16
        pilots = int(subcarriers * 0.8)
        return tx_ofdm(
            data, subcarriers, pilots, SAMPLE_RATE, SYMBOL_RATE, CARRIER_FREQ
        )

    def ofdm_rx(signal: List[float]) -> List[bool]:
        subcarriers = 16
        pilots = int(subcarriers * 0.8)
        return rx_ofdm(
            data, subcarriers, pilots, SAMPLE_RATE, SYMBOL_RATE, CARRIER_FREQ
        )

    comms_schemes: List[BitErrorTestResults] = [
        BitErrorTestResults(
            "BPSK",
            data,
            tx_bpsk(data, SAMPLE_RATE, SYMBOL_RATE, CARRIER_FREQ),
            bpsk_rx,
            snrs,
        ),
        BitErrorTestResults(
            "QPSK",
            data,
            tx_qpsk(data, SAMPLE_RATE, SYMBOL_RATE, CARRIER_FREQ),
            qpsk_rx,
            snrs,
        ),
        BitErrorTestResults(
            "OFDM",
            data,
            ofdm_signal(data),
            ofdm_rx,
            snrs,
        ),
    ]

    comms_schemes: List[BitErrorTestResults] = [
        calc_ber(scheme)
        for scheme in comms_schemes
    ]

    fig, ax = plt.subplots()
    for scheme in comms_schemes:
        ax.plot(snrs, scheme.bers, label=scheme.name)
    ax.set_yscale("log")
    ax.legend(loc='best')
    plt.show()
