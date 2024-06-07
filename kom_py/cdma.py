#!/usr/bin/env python3
from psk import tx_bpsk

import numpy as np
from typing import List

KEY = np.array([1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1])


def tx_bpsk_cdma(
    data: List[bool], samp_rate: int, symb_rate: int, carrier_freq: float
) -> List[float]:
    samples_per_symbol = int(samp_rate / symb_rate)
    samples_per_chip = int(samples_per_symbol / len(KEY))
    assert samp_rate / 2 >= carrier_freq
    assert samp_rate / 2 >= len(KEY) * symb_rate
    assert samp_rate % symb_rate == 0
    assert (samp_rate / symb_rate).is_integer()
    assert (samp_rate / (symb_rate * len(KEY))).is_integer()

    bpsk = tx_bpsk(data, samp_rate, symb_rate, carrier_freq)
    spreading_code = np.tile(
        np.repeat(KEY, samples_per_chip), len(bpsk) // samples_per_chip // len(KEY)
    )
    print(f"BPSK: {len(bpsk)} | spreading_code: {len(spreading_code)}")
    assert len(bpsk) == len(spreading_code)

    return bpsk * spreading_code


def rx_bpsk_cdma(
    signal: List[float], samp_rate: int, symb_rate: int, carrier_freq: float
) -> List[bool]:
    samples_per_symbol = int(samp_rate / symb_rate)
    samples_per_chip = int(samples_per_symbol / len(KEY))
    assert samp_rate / 2 >= carrier_freq
    assert samp_rate / 2 >= len(KEY) * symb_rate
    assert samp_rate % symb_rate == 0
    assert (samp_rate / symb_rate).is_integer()
    assert (samp_rate / (symb_rate * len(KEY))).is_integer()
    filter = [1 for _ in range(samples_per_symbol)]

    if_sig = [
        s_i * np.cos(2 * np.pi * carrier_freq * (i / samp_rate))
        for (i, s_i) in enumerate(signal)
    ]

    sig = np.convolve(if_sig, filter)

    return [th_i > 0 for th_i in sig[::samples_per_symbol]][1:]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fs = 48_000
    sr = 1000
    fc = 2500

    num_bits = 10
    data = np.random.choice([True, False], num_bits)
    bpsk_sig = tx_bpsk(data, fs, sr, fc)
    cdma_sig = tx_bpsk_cdma(data, fs, sr, fc)

    _, ax = plt.subplots()
    ax.plot(bpsk_sig)

    _, ax = plt.subplots()
    plt.plot(cdma_sig)

    plt.show()
