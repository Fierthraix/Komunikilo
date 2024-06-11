#!/usr/bin/env python3
from typing import List
from util import bit_to_nrz, bitz_to_nrz, chunk
import numpy as np


def tx_baseband_bpsk(data: List[bool]) -> List[complex]:
    return [complex(1) if bit else complex(-1) for bit in data]


def rx_baseband_bpsk(signal: List[complex]) -> List[bool]:
    return [s_i.real >= 0 for s_i in signal]


def tx_baseband_qpsk(data: List[bool]) -> List[complex]:
    return [
        complex(bit_to_nrz(b1) * np.sqrt(2), bit_to_nrz(b2) * np.sqrt(2))
        for (b1, b2) in chunk(data, 2, fillvalue=False)
    ]


def rx_baseband_qpsk(signal: List[complex]) -> List[bool]:
    return list(np.array([[s_i.real >= 0, s_i.imag >= 0] for s_i in signal]).flatten())


def tx_bpsk(
    data: List[bool], samp_rate: int, symb_rate: int, carrier_freq: float
) -> List[float]:
    samples_per_symbol = int(samp_rate / symb_rate)
    assert (
        samp_rate / 2 >= carrier_freq
    ), f"SAMP RATE: {samp_rate} | FREQ: {carrier_freq}"
    assert samp_rate % symb_rate == 0

    # Expand each bit in the time array:
    datums = np.repeat(list(bitz_to_nrz(data)), samples_per_symbol)

    # Make time array
    time = np.arange(len(datums)) / samp_rate

    return datums * np.cos(2 * np.pi * carrier_freq * time)


def rx_bpsk(
    signal: List[float], samp_rate: int, symb_rate: int, carrier_freq: float
) -> List[bool]:
    samples_per_symbol = int(samp_rate / symb_rate)
    assert samp_rate / 2 >= carrier_freq
    assert samp_rate % symb_rate == 0
    filter = [1 for _ in range(samples_per_symbol)]

    if_sig = [
        s_i * np.cos(2 * np.pi * carrier_freq * (i / samp_rate))
        for (i, s_i) in enumerate(signal)
    ]

    sig = np.convolve(if_sig, filter)

    return [th_i > 0 for th_i in sig[::samples_per_symbol]][1:]


def tx_qpsk(
    data: List[bool], samp_rate: int, symb_rate: int, carrier_freq: float
) -> List[float]:
    samples_per_symbol = int(samp_rate / symb_rate)
    assert samp_rate / 2 >= carrier_freq
    assert samp_rate % symb_rate == 0

    symbols = (c_i for c_i in tx_baseband_qpsk(data) for _ in range(samples_per_symbol))
    # assert len(symbols) == (len(data) // 2) * samples_per_symbol

    # Make time array
    # time = [i / samp_rate for i in range(len(symbols))]
    return [
        (c_i * np.exp(1j * np.pi * carrier_freq * (i / samp_rate))).real
        for (i, c_i) in enumerate(symbols)
    ]


def rx_qpsk(
    signal: List[float], samp_rate: int, symb_rate: int, carrier_freq: float
) -> List[bool]:
    samples_per_symbol = int(samp_rate / symb_rate)
    assert samp_rate / 2 >= carrier_freq
    assert samp_rate % symb_rate == 0
    filter = [1 for _ in range(samples_per_symbol)]

    ii = np.convolve(
        [
            s_i * np.cos(2 * np.pi * carrier_freq * (i / samp_rate))
            for (i, s_i) in enumerate(signal)
        ],
        filter,
    )

    qi = np.convolve(
        [
            s_i * -np.sin(2 * np.pi * carrier_freq * (i / samp_rate))
            for (i, s_i) in enumerate(signal)
        ],
        filter,
    )

    i_bits = [i > 0 for i in ii[::samples_per_symbol]][1:]
    q_bits = [i > 0 for i in qi[::samples_per_symbol]][1:]

    return list(np.array(list(zip(i_bits, q_bits))).flatten())
