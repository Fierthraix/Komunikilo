#!/usr/bin/env python3

from komunikilo import tx_fh_css, random_data

import matplotlib.pyplot as plt
from typing import List


# TODO: FIXME: Plot the FH-CSS spectrum to see the chirps as lines.
if __name__ == "__main__":
    NUM_FREQS: int = 4
    LOW_FREQ: float = 5e3
    HIGH_FREQ: float = 25e3
    SAMPLE_RATE = 65536
    SYMBOL_RATE = 8

    NUM_BITS = 16

    data: List[bool] = random_data(NUM_BITS)

    signal: List[float] = tx_fh_css(
        data, SAMPLE_RATE, SYMBOL_RATE, LOW_FREQ, HIGH_FREQ, NUM_FREQS
    )

    # t = np.linspace(0, SYMBOL_RATE * NUM_BITS, len(signal))
    # plt.plot(t, signal)

    plt.specgram(signal, Fs=SAMPLE_RATE)  # , Fc=(HIGH_FREQ-LOW_FREQ)/2)
    plt.show()
