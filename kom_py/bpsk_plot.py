#!/usr/bin/env python3
from komunikilo import tx_bpsk

import matplotlib.pyplot as plt
from typing import List

if __name__ == '__main__':
    SAMPLE_RATE: float = 2**16  # 65536
    SYMBOL_RATE: float = 2**8  # 512
    CARRIER_FREQ: float = 2000

    data: List[bool] = [True, False, True, True, False, False, True, False]
    signal: List[float] = tx_bpsk(data, SAMPLE_RATE, SYMBOL_RATE, CARRIER_FREQ)
    # signal: List[float] = tx_bpsk([True, False], 2**16, 2**4, 2**12)
    # signal: List[float] = tx_bpsk([True, False], 2**16, 2**4, 2.1**11)

    plt.plot(signal)
    plt.show()
