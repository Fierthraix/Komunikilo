#!/usr/bin/env python3

from util import db, undb

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from typing import List


def logistic_curve(x: float, x0=0, L=1, k=1) -> float:
    return L / (1 + np.exp(-k * (x - x0)))


def ber(eb_n0: float, offset=0) -> float:
    return 0.5 * erfc(np.sqrt(eb_n0 - offset))


def find_idx_closest_to_point(point: float, vec: List[float]) -> int:
    idx: int = 0
    while True:
        if point < vec[idx]:
            ...


if __name__ == "__main__":

    good_ber = 0.05
    good_pd = 0.95

    # Plot the two cases of covert and non-covert.
    fig, ax = plt.subplots(2, 1)

    ##
    # Uncovert Signal
    ##
    ber_ax = ax[0]
    pd_ax = ax[0].twinx()

    snrs = undb(np.linspace(-25, 6, 100))
    ber_ax.plot(db(snrs), ber(snrs), color="Red")
    ber_ax.plot(1.31225, good_ber, 'ro')
    ber_ax.axhline(good_ber, color='Red', ls='--', label=f"Acceptable BER ({good_ber})")
    ber_ax.set_ylim([0, 0.51])
    ber_ax.tick_params(axis="y", colors="Red")
    ber_ax.set_ylabel("Bit Error Rate (BER)", color="Red")
    ber_ax.legend(loc='best')

    uncovert_log = lambda x: logistic_curve(db(x), x0=-15)
    pd_ax.plot(db(snrs), uncovert_log(snrs), color="Blue")
    pd_ax.set_ylim([0, 1.1])
    pd_ax.plot(-12.0556, good_pd, "bo")
    pd_ax.axhline(good_pd, color='Blue', ls='--', label=f'Acceptable ℙd ({good_pd})')
    pd_ax.tick_params(axis="y", colors="Blue")
    pd_ax.set_ylabel("Probability of Detection ($\mathcal{P}_D$)", color="Blue")
    pd_ax.legend(loc='best')
    ax[0].set_xlabel("Signal to Noise Ratio dB (SNR dB)")
    ax[0].set_title("Uncovert Case")

    ##
    # Covert Signal
    ##
    snrs = undb(np.linspace(-5, 10, 100))
    ber_ax = ax[1]
    pd_ax = ax[1].twinx()
    ber_ax.plot(db(snrs), ber(snrs), color="Red")
    ber_ax.plot(1.31225, good_ber, 'ro')
    ber_ax.axhline(good_ber, color='Red', ls='--', label=f"Acceptable BER ({good_ber})")
    ber_ax.set_ylim([0, 0.51])
    ber_ax.tick_params(axis="y", colors="Red")
    ber_ax.set_ylabel("Bit Error Rate (BER)", color="Red")
    ber_ax.legend(loc='best')

    covert_log = lambda x: logistic_curve(db(x), x0=3)
    pd_ax.plot(db(snrs), covert_log(snrs), color="Blue")
    pd_ax.set_ylim([0, 1.1])
    pd_ax.plot(5.9444, good_pd, "bo")
    pd_ax.axhline(good_pd, color='Blue', ls='--', label=f'Acceptable ℙd ({good_pd})')
    pd_ax.tick_params(axis="y", colors="Blue")
    pd_ax.set_ylabel("Probability of Detection ($\mathcal{P}_D$)", color="Blue")
    pd_ax.legend(loc='best')
    ax[1].set_xlabel("Signal to Noise Ratio dB (SNR dB)")
    ax[1].set_title("Covert Case")

    plt.show()
