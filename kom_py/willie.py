#!/usr/bin/env python3
from psk import tx_bpsk, rx_bpsk, tx_baseband_bpsk, rx_baseband_bpsk
from typing import Iterable, List
import matplotlib.pyplot as plt
from random import choice, gauss
import numpy as np
from scipy.stats import norm, normaltest


def awgn(signal: Iterable[float], n0) -> List[float]:
    return [s_i + gauss(sigma=n0) for s_i in signal]


def awgn_iq(signal: Iterable[complex], n0) -> List[complex]:
    return [s_i + (gauss(sigma=n0) + 1j * gauss(sigma=n0)) for s_i in signal]


def avg_energy(signal: Iterable[float]) -> float:
    """Square, integrate, and average."""
    return sum(s_i**2 for s_i in signal) / len(signal)


def avg_energy_iq(signal: Iterable[complex]) -> float:
    """Square, integrate, and average."""
    return sum(abs(s_i**2) for s_i in signal) / len(signal) / 2


def scale(vec: Iterable[float], scalar) -> List[float]:
    return [scalar * v_i for v_i in vec]


def neyman_pearson(signal: Iterable[float], n0: float, α: float = 0.05) -> bool:
    """Do a neyman-pearson test on the signal, where H0 is the N0"""
    mu0 = 0
    mu1 = np.mean(signal)
    sigma0 = n0
    sigma1 = np.std(signal)
    print(f"μ = {mu1} || σ = {sigma1}")

    # likelihood_ratio = np.prod(
    #     norm.pdf(signal, mu1, sigma1) / np.prod(norm.pdf(signal, mu0, sigma0))
    # )
    p0 = np.prod(norm.pdf(signal, mu0, sigma0))
    p1 = np.prod(norm.pdf(signal, mu1, sigma1))
    likelihood_ratio = p0 / p1

    threshold = norm.ppf(1 - α)
    print(f"p0: {p0} || p1: {p1} || {likelihood_ratio} || k = {threshold}")
    return likelihood_ratio > threshold


def p_value(signal: Iterable[float], n0: float) -> bool:
    res = normaltest(signal)
    print(f"p-val: {res.pvalue}")
    return res.pvalue < 0.05


def test_1():
    NUM_BITS = 10_000
    data = [choice((True, False)) for _ in range(NUM_BITS)]
    samp_rate = 441_00
    symb_rate = 450
    fc = 2000

    tx_sig = tx_bpsk(data, samp_rate, symb_rate, fc)

    N0s = list(np.arange(1e-10, 5, 0.1))
    p_vals = []
    e_vals = []
    for N0 in N0s:
        # chan_sig = awgn(np.concatenate(([0] * 4000, tx_sig, [0] * 4000)), N0)
        chan_sig = awgn(tx_sig, N0)
        res = normaltest(chan_sig)
        p_vals.append(res.pvalue)

        asdf = awgn([0 for _ in range(len(tx_sig))], N0)
        res2 = normaltest(asdf)
        e_vals.append(res2.pvalue)

    plt.plot(N0s, p_vals)
    plt.plot(N0s, e_vals)
    # plt.plot(N0s, p_vals)
    plt.show()


if __name__ == "__main__":
    # data = [True, False, True, True, False, False, True, False]
    NUM_BITS = 40 #_000
    data = [choice((True, False)) for _ in range(NUM_BITS)]
    samp_rate = 441_00
    symb_rate = 450
    fc = 2000

    # N0 = 5
    N0 = 2
    tx_sig = tx_bpsk(data, samp_rate, symb_rate, fc)
    # chan_sig = awgn([0] * 4000 + tx_sig + [0] * 4000, N0)
    chan_sig = awgn(np.concatenate(([0] * 4000, tx_sig, [0] * 4000)), N0)
    empty_chan = awgn([0] * len(chan_sig), N0)

    # sig_nrg = avg_energy(tx_sig)
    # noise_nrg = avg_energy(chan_sig)
    # empty_noise_nrg = avg_energy(empty_chan)
    # print("REAL SIG:", sig_nrg, noise_nrg, empty_noise_nrg)

    # tx_iq = tx_baseband_bpsk(data)
    # chan_iq = awgn_iq(tx_iq, N0)
    # baseband_empty_chan = awgn_iq([0j] * len(chan_iq), N0)
    # baseband_nrg = avg_energy_iq(tx_iq)
    # baseband_noise_nrg = avg_energy_iq(chan_iq)
    # baseband_empty_noise_nrg = avg_energy_iq(baseband_empty_chan)
    # print("BASEBAND:", baseband_nrg, baseband_noise_nrg, baseband_empty_noise_nrg)

    # asdf = neyman_pearson(tx_sig, N0)
    asdf = neyman_pearson(empty_chan, N0)
    print(asdf)

    fdsa = p_value(empty_chan, N0)
    print(fdsa)

    asdf1 = neyman_pearson(chan_sig, N0)
    print(asdf1)
    fdsa1 = p_value(chan_sig, N0)
    print(fdsa1)
    # plt.plot(chan_sig)
    # plt.plot(tx_sig)
    # plt.show()

    # signal = chan_iq
    # signal = baseband_empty_chan
    # print(sum(s_i * s_i.conjugate() for s_i in signal) / len(signal) / 2)
    # print(sum(abs(s_i * s_i.conjugate()) for s_i in signal) / len(signal) / 2)
    # print(sum(abs(s_i**2) for s_i in signal) / len(signal) / 2)

    test_1()
