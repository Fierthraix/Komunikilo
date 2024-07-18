#!/usr/bin/env python3
import numpy as np
from typing import List

from util import timeit
from willie import awgn


def plot_ssca_triangle(s, log=True):
    import matplotlib.pyplot as plt

    x = np.linspace(1, 0, s.shape[1] // 2 + 1)
    y = np.linspace(-0.5, 0.5, s.shape[0])
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ss = s[:, : s.shape[1] // 2 + 1]
    if log:
        ax.plot_surface(X, Y, 10 * np.log(np.abs(ss)), cmap="plasma")
    else:
        ax.plot_surface(X, Y, np.abs(ss), cmap="plasma")


def plot_ssca_diamond(s, log=True):
    import matplotlib.pyplot as plt

    x = np.linspace(-1, 1, s.shape[1])
    y = np.linspace(-0.5, 0.5, s.shape[0])
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ss = s[:, : s.shape[1]]
    if log:
        ax.plot_surface(X, Y, 10 * np.log(np.abs(ss)), cmap="plasma")
    else:
        ax.plot_surface(X, Y, np.abs(ss), cmap="plasma")


def plot_lambda(l_a: np.ndarray):
    import matplotlib.pyplot as plt

    x = np.linspace(0, 1, len(l_a))
    _, ax = plt.subplots()
    ax.plot(x, l_a)


# TODO:
# * Check if FFTshfit happens in same spot of matlab code.
def ssca(s: List[float], N, Np, map_output=True) -> np.ndarray:
    s = np.array(s[: N + Np])  # Limit to one window for now.
    assert s.shape == (N + Np,), f"{s.shape} != {N + Np}"
    # STEP 1:
    x = np.array([s[i : i + Np] for i in range(N)])
    assert x.shape == (N, Np)

    # STEP 2:
    # `a` is the window function.
    a = np.hamming(Np)

    # xat = np.array([np.fft.fft(xi * a, Np) for xi in x])
    xat = np.array([np.fft.fftshift(np.fft.fft(xi * a, Np)) for xi in x])
    assert xat.shape == (N, Np)

    def e(k, m):
        return np.exp(-2j * np.pi * m * k / Np)

    # STEP 3:
    em = np.array([[e(k, m) for k in range(-Np // 2, Np // 2)] for m in range(N)])

    g = np.array([np.hamming(N) for _ in range(Np)]).transpose()

    xs = np.conj([[s[Np // 2 + i] for _ in range(Np)] for i in range(N)])

    assert all(m.shape == (N, Np) for m in (xat, em, xs, g))
    xg = xat * em * xs * g

    # STEP 4:
    # sx = np.array([np.fft.fft(xgi, N) for xgi in xg.transpose()]).transpose()
    sx = np.array(
        [np.fft.fftshift(np.fft.fft(xgi, N)) for xgi in xg.transpose()]
    ).transpose()
    assert sx.shape == (N, Np), f"{sx.shape}, {(N , Np)}"

    if map_output:
        # STEP 5:
        # f = k / (2 * Νp) - q / (2 * N)
        # a = k / Np + q / N
        Sxf = np.zeros((Np + 1, 2 * N + 1), dtype=complex)
        for q_p in range(N):
            for k_p in range(Np):
                f = k_p / (2 * Np) - q_p / (2 * N)
                a = k_p / Np + q_p / N
                k = int(Np * (f + 0.5))
                q = int(N * a)
                Sxf[k][q] = sx[q_p][k_p]
        return Sxf
    else:
        return sx


def max_cut(sx: np.ndarray) -> np.ndarray:
    # l_max(t, a) = 10 log_10( max_f( | S(a, f) |^2 ) )
    # TODO: verify if axis zero or axis 1.
    λ = np.abs(np.max(sx, axis=1)) ** 2
    return 10 * np.log10(λ)
    # return λ


def dcs(sx: np.ndarray) -> np.ndarray:
    # λ = np.sum(np.abs(sx[1:]) ** 2, axis=0) / np.abs(sx[0, :]) ** 2
    λ = np.sum(np.abs(sx) ** 2, axis=0) / np.abs(sx[0, :]) ** 2
    return 10 * np.log10(λ)
    # return λ


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
            sx = ssca(sig, N, Np)
            mc = max_cut(sx)
            dc = dcs(sx)
        plot_ssca_triangle(sx)
        # plot_ssca_diamond(sx)
        plot_lambda(mc)
        plot_lambda(dc)
        import matplotlib.pyplot as plt

        # plt.show()

    results = [do_signal(signal) for signal in signals]
    # with Pool(8) as p:
    #     results = p.map(do_signal, signals)

    import matplotlib.pyplot as plt
    plt.show()
