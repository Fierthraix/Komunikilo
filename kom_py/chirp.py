#!/usr/bin/env python3
from komunikilo import chirp

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import chirp as scipy_chirp


def plot_chirp():
    high_freq = 6  # Hz
    low_freq = 1  # Hz
    sample_rate = 150  # Hz
    chirp_rate = 1 / 10  # chirps / second

    upchirp = chirp(chirp_rate, sample_rate, low_freq, high_freq)
    downchirp = chirp(chirp_rate, sample_rate, high_freq, low_freq)

    t: np.array = np.linspace(0, 1 / chirp_rate, len(upchirp))

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t, downchirp)
    ax[1].plot(t, upchirp)
    ax[0].set_xlabel("t (sec)")
    ax[1].set_xlabel("t (sec)")
    fig.suptitle("Linear Chirp, f(0)=6, f(10)=1")


def plot_chirp_py():
    high_freq = 6  # Hz
    low_freq = 1  # Hz
    sample_rate = 150  # Hz
    chirp_rate = 1 / 10  # chirps / second

    upchirp = chirpz(chirp_rate, sample_rate, low_freq, high_freq)
    downchirp = chirpz(chirp_rate, sample_rate, high_freq, low_freq)

    t: np.array = np.linspace(0, 1 / chirp_rate, len(upchirp))

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t, downchirp)
    ax[1].plot(t, upchirp)
    ax[0].set_xlabel("t (sec)")
    ax[1].set_xlabel("t (sec)")
    fig.suptitle("Linear Chirp, f(0)=6, f(10)=1")


def plot_chirp_scipy():
    high_freq = 6  # Hz
    low_freq = 1  # Hz
    t = np.linspace(0, 10, 1500)
    downchirp = scipy_chirp(t, f0=high_freq, f1=low_freq, t1=max(t), method="linear")
    upchirp = scipy_chirp(t, f0=low_freq, f1=high_freq, t1=max(t), method="linear")
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t, downchirp)
    ax[1].plot(t, upchirp)
    ax[0].set_xlabel("t (sec)")
    ax[1].set_xlabel("t (sec)")
    fig.suptitle("Linear Chirp, f(0)=6, f(10)=1")


def chirpz(
    chirp_rate: float, sample_rate: int, f0: float, f1: float, periods=1, phase_rad=0
):

    chirp_period = 1 / chirp_rate
    c = (f1 - f0) / chirp_period
    samps_per_chirp = int(sample_rate / chirp_rate)
    t_s = np.linspace(0, chirp_period, samps_per_chirp)

    # Phase, phi_Hz, is integral of frequency, f(t) = ct + f0.
    phi_Hz = (c * t_s**2) / 2 + (f0 * t_s)  # Instantaneous phase.
    phi_rad = 2 * np.pi * phi_Hz  # Convert to radians.
    phi_rad += phase_rad  # Offset by user-specified initial phase.
    # return np.real(np.exp(1j * phi_rad))
    return np.sin(phi_rad)


def chirpCtr(fs_Hz, fc_Hz, rep_Hz, bw_Hz, phase_rad=0):
    """
    Convenience function to create a chirp based on center frequency and
    bandwidth.  It simply calculates start and stop freuqncies of chirp and
    calls the chirp creation function.
    Inputs
     fs_Hz: sample rate in Hz of chirp waveform.
     fc_Hz: float, center frequency in Hz of the chirp.
     rep_Hz: integer, number of full chirps per second.
     bw_Hz: float, bandwidth of chirp.
     phase_rad: phase in radians at waveform start, default is 0.
    """
    f0_Hz = fc_Hz - bw_Hz / 2.0
    f1_Hz = fc_Hz + bw_Hz / 2.0
    return chirpz(fs_Hz, rep_Hz, f0_Hz, f1_Hz, phase_rad)


if __name__ == "__main__":
    plot_chirp()
    plot_chirp_py()
    plot_chirp_scipy()
    plt.show()
