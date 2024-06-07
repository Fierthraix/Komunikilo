#!/usr/bin/env python3
from psk import tx_bpsk as tx_bpsk_py
from komunikilo import awgn, energy_detector, tx_bpsk, max_cut_detector, random_data
from ssca import max_cut, ssca as ssca_py
from util import timeit
from willie import avg_energy, awgn as awgn_py

import functools
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from typing import List


def make_data(num_bits: int) -> np.array:
    return np.random.choice([True, False], num_bits)


def __log_regress(h0_λs: np.array, h1_λs: np.array) -> pd.DataFrame:
    x_var = np.concatenate((h0_λs, h1_λs)).reshape(-1, 1)
    y_var = np.concatenate((np.zeros(len(h0_λs)), np.ones(len(h1_λs))))
    x_train, x_test, y_train, y_test = train_test_split(
        x_var, y_var, test_size=0.5, random_state=0
    )

    log_regression = LogisticRegression()
    log_regression.fit(x_train, y_train)

    y_pred_proba = log_regression.predict_proba(x_test)[::, 1]
    fpr, tpr, thresholds = metrics.roc_curve(
        y_test, y_pred_proba, drop_intermediate=False
    )

    df_test = pd.DataFrame(
        {
            "x": x_test.flatten(),
            "y": y_test,
            "proba": y_pred_proba,
        }
    )

    # sort it by predicted probabilities
    # because thresholds[1:] = y_proba[::-1]
    df_test.sort_values(by="proba", inplace=True)
    df_test["tpr"] = tpr[1:][::-1]
    df_test["fpr"] = fpr[1:][::-1]
    df_test["youden_j"] = df_test.tpr - df_test.fpr
    return df_test


def __auc(h0_λs: np.array, h1_λs: np.array) -> float:
    x_var = np.concatenate((h0_λs, h1_λs)).reshape(-1, 1)
    y_var = np.concatenate((np.zeros(len(h0_λs)), np.ones(len(h1_λs))))
    x_train, x_test, y_train, y_test = train_test_split(
        x_var, y_var, test_size=0.5, random_state=0
    )
    log_regression = LogisticRegression()
    log_regression.fit(x_train, y_train)
    y_pred_proba = log_regression.predict_proba(x_test)[::, 1]
    fpr, tpr, thresholds = metrics.roc_curve(
        y_test, y_pred_proba, drop_intermediate=False
    )
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    return auc


def plot_multi_roc_curve(
    h0_λs: List[np.array], h1_λs: List[np.array], n0s: List[float]
):
    df_tests: List[pd.DataFrame] = [
        __log_regress(h0_λ, h1_λ) for h0_λ, h1_λ, n0 in zip(h0_λs, h1_λs, n0s)
    ]
    cut_offs: List[object] = [
        df_test.sort_values(by="youden_j", ascending=False, ignore_index=True).iloc[0]
        for df_test in df_tests
    ]
    cut_off_x: List[float] = [c.fpr for c in cut_offs]
    cut_off_y: List[float] = [c.tpr for c in cut_offs]
    fig, ax = plt.subplots()
    for df_test in df_tests:  # [df_tests[5]]: #, df_tests[-5]):
        metrics.RocCurveDisplay(fpr=df_test.fpr, tpr=df_test.tpr).plot(ax=ax)
        ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")
    # ax.legend(loc=4)
    # ax.plot(cut_off_x, cut_off_y, label="Optimal λ", ls="-.")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")
    ax.set_title("ROC Curve")
    # ax.axline(xy1=(0, 0), slope=1, color="r", ls=":")
    # plt.show()


def plot_auc_vs_snr(h0_λs: List[np.array], h1_λs: List[np.array], snrs: List[float]):
    aucs: List[float] = [__auc(h0_λ, h1_λ) for h0_λ, h1_λ in zip(h0_λs, h1_λs)]
    fig, ax = plt.subplots()
    ax.plot(snrs, aucs)
    ax.set_xlabel("SNR (Linear)")
    ax.set_ylabel("AUC")
    ax.set_title("AUC vs SNR Curve")
    # plt.show()


def get_best_lambda(h0_λs: np.array, h1_λs: np.array) -> float:
    df_test = __log_regress(h0_λs, h1_λs)
    cut_off = df_test.sort_values(
        by="youden_j", ascending=False, ignore_index=True
    ).iloc[0]

    """
    # Plot the λs.
    binification = 128
    h0_hist = rv_histogram(np.histogram(h1_λs, bins=binification))
    h1_hist = rv_histogram(np.histogram(h0_λs, bins=binification))
    x_hist: np.array = np.linspace(
        min(itertools.chain(h1_λs, h0_λs)),
        max(itertools.chain(h1_λs, h0_λs)),
        binification,
    )
    fig, ax = plt.subplots()
    ax.plot(x_hist, h0_hist.pdf(x_hist))
    ax.plot(x_hist, h1_hist.pdf(x_hist))
    ax.axvline(cut_off.x, color="k", ls="--")
    # plt.show()
    """

    return cut_off.x


if __name__ == "__main__":
    SAMPLE_RATE: float = 44100
    SYMBOL_RATE: float = 900
    CARRIER_FREQ: float = 1800

    NUM_BITS: int = 100
    NUM_ATTEMPTS: int = 100

    # snrs: np.array = np.linspace(10e-9, 0.5, 10) # np.linspace(0.5, 2, 10)
    # snrs: np.array = np.linspace(0.5, 2, 10)
    snrs: np.array = np.linspace(1e-5, 1, 25)

    def bpsk_signal() -> np.array:
        return tx_bpsk(random_data(NUM_BITS), SAMPLE_RATE, SYMBOL_RATE, CARRIER_FREQ)

    def bpsk_signal_py() -> np.array:
        return tx_bpsk_py(make_data(NUM_BITS), SAMPLE_RATE, SYMBOL_RATE, CARRIER_FREQ)

    def max_cut_detector_py(signal: np.array) -> float:
        Np = 64
        # N = floor_power_of_2(len(signal) - 64)
        N = 4096
        Sxf = ssca_py(signal, N, Np, map_output=False)
        return np.max(max_cut(Sxf))

    def energy_detector_py(signal: np.array) -> float:
        # import pdb; pdb.set_trace()
        # return np.sum(signal**2)
        if not isinstance(signal, type(np.array)):
            signal = np.array(signal)
        return 10 * np.log10(np.sum(signal**2))

    # TODO: Make a plot of λ vs SNR.
    # TODO: Make a plot of P_d vs SNR.

    # Calculate Eb
    signal: np.array = bpsk_signal()
    eb: float = avg_energy(signal)

    # Calculate N0 : # SNR = Eb/N0 -> N0 = Eb / SNR
    n0s: np.array = eb / snrs

    h0L: List[List[float]] = []
    h1L: List[List[float]] = []

    h0C: List[List[float]] = []
    h1C: List[List[float]] = []

    optimal_λs: List[float] = []
    with multiprocessing.Pool(8) as p, timeit("Rust") as _:
        for n0 in n0s:  # [:1]:  # TODO: FIXME: Handle all SNRs
            # Make the H0's
            make_awgn = functools.partial(awgn, np.zeros(len(signal)))
            make_signal = functools.partial(awgn, bpsk_signal())
            signals: List[np.array] = [
                awgn(bpsk_signal(), n0) for _ in range(NUM_ATTEMPTS)
            ]
            noises: List[np.array] = [
                awgn(np.zeros(len(signals[0])), n0) for _ in range(NUM_ATTEMPTS)
            ]

            # Calculate all λs.
            h0_λs: List[float] = p.map(max_cut_detector, noises)
            h1_λs: List[float] = p.map(max_cut_detector, signals)
            h0C.append(h0_λs)
            h1C.append(h1_λs)
            h0_λs: List[float] = p.map(energy_detector, noises)
            h1_λs: List[float] = p.map(energy_detector, signals)
            h0L.append(h0_λs)
            h1L.append(h1_λs)

            # """
            # Get optimal λ.
            # λ_best: float = get_best_lambda(h0_λs, h1_λs)
            # optimal_λs.append(λ_best)

    h0L: List[List[float]] = []
    h1L: List[List[float]] = []

    h0C: List[List[float]] = []
    h1C: List[List[float]] = []

    optimal_λs: List[float] = []
    with multiprocessing.Pool(8) as p, timeit("Python") as _:
        for n0 in n0s:  # [:1]:  # TODO: FIXME: Handle all SNRs
            # Make the H0's
            make_awgn = functools.partial(awgn, np.zeros(len(signal)))
            make_signal = functools.partial(awgn, bpsk_signal())
            signals: List[np.array] = [
                awgn_py(bpsk_signal_py(), n0) for _ in range(NUM_ATTEMPTS)
            ]
            noises: List[np.array] = [
                awgn_py(np.zeros(len(signals[0])), n0) for _ in range(NUM_ATTEMPTS)
            ]

            # Calculate all λs.
            h0_λs: List[float] = p.map(max_cut_detector_py, noises)
            h1_λs: List[float] = p.map(max_cut_detector_py, signals)
            h0C.append(h0_λs)
            h1C.append(h1_λs)
            h0_λs: List[float] = p.map(energy_detector_py, noises)
            h1_λs: List[float] = p.map(energy_detector_py, signals)
            h0L.append(h0_λs)
            h1L.append(h1_λs)

            # """
            # Get optimal λ.
            # λ_best: float = get_best_lambda(h0_λs, h1_λs)
            # optimal_λs.append(λ_best)

    # plot_multi_roc_curve(h0L, h1L, n0s)
    # plot_multi_roc_curve(h0C, h1C, n0s)

    # Plot of optimal λ vs SNR.
    # fig, ax = plt.subplots()
    # ax.plot(n0s, optimal_λs)
    # plt.show()
