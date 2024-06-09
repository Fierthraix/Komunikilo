#!/usr/bin/env python3
from komunikilo import (
    awgn,
    energy_detector,
    max_cut_detector,
    random_data,
    tx_bfsk,
    tx_bpsk,
    tx_cdma_bpsk,
    tx_fh_css,
    tx_ofdm,
    tx_csk,
    tx_dcsk,
    tx_fh_ofdm_dcsk,
)
from util import timeit, db, undb
from willie import avg_energy

from dataclasses import dataclass, field
import itertools
import logging
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import os
import pandas as pd
from scipy.stats import rv_histogram
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from tqdm import tqdm
from typing import Callable, Dict, List, Optional, Tuple, Union

logging.basicConfig(level=os.environ.get("LOGLEVEL", logging.INFO))
logger = logging.getLogger(__name__)


class Tx(Callable[[List[bool]], List[float]]): ...


class Detector(Callable[[List[float]], float]): ...


@dataclass
class DetectorTestResults:
    detect_fn: Detector
    snrs: List[float] = field(default_factory=lambda: [])
    h0s: List[List[float]] = field(default_factory=lambda: [])
    h1s: List[List[float]] = field(default_factory=lambda: [])
    lr: Optional[LogisticRegression] = None


def __log_regress(
    h0_λs: np.array, h1_λs: np.array, return_lr: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, LogisticRegression]]:
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
    if return_lr:
        return df_test, log_regression
    return df_test


def __auc(lr: pd.DataFrame) -> float:
    auc = metrics.roc_auc_score(lr["y"], lr["proba"])
    return auc


def __youden_j(lr: pd.DataFrame) -> float:
    return max(abs(lr["youden_j"]))


def plot_multi_roc_curve(lrs: List[pd.DataFrame], label=""):
    fig, ax = plt.subplots()
    for lr in lrs:  # [lrs[5]]: #, lrs[-5]):
        metrics.RocCurveDisplay(fpr=lr.fpr, tpr=lr.tpr).plot(ax=ax)
    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")
    ax.set_title("ROC Curve")
    fig.suptitle(label)


def plot_auc_vs_snr(h0_λs: List[np.array], h1_λs: List[np.array], snrs: List[float]):
    aucs: List[float] = [__auc(h0_λ, h1_λ) for h0_λ, h1_λ in zip(h0_λs, h1_λs)]
    fig, ax = plt.subplots()
    ax.plot(db(snrs), aucs)
    ax.set_xlabel("SNR (db)")
    ax.set_ylabel("AUC")
    ax.set_title("AUC vs SNR Curve")


def plot_youden_j_and_auc_vs_snr(lrs: List[pd.DataFrame], snrs: List[float]):
    lambdas = [__best_threshold(lr) for lr in lrs]
    aucs = [__auc(lr) for lr in lrs]
    youden_js = [__youden_j(lr) for lr in lrs]

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(db(snrs), youden_js)
    ax[0].set_xlabel("SNR (db)")
    ax[0].set_ylabel("Youden J")
    ax[1].plot(db(snrs), aucs)
    ax[1].set_xlabel("SNR (db)")
    ax[1].set_ylabel("AUC")


def plot_multi_youden_j_and_auc_vs_snr(
    labels: List[str],
    multi_lrs: List[List[pd.DataFrame]],
    snrs: List[float],
    title: str = "",
):
    aucs: List[List[float]] = [[__auc(lr) for lr in lrs] for lrs in multi_lrs]
    youden_js: List[List[float]] = [[__youden_j(lr) for lr in lrs] for lrs in multi_lrs]

    fig, ax = plt.subplots(2, 1)
    for y_j, auc, label in zip(youden_js, aucs, labels):
        ax[0].plot(db(snrs), y_j, label=label)
        ax[1].plot(db(snrs), auc, label=label)
    ax[0].set_xlabel("SNR (db)")
    ax[0].set_ylabel("Youden J")
    ax[0].legend(loc="best")
    ax[1].set_xlabel("SNR (db)")
    ax[1].set_ylabel("AUC")
    ax[1].legend(loc="best")
    fig.suptitle(title)


def plot_youden_j_with_multiple_modulations(
    modulation_test_results: Dict[str, List[Dict[str, DetectorTestResults]]],
    key: str,
):
    # [{"MaxCut": DetectorTestResults.lr: pd.DataFrame}, "NRG": DetectorTestResults.lr: pd.DataFrame]
    # aucs: List[List[float]] = [[__auc(lr) for lr in lrs] for lrs in multi_lrs]
    youden_js: Dict[str, List[float]] = {
        label: [__youden_j(lr) for lr in lrs[key].lr]
        for label, lrs in modulation_test_results.items()
    }

    # fig, ax = plt.subplots(2, 1)
    fig, ax = plt.subplots()
    for label, y_j in youden_js.items():
        ax.plot(db(snrs), y_j, label=label)
        # ax[1].plot(db(snrs), auc, label=label)
    ax.set_xlabel("SNR (db)")
    ax.set_ylabel("Youden J")
    ax.legend(loc="best")
    # ax[1].set_xlabel("SNR (db)")
    # ax[1].set_ylabel("AUC")
    # ax[1].legend(loc="best")
    fig.suptitle(key)


def plot_best_lambda(h0_λs: np.array, h1_λs: np.array, lr=None):
    # Get cutoff
    if not lr:
        lr = __log_regress(h0_λs, h1_λs)
    cut_off = __best_threshold(lr)

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


def __best_threshold(lr: pd.DataFrame) -> float:
    cut_off = lr.sort_values(by="youden_j", ascending=False, ignore_index=True).iloc[0]
    return cut_off.x


def curry_lr(ht) -> pd.DataFrame:
    h0_λs, h1_λs = ht
    return __log_regress(h0_λs, h1_λs)


def __run(
    tx: Tx,
    snrs: List[float],
    label="",
    num_attempts: int = 200,
    progress: Optional[tqdm] = None,
) -> Dict[str, DetectorTestResults]:
    NUM_BITS: int = 120

    # Calculate Eb
    signal: np.array = tx(random_data(NUM_BITS))
    while len(signal) < 4096 + 64:
        NUM_BITS += 10
        signal: np.array = tx(random_data(NUM_BITS))
    logger.debug(f"NUM_BITS is {NUM_BITS}")

    eb: float = avg_energy(signal)
    # Calculate N0 : # SNR = Eb/N0 -> N0 = Eb / SNR
    n0s: np.array = eb / snrs
    logger.debug("EB: {eb}")
    logger.debug("N0s: {list(n0s)}")

    max_cut = DetectorTestResults(detect_fn=max_cut_detector, snrs=snrs)
    nrg = DetectorTestResults(detect_fn=energy_detector, snrs=snrs)
    # dcs_cut = DetectorTestResults

    if not progress:
        progress = tqdm(total=len(n0s))

    with multiprocessing.Pool(8) as p:
        # for i, n0 in progress(enumerate(n0s), desc=label or "Detectors"):
        for i, n0 in enumerate(n0s):
            # Make the H0's
            logger.debug(f"N0: {n0}")
            signals: List[np.array] = [
                awgn(tx(random_data(NUM_BITS)), n0) for _ in range(num_attempts)
            ]
            noises: List[np.array] = [
                awgn(np.zeros(len(signals[0])), n0) for _ in range(num_attempts)
            ]

            # Calculate all λs.
            h0_λs: List[float] = p.map(max_cut.detect_fn, noises)
            h1_λs: List[float] = p.map(max_cut.detect_fn, signals)
            max_cut.h0s.append(h0_λs)
            max_cut.h1s.append(h1_λs)

            h0_λs: List[float] = p.map(energy_detector, noises)
            h1_λs: List[float] = p.map(energy_detector, signals)
            nrg.h0s.append(h0_λs)
            nrg.h1s.append(h1_λs)
            progress.update(1)

    with multiprocessing.Pool(8) as p, timeit("Logistic Regressions") as _:
        lrs_max_cut: List[pd.DataFrame] = p.map(curry_lr, zip(max_cut.h0s, max_cut.h1s))
        lrs_nrg: List[pd.DataFrame] = p.map(curry_lr, zip(nrg.h0s, nrg.h1s))

    with timeit("Plots") as _:
        plot_multi_roc_curve(lrs_max_cut)
        plot_multi_roc_curve(lrs_nrg)
        plot_multi_youden_j_and_auc_vs_snr(
            labels=["Energy Detector", "Max Cut"],
            multi_lrs=[lrs_nrg, lrs_max_cut],
            snrs=snrs,
            title=label,
        )

    max_cut.lr = lrs_max_cut
    nrg.lr = lrs_nrg

    return {"Radiometer": nrg, "MaxCut": max_cut}


if __name__ == "__main__":
    SAMPLE_RATE: float = 2**16  # 65536
    SYMBOL_RATE: float = 2**8  # 256
    CARRIER_FREQ: float = 2000

    def ofdm_signal(data: List[bool]) -> np.array:
        return tx_ofdm(data, SAMPLE_RATE, SYMBOL_RATE, CARRIER_FREQ)

    def bpsk_signal(data: List[bool]) -> np.array:
        return tx_bpsk(data, SAMPLE_RATE, SYMBOL_RATE, CARRIER_FREQ)

    def cdma_signal(data: List[bool]) -> np.array:
        return tx_cdma_bpsk(data, SAMPLE_RATE, SYMBOL_RATE, CARRIER_FREQ)

    def bfsk_signal(data: List[bool]) -> np.array:
        return tx_bfsk(data, SAMPLE_RATE, SYMBOL_RATE, CARRIER_FREQ * 0.8, CARRIER_FREQ)

    def fh_css_signal(data: List[bool]) -> np.array:
        f_low = 2e3
        f_high = 1e4
        num_freqs = 8
        return tx_fh_css(data, SAMPLE_RATE, SYMBOL_RATE, f_low, f_high, num_freqs)

    def csk_signal(data: List[bool]) -> np.array:
        return tx_csk(data, SAMPLE_RATE, SYMBOL_RATE)

    def dcsk_signal(data: List[bool]) -> np.array:
        return tx_dcsk(data, SAMPLE_RATE, SYMBOL_RATE)

    def fh_ofdm_dcsk_signal(data: List[bool]) -> np.array:
        return tx_fh_ofdm_dcsk(data, SAMPLE_RATE, SYMBOL_RATE, CARRIER_FREQ)

    harness: Dict[str, Callable] = {
        "BFSK": bfsk_signal,
        "BPSK": bpsk_signal,
        "CDMA": cdma_signal,
        "OFDM": ofdm_signal,
        "FH-CSS": fh_css_signal,
        "CSK": csk_signal,
        "DCSK": dcsk_signal,
        # "FH-OFDM-DCSK": fh_ofdm_dcsk_signal,
    }

    NUM_ATTEMPTS: int = 2000
    # NUM_ATTEMPTS: int = 50
    # snrs: np.array = undb(np.linspace(-75, -3, 50))
    # snrs: np.array = undb(np.linspace(-75, 0, 150))
    # snrs: np.array = undb(np.linspace(-30, -3, 150))
    snrs: np.array = undb(np.linspace(-20, -3, 100))
    progress = tqdm(total=len(snrs) * len(harness))
    results = {
        key: __run(func, snrs, label=key, num_attempts=NUM_ATTEMPTS, progress=progress)
        for key, func in harness.items()
    }

    plot_youden_j_with_multiple_modulations(results, "MaxCut")
    plot_youden_j_with_multiple_modulations(results, "Radiometer")

    plt.show()
