#!/usr/bin/env python3
from komunikilo import (
    avg_energy,
    pure_awgn,
    awgn,
    # dcs_detector,
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
from util import db, undb

from dataclasses import dataclass, field
from functools import partial
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
from typing import Callable, List, Optional, Tuple, Union

logging.basicConfig(level=os.environ.get("LOGLEVEL", logging.INFO))
logger = logging.getLogger(__name__)


class Tx(Callable[[List[bool]], List[float]]): ...


class Detector(Callable[[List[float]], float]): ...


@dataclass
class DetectorTestResults:
    name: str
    detect_fn: Detector
    h0s: List[List[float]] = field(default_factory=lambda: [])
    h1s: List[List[float]] = field(default_factory=lambda: [])
    lr: Optional[LogisticRegression] = None


def _detectors() -> List[DetectorTestResults]:
    return [
        DetectorTestResults(name="MaxCut", detect_fn=max_cut_detector),
        # DetectorTestResults(name="DCS", detect_fn=dcs_detector),
        DetectorTestResults(name="Radiometer", detect_fn=energy_detector),
    ]


class ModulationTestResults:
    name: str
    tx_func: Tx
    detectors: List[DetectorTestResults]

    def __init__(self, name: str, tx_func: Tx):
        self.name = name
        self.tx_func = tx_func
        self.detectors = _detectors()


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
    if len(tpr) == len(h0_λs):
        df_test["tpr"] = tpr[::-1]
    else:
        df_test["tpr"] = tpr[1:][::-1]
    if len(fpr) == len(h0_λs):
        df_test["fpr"] = fpr[::-1]
    else:
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
    for lr in lrs:
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
    # lambdas = [__best_threshold(lr) for lr in lrs]
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
    detector_results: List[DetectorTestResults],
    snrs: List[float],
    title: str = "",
):
    fig, ax = plt.subplots(2, 1)

    for detector in detector_results:
        aucs: List[float] = [__auc(lr) for lr in detector.lr]
        y_js: List[float] = [__youden_j(lr) for lr in detector.lr]

        ax[0].plot(db(snrs), y_js, label=detector.name)
        ax[1].plot(db(snrs), aucs, label=detector.name)
    ax[0].set_xlabel("SNR (db)")
    ax[0].set_ylabel("Youden J")
    ax[0].legend(loc="best")
    ax[1].set_xlabel("SNR (db)")
    ax[1].set_ylabel("AUC")
    ax[1].legend(loc="best")
    fig.suptitle(title)


def plot_youden_j_with_multiple_modulations(
    modulation_test_results: List[ModulationTestResults],
    key: str,
):
    fig, ax = plt.subplots()
    for modulation in modulation_test_results:
        detector = next(d for d in modulation.detectors if d.name == key)
        youden_js: List[float] = [__youden_j(lr) for lr in detector.lr]
        ax.plot(db(snrs), youden_js, label=modulation.name)

    ax.set_xlabel("SNR (db)")
    ax.set_ylabel("Youden J")
    ax.legend(loc="best")
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


def __run(
    modulation: ModulationTestResults,
    snrs: List[float],
    num_attempts: int = 200,
    progress: Optional[tqdm] = None,
) -> ModulationTestResults:
    NUM_BITS: int = 120

    # Calculate Eb
    signal: np.array = modulation.tx_func(random_data(NUM_BITS))
    while len(signal) < 4096 + 64:
        NUM_BITS += 10
        signal: np.array = modulation.tx_func(random_data(NUM_BITS))
    logger.debug(f"NUM_BITS is {NUM_BITS}")

    eb: float = avg_energy(signal)
    # Calculate N0 : # SNR = Eb/N0 -> N0 = Eb / SNR
    n0s: np.array = eb / snrs
    logger.debug("EB: {eb}")
    logger.debug("N0s: {list(n0s)}")

    if not progress:
        progress = tqdm(total=len(n0s))

    with multiprocessing.Pool(8) as p:
        for i, n0 in enumerate(n0s):
            # Make the H0's
            logger.debug(f"N0: {n0}")
            signals: List[np.array] = [
                awgn(modulation.tx_func(random_data(NUM_BITS)), n0)
                for _ in range(num_attempts)
            ]
            noises: List[np.array] = [
                pure_awgn(len(signals[0]), n0) for _ in range(num_attempts)
            ]

            for detector in modulation.detectors:
                # Calculate all λs.
                h0_λs: List[float] = p.map(detector.detect_fn, noises)
                h1_λs: List[float] = p.map(detector.detect_fn, signals)
                detector.h0s.append(h0_λs)
                detector.h1s.append(h1_λs)

            # Update progress meter.
            progress.update(1)

        # Perform Logistic Regressions.
        for detector in modulation.detectors:
            lrs: List[pd.DataFrame] = p.starmap(
                __log_regress, zip(detector.h0s, detector.h1s)
            )
            detector.lr = lrs

        # Create plots.
        for detector in modulation.detectors:
            plot_multi_roc_curve(
                detector.lr, label=f"{modulation.name}: {detector.name}"
            )

        plot_multi_youden_j_and_auc_vs_snr(
            detector_results=modulation.detectors,
            snrs=snrs,
            title=modulation.name,
        )

    return modulation


SAMPLE_RATE: float = 2**16  # 65536
SYMBOL_RATE: float = 2**8  # 256
CARRIER_FREQ: float = 2000

bpsk_signal = partial(
    tx_bpsk, sample_rate=SAMPLE_RATE, symbol_rate=SYMBOL_RATE, carrier_freq=CARRIER_FREQ
)
cdma_signal = partial(
    tx_cdma_bpsk,
    sample_rate=SAMPLE_RATE,
    symbol_rate=SYMBOL_RATE,
    carrier_freq=CARRIER_FREQ,
)
bfsk_signal = partial(
    tx_bfsk,
    sample_rate=SAMPLE_RATE,
    symbol_rate=SYMBOL_RATE,
    freq_low=CARRIER_FREQ * 0.8,
    freq_high=CARRIER_FREQ,
)
csk_signal = partial(tx_csk, sample_rate=SAMPLE_RATE, symbol_rate=SYMBOL_RATE)
dcsk_signal = partial(tx_dcsk, sample_rate=SAMPLE_RATE, symbol_rate=SYMBOL_RATE)
fh_css_signal = partial(
    tx_fh_css,
    sample_rate=SAMPLE_RATE,
    symbol_rate=SYMBOL_RATE,
    freq_low=2e3,
    freq_high=1e4,
    num_freqs=8,
)
fh_ofdm_dcsk_signal = partial(
    tx_fh_ofdm_dcsk,
    sample_rate=SAMPLE_RATE,
    symbol_rate=SYMBOL_RATE,
    carrier_freq=CARRIER_FREQ,
)


def ofdm_signal(data: List[bool]) -> np.array:
    subcarriers = 8
    pilots = int(subcarriers * 0.8)
    # subcarriers = 16
    # pilots = 4
    return tx_ofdm(data, subcarriers, pilots, SAMPLE_RATE, SYMBOL_RATE, CARRIER_FREQ)


harness: List[ModulationTestResults] = [
    ModulationTestResults(name="BFSK", tx_func=bfsk_signal),
    ModulationTestResults(name="BPSK", tx_func=bpsk_signal),
    ModulationTestResults(name="CDMA", tx_func=cdma_signal),
    ModulationTestResults(name="OFDM", tx_func=ofdm_signal),
    ModulationTestResults(name="FH-CSS", tx_func=fh_css_signal),
    ModulationTestResults(name="CSK", tx_func=csk_signal),
    ModulationTestResults(name="DCSK", tx_func=dcsk_signal),
    ModulationTestResults(name="FH-OFDM-DCSK", tx_func=fh_ofdm_dcsk_signal),
]


if __name__ == "__main__":

    # NUM_ATTEMPTS: int = 2000
    NUM_ATTEMPTS: int = 150
    # snrs: np.array = undb(np.linspace(-75, 0, 150))
    snrs: np.array = undb(np.linspace(-25, -6, 100))
    progress = tqdm(total=len(snrs) * len(harness))
    results: List[ModulationTestResults] = [
        __run(
            modulation,
            snrs,
            num_attempts=NUM_ATTEMPTS,
            progress=progress,
        )
        for modulation in harness
    ]

    for d in _detectors():
        plot_youden_j_with_multiple_modulations(results, d.name)

    plt.show()
