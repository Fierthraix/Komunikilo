#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import rv_histogram
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

if __name__ == "__main__":
    binification = 128
    # Get Points
    num_iters: int = 9001

    # These values represent the output statistic
    # of the detector in the H0 and H1 scenarios.

    μ0: float = 86
    σ0: float = 2
    h0_samps: np.ndarray = np.random.normal(μ0, σ0, num_iters)

    h0 = rv_histogram(np.histogram(h0_samps, bins=binification))

    μ1: float = 91
    σ1: float = 2.5
    h1_samps: np.ndarray = np.random.normal(μ1, σ1, num_iters)

    h1 = rv_histogram(np.histogram(h1_samps, bins=binification))

    x = np.linspace(75, 100, binification)
    h0_pdf = h0.pdf(x)
    h1_pdf = h1.pdf(x)

    x_var = np.concatenate((h0_samps, h1_samps)).reshape(-1, 1)
    y_var = np.concatenate((np.zeros(len(h0_samps)), np.ones(len(h1_samps))))
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

    df = pd.DataFrame(
        {
            "x": np.concatenate((h0_samps, h1_samps)),
            "y": np.concatenate((np.zeros(len(h0_samps)), np.ones(len(h1_samps)))),
        }
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
    # add reversed TPR and FPR
    df_test["tpr"] = tpr[1:][::-1]
    df_test["fpr"] = fpr[1:][::-1]
    # add thresholds to check
    df_test["thresholds"] = thresholds[1:][::-1]
    # add Youden's j index
    df_test["youden_j"] = df_test.tpr - df_test.fpr

    cut_off = df_test.sort_values(
        by="youden_j", ascending=False, ignore_index=True
    ).iloc[0]

    fig, ax = plt.subplots()
    ax.plot(x, h0_pdf)
    ax.plot(x, h1_pdf)
    ax.axvline(cut_off.x, color="k", ls="--")

    fig, ax = plt.subplots()
    metrics.RocCurveDisplay(fpr=df_test.fpr, tpr=df_test.tpr, roc_auc=auc).plot(ax=ax)
    ax.set_title("ROC Curve")
    ax.axline(xy1=(0, 0), slope=1, color="r", ls=":")
    ax.plot(
        cut_off.fpr, cut_off.tpr, "ko", ms=10, label=f"Best Threshold = {cut_off.x}"
    )
    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")
    ax.legend(loc=4)

    plt.show()
