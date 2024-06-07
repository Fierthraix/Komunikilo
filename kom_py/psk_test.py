#!/usr/bin/env python3
import pytest
import random
from typing import List

from psk import (
    tx_baseband_bpsk,
    tx_baseband_qpsk,
    tx_bpsk,
    tx_qpsk,
    rx_baseband_bpsk,
    rx_baseband_qpsk,
    rx_bpsk,
    rx_qpsk,
)

SAMPLE_RATE = 44100
SYMBOL_RATE = 900
CARRIER_FREQ = 1800
NUM_BITS =  90 #9002


def rand_data(num_bits: int) -> List[bool]:
    return [random.choice((True, False)) for _ in range(num_bits)]


def test_baseband_bpsk():
    data: List[bool] = rand_data(NUM_BITS)
    tx: List[complex] = tx_baseband_bpsk(data)
    rx: List[bool] = rx_baseband_bpsk(data)

    assert data == rx


def test_bpsk():
    data: List[bool] = rand_data(NUM_BITS)
    tx: List[float] = tx_bpsk(data, SAMPLE_RATE, SYMBOL_RATE, CARRIER_FREQ)
    rx: List[bool] = rx_bpsk(data, SAMPLE_RATE, SYMBOL_RATE, CARRIER_FREQ)

    assert len(data) == len(rx)
    assert data == rx
