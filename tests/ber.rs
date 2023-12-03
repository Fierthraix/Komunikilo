use komunikilo::bpsk::{
    rx_baseband_bpsk_signal, rx_bpsk_signal, tx_baseband_bpsk_signal, tx_bpsk_signal,
};
use komunikilo::cdma::{rx_cdma_bpsk_signal, tx_cdma_bpsk_signal};
use komunikilo::hadamard::HadamardMatrix;
use komunikilo::qpsk::{
    rx_baseband_qpsk_signal, rx_qpsk_signal, tx_baseband_qpsk_signal, tx_qpsk_signal,
};
use komunikilo::{awgn_complex, erfc, linspace, Bit};
use num::complex::Complex;
use plotpy::{Curve, Plot};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

use util::not_inf;

#[macro_use]
mod util;

fn ber_bpsk(eb_no: f64) -> f64 {
    0.5 * erfc(eb_no.sqrt())
}

fn ber_qpsk(eb_no: f64) -> f64 {
    0.5 * erfc(eb_no.sqrt()) - 0.25 * erfc(eb_no.sqrt()).powi(2)
}

#[test]
fn baseband_bpsk_works() {
    // Make some data.
    let mut rng = rand::thread_rng();
    let num_bits = 9001;
    // let num_bits = 1_000_000;
    // let num_bits = 100000;
    let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

    // Transmit the signal.
    let bpsk_tx: Vec<Complex<f64>> = tx_baseband_bpsk_signal(data_bits.iter().cloned()).collect();

    // An x-axis for plotting Eb/N0.
    let xmin = f64::MIN_POSITIVE;
    let xmax = 15f64;
    let x: Vec<f64> = linspace(xmin, xmax, 100).collect();

    // Container for the Eb/N0.
    let y: Vec<f64> = x
        .par_iter()
        .map(|&i| {
            let sigma = (1f64 / (2f64 * i as f64)).sqrt();
            let noisy_signal = awgn_complex(bpsk_tx.iter().cloned(), sigma);
            let rx = rx_baseband_bpsk_signal(noisy_signal);

            rx.zip(data_bits.iter())
                .map(|(rx, &tx)| if rx == tx { 0f64 } else { 1f64 })
                .sum::<f64>()
                / num_bits as f64
        })
        .collect();

    let y_theory: Vec<f64> = x.iter().map(|&x| ber_bpsk(x)).collect();

    ber_plot!(x, y, y_theory, "/tmp/ber_baseband_bpsk.png");

    let bpsk_rx: Vec<Bit> = rx_baseband_bpsk_signal(bpsk_tx.iter().cloned()).collect();
    assert_eq!(data_bits, bpsk_rx);
}

#[test]
fn baseband_qpsk_works() {
    // Make some data.
    let mut rng = rand::thread_rng();
    let num_bits = 9002;
    // let num_bits = 100_000_000;
    let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

    // Transmit the signal.
    let qpsk_tx: Vec<Complex<f64>> = tx_baseband_qpsk_signal(data_bits.iter().cloned()).collect();

    // An x-axis for plotting Eb/N0.
    let xmin = f64::MIN_POSITIVE;
    let xmax = 15f64;
    let x: Vec<f64> = linspace(xmin, xmax, 100).collect();

    // Container for the Eb/N0.
    let y: Vec<f64> = x
        .par_iter()
        .map(|&i| {
            let sigma = (1f64 / (i as f64)).sqrt();
            // let noisy_signal = awgn_complex(qpsk_tx.iter().cloned(), sigma);
            let noisy_signal = awgn_complex(qpsk_tx.iter().cloned(), sigma);
            let rx = rx_baseband_qpsk_signal(noisy_signal);

            rx.zip(data_bits.iter())
                .map(|(rx, &tx)| if rx == tx { 0f64 } else { 1f64 })
                .sum::<f64>()
                / num_bits as f64
        })
        .collect();

    let y_theory: Vec<f64> = x.iter().map(|&x| ber_qpsk(x)).collect();

    ber_plot!(x, y, y_theory, "/tmp/ber_baseband_qpsk.png");

    let qpsk_rx: Vec<Bit> = rx_baseband_qpsk_signal(qpsk_tx.iter().cloned()).collect();
    assert_eq!(data_bits, qpsk_rx);
}

#[test]
fn bpsk_works() {
    // Simulation parameters.
    let num_bits = 4_000; //1_000_000; //4000; // How many bits to transmit overall.
    let samp_rate = 44100; // Clock rate for both RX and TX.
    let symbol_rate = 900; // Rate symbols come out the things.
    let carrier_freq = 1800_f64;
    let n_scale = samp_rate as f64 / carrier_freq as f64;

    // Test data.
    let mut rng = rand::thread_rng();
    let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

    // An x-axis for plotting Eb/N0.
    let xmin = f64::MIN_POSITIVE;
    let xmax = 15f64;
    let x: Vec<f64> = linspace(xmin, xmax, 100).collect();

    // Tx output.
    let bpsk_tx: Vec<f64> = tx_bpsk_signal(
        data_bits.iter().cloned(),
        samp_rate,
        symbol_rate,
        carrier_freq,
    )
    .collect();

    // Container for the Eb/N0.
    let y: Vec<f64> = x
        .par_iter()
        // .iter()
        .map(|&i| {
            let sigma = not_inf((1f64 / (2f64 * i)).sqrt());
            let awgn_noise = Normal::new(0f64, sigma).unwrap();

            let konst = n_scale.sqrt();
            let noisy_signal = bpsk_tx
                .iter()
                .cloned()
                .zip(awgn_noise.sample_iter(rand::thread_rng()))
                .map(|(symb, noise)| symb + noise * konst);

            let rx = rx_bpsk_signal(noisy_signal, samp_rate, symbol_rate, carrier_freq);

            rx.zip(data_bits.iter())
                .map(|(rx, &tx)| if rx == tx { 0f64 } else { 1f64 })
                .sum::<f64>()
                / num_bits as f64
        })
        .collect();

    let y_theory: Vec<f64> = x.iter().cloned().map(ber_bpsk).collect();

    ber_plot!(x, y, y_theory, "/tmp/ber_bpsk.png");
}

#[test]
fn qpsk_works() {
    // Simulation parameters.
    let num_bits = 4_000; //1_000_000; //4000; // How many bits to transmit overall.
    let samp_rate = 44100; // Clock rate for both RX and TX.
    let symbol_rate = 800; // Rate symbols come out the things.
    let carrier_freq = 2000f64;
    let n_scale = samp_rate as f64 / carrier_freq as f64;

    // Test data.
    let mut rng = rand::thread_rng();
    let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

    // An x-axis for plotting Eb/N0.
    let xmin = f64::MIN_POSITIVE;
    let xmax = 15f64;
    let x: Vec<f64> = linspace(xmin, xmax, 50).collect();

    // Tx output.
    let qpsk_tx: Vec<f64> = tx_qpsk_signal(
        data_bits.iter().cloned(),
        samp_rate,
        symbol_rate,
        carrier_freq,
    )
    .collect();

    // Container for the Eb/N0.
    let y: Vec<f64> = x
        .par_iter()
        .map(|&i| {
            let konst = n_scale.sqrt();
            let sigma = (1f64 / (2f64 * i)).sqrt();

            let normal = Normal::new(0f64, sigma).unwrap();
            let noisy_signal = qpsk_tx
                .iter()
                .cloned()
                .zip(normal.sample_iter(rand::thread_rng()))
                .map(|(symb, noise)| symb + noise * konst);
            let rx = rx_qpsk_signal(noisy_signal, samp_rate, symbol_rate, carrier_freq);

            rx.zip(data_bits.iter())
                .map(|(rx, &tx)| if rx == tx { 0f64 } else { 1f64 })
                .sum::<f64>()
                / num_bits as f64
        })
        .collect();

    let y_theory: Vec<f64> = x.iter().map(|&x| ber_qpsk(x)).collect();

    ber_plot!(x, y, y_theory, "/tmp/ber_qpsk.png");
    plot!(x, y, y_theory, "/tmp/ber_qpsk_unlog.png");
    // plot2!(x, y, y_theory, "/tmp/ber_qpsk_unlog2.png");
}

#[test]
fn cdma_works() {
    // Simulation parameters.
    let num_bits = 4_000; //1_000_000; //4000; // How many bits to transmit overall.
    let samp_rate = 80_000; // Clock rate for both RX and TX.
    let symbol_rate = 1000; // Rate symbols come out the things.
    let carrier_freq = 2500f64;
    let n_scale = samp_rate as f64 / carrier_freq as f64;

    // Test data.
    let mut rng = rand::thread_rng();
    let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();
    let h = HadamardMatrix::new(8);
    let key = h.key(2);

    // An x-axis for plotting Eb/N0.
    let xmin = f64::MIN_POSITIVE;
    let xmax = 15f64;
    let x: Vec<f64> = linspace(xmin, xmax, 50).collect();

    // Tx output.
    let cdma_tx: Vec<f64> = tx_cdma_bpsk_signal(
        data_bits.iter().cloned(),
        samp_rate,
        symbol_rate,
        carrier_freq,
        key,
    )
    .collect();

    // Container for the Eb/N0.
    let y: Vec<f64> = x
        .par_iter()
        .map(|&i| {
            let konst = n_scale.sqrt();
            let sigma = (1f64 / (2f64 * i)).sqrt();

            let normal = Normal::new(0f64, sigma).unwrap();
            let noisy_signal = cdma_tx
                .iter()
                .cloned()
                .zip(normal.sample_iter(rand::thread_rng()))
                .map(|(symb, noise)| symb + noise * konst);
            let rx = rx_cdma_bpsk_signal(noisy_signal, samp_rate, symbol_rate, carrier_freq, key);

            rx.zip(data_bits.iter())
                .map(|(rx, &tx)| if rx == tx { 0f64 } else { 1f64 })
                .sum::<f64>()
                / num_bits as f64
        })
        .collect();

    let y_theory: Vec<f64> = x.iter().map(|&x| ber_qpsk(x)).collect();

    ber_plot!(x, y, y_theory, "/tmp/ber_cdma.png");
    plot!(x, y, y_theory, "/tmp/ber_cdma_unlog.png");
}
