use comms::bpsk::{
    rx_baseband_bpsk_signal, rx_bpsk_signal, tx_baseband_bpsk_signal, tx_bpsk_signal,
};
use comms::qpsk::{rx_baseband_qpsk_signal, tx_baseband_qpsk_signal};
use comms::{awgn, awgn_complex, erfc, linspace, Bit};
use num::complex::Complex;
use plotpy::{Curve, Plot};
use rand::Rng;
use rayon::prelude::*;

#[macro_use]
mod util;

fn ber_bpsk(eb_no: f64) -> f64 {
    0.5 * erfc(eb_no.sqrt())
}

fn ber_qpsk(eb_no: f64) -> f64 {
    0.5 * erfc(eb_no.sqrt()) - 0.25 * erfc(eb_no.sqrt()).powi(2)
}

#[test]
fn a_graph() {
    let xmin = 0f64;
    let xmax = 1f64;
    let x: Vec<f64> = linspace(xmin, xmax, 100).collect();
    let y1: Vec<f64> = x.iter().map(|&x| ber_bpsk(x)).collect();
    let y2: Vec<f64> = x.iter().map(|&x| erfc(x)).collect();

    plot!(x, y1, y2, "/tmp/asdf.svg");
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

    plot!(x, y, y_theory, "/tmp/ber_baseband_qpsk.png");

    let qpsk_rx: Vec<Bit> = rx_baseband_qpsk_signal(qpsk_tx.iter().cloned()).collect();
    assert_eq!(data_bits, qpsk_rx);
}

#[test]
fn basic_bpsk_works() {
    // Simulation parameters.
    let num_bits = 4000; //1_000_000; //4000; // How many bits to transmit overall.
    let samp_rate = 44100; // Clock rate for both RX and TX.
    let symbol_rate = 900; // Rate symbols come out the things.
    let carrier_freq = 1800_f64;

    // Test data.
    let mut rng = rand::thread_rng();
    let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

    // An x-axis for plotting Eb/N0.
    // let xmin = f64::MIN_POSITIVE;
    // let xmax = 15f64;

    // Tx output.
    let bpsk_tx: Vec<f64> = tx_bpsk_signal(
        data_bits.iter().cloned(),
        samp_rate,
        symbol_rate,
        carrier_freq,
        0_f64,
    )
    .collect();

    // Check if the rx is off on the data bits by some constant amount.

    let eb_n0 = 0.01;
    let sigma = (1f64 / (2f64 * eb_n0)).sqrt();
    // let noisy_signal = awgn(bpsk_tx.iter().cloned(), sigma);
    let noisy_signal = awgn(bpsk_tx.iter().cloned(), eb_n0);

    let rx: Vec<Bit> = rx_bpsk_signal(
        // bpsk_tx.iter().cloned(),
        noisy_signal,
        samp_rate,
        symbol_rate,
        carrier_freq,
        0_f64,
    )
    .collect();

    // assert_eq!(data_bits, rx);
    let percent_correct: f64 = rx
        .iter()
        .zip(data_bits.iter())
        .map(|(&rx, &tx)| if rx == tx { 1f64 } else { 0f64 })
        .sum::<f64>()
        / rx.len() as f64;
    println!("Correct RX: {}%", percent_correct * 100f64);
    println!("Eb/N0: {} || BER {}", eb_n0, 1f64 - percent_correct);
    println!("Theory BER: {} || {}", ber_bpsk(eb_n0), ber_bpsk(sigma));
    println!(
        "tx ({}): {:?}",
        data_bits.len(),
        data_bits.iter().take(20).collect::<Vec<_>>()
    );
    println!(
        "rx ({}): {:?}",
        rx.len(),
        rx.iter().take(20).collect::<Vec<_>>()
    );
    // assert!(false);
}

#[test]
fn bpsk_works() {
    // Simulation parameters.
    let num_bits = 4000; //1_000_000; //4000; // How many bits to transmit overall.
    let samp_rate = 44100; // Clock rate for both RX and TX.
    let symbol_rate = 900; // Rate symbols come out the things.
    let carrier_freq = 1800_f64;

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
        0_f64,
    )
    .collect();

    // Container for the Eb/N0.
    let y: Vec<f64> = x
        .par_iter()
        .map(|&i| {
            // let sigma = (7.95f64 / (2f64 * i as f64)).sqrt();
            // let sigma = (1f64 / (2f64 * i as f64)).sqrt();
            let sigma = 7f64 / (2f64 * i);
            let noisy_signal = awgn(bpsk_tx.iter().cloned(), sigma);
            let rx = rx_bpsk_signal(noisy_signal, samp_rate, symbol_rate, carrier_freq, 0_f64);

            rx.zip(data_bits.iter())
                .map(|(rx, &tx)| if rx == tx { 0f64 } else { 1f64 })
                .sum::<f64>()
                / num_bits as f64
        })
        .collect();

    let y_theory: Vec<f64> = x.iter().map(|&x| ber_bpsk(x)).collect();

    let mut curve_practice = Curve::new();
    curve_practice.draw(&x, &y);
    let mut curve_theory = Curve::new();
    curve_theory.draw(&x, &y_theory);

    ber_plot!(x, y, y_theory, "/tmp/ber_bpsk.png");
}
