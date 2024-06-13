use komunikilo::bpsk::{rx_bpsk_signal, tx_bpsk_signal};
use komunikilo::iter::Iter;
use komunikilo::{awgn, Bit};
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use std::f64::consts::PI;
use welch_sde::{Build, PowerSpectrum, SpectralDensity};

#[macro_use]
mod util;

#[test]
fn bpsk_graphs() {
    let data: Vec<Bit> = vec![
        false, false, true, false, false, true, false, true, true, true,
    ];

    // Simulation parameters.
    let samp_rate = 44100; // Clock rate for both RX and TX.
    let symbol_rate = 900; // Rate symbols come out the things.
    let carrier_freq = 1800_f64;

    let tx: Vec<f64> =
        tx_bpsk_signal(data.iter().cloned(), samp_rate, symbol_rate, carrier_freq).collect();

    let rx_clean: Vec<Bit> =
        rx_bpsk_signal(tx.iter().cloned(), samp_rate, symbol_rate, carrier_freq).collect();

    let sigma = 2f64;
    let noisy_signal: Vec<f64> = awgn(tx.iter().cloned(), sigma).collect();

    let rx_dirty: Vec<Bit> = rx_bpsk_signal(
        noisy_signal.iter().cloned(),
        samp_rate,
        symbol_rate,
        carrier_freq,
    )
    .collect();

    let t: Vec<f64> = (0..tx.len())
        .map(|idx| {
            let time_step = symbol_rate as f64 / samp_rate as f64;
            idx as f64 * time_step
        })
        .collect();

    plot!(t, tx, "/tmp/bpsk_tx.png");
    plot!(t, tx, noisy_signal, "/tmp/bpsk_tx_awgn.png");

    // plot!(t, rx_clean, rx_dirty, "/tmp/bpsk_rx_awgn.png");
    println!("ERROR: {}", error!(rx_clean, rx_dirty));
    assert!(error!(rx_clean, rx_dirty) <= 0.2);
    // assert_eq!(rx_clean, rx_dirty);

    let psd: SpectralDensity<f64> = SpectralDensity::builder(&noisy_signal, carrier_freq).build();
    let sd = psd.periodogram();
    plot!(sd.frequency(), (*sd).to_vec(), "/tmp/bpsk_specdens.png");
    let psd: SpectralDensity<f64> = SpectralDensity::builder(&tx, carrier_freq).build();
    let sd = psd.periodogram();
    plot!(
        sd.frequency(),
        (*sd).to_vec(),
        "/tmp/bpsk_specdens_clean.png"
    );

    let psd: PowerSpectrum<f64> = PowerSpectrum::builder(&noisy_signal).build();
    let sd = psd.periodogram();
    plot!(sd.frequency(), (*sd).to_vec(), "/tmp/bpsk_pwrspctrm.png");
    let psd: PowerSpectrum<f64> = PowerSpectrum::builder(&tx).build();
    let sd = psd.periodogram();
    plot!(
        sd.frequency(),
        (*sd).to_vec(),
        "/tmp/bpsk_pwrspctrm_clean.png"
    );
}

#[test]
fn bpsk_constellation() {
    let data: Vec<Bit> = vec![
        false, false, true, false, false, true, false, true, true, true,
    ];

    // Simulation parameters.
    let samp_rate = 441000; // Clock rate for both RX and TX.
    let symbol_rate = 900; // Rate symbols come out the things.
    let carrier_freq = 1800_f64;

    let tx: Vec<f64> =
        tx_bpsk_signal(data.iter().cloned(), samp_rate, symbol_rate, carrier_freq).collect();

    let samples_per_symbol: usize = samp_rate / symbol_rate;
    let filter: Vec<f64> = (0..samples_per_symbol).map(|_| 1f64).collect();

    let i_01: Vec<f64> = tx
        .iter()
        .enumerate()
        .map(move |(idx, &sample)| {
            let time = idx as f64 / samp_rate as f64;
            sample * (2_f64 * PI * carrier_freq * time).cos()
        })
        .convolve(filter.clone())
        .take_every(samples_per_symbol)
        .skip(1)
        .collect();

    let q_01: Vec<f64> = tx
        .iter()
        .enumerate()
        .map(move |(idx, &sample)| {
            let time = idx as f64 / samp_rate as f64;
            sample * -(2_f64 * PI * carrier_freq * time).sin()
        })
        .convolve(filter)
        .take_every(samples_per_symbol)
        .skip(1)
        .collect();

    let i_02: Vec<f64> = tx
        .iter()
        .enumerate()
        .map(move |(idx, &sample)| {
            let time = idx as f64 / samp_rate as f64;
            sample * (2_f64 * PI * carrier_freq * time).cos()
        })
        .integrate_and_dump(samples_per_symbol)
        // .take_every(samples_per_symbol)
        .collect();

    let q_02: Vec<f64> = tx
        .iter()
        .enumerate()
        .map(move |(idx, &sample)| {
            let time = idx as f64 / samp_rate as f64;
            sample * -(2_f64 * PI * carrier_freq * time).sin()
        })
        .integrate_and_dump(samples_per_symbol)
        // .take_every(samples_per_symbol)
        .collect();

    let i_03: Vec<f64> = tx
        .iter()
        .enumerate()
        .map(move |(idx, &sample)| {
            let time = idx as f64 / samp_rate as f64;
            sample * (2_f64 * PI * carrier_freq * time).cos()
        })
        .integrate()
        // .take_every(samples_per_symbol)
        .collect();

    let q_03: Vec<f64> = tx
        .iter()
        .enumerate()
        .map(move |(idx, &sample)| {
            let time = idx as f64 / samp_rate as f64;
            sample * -(2_f64 * PI * carrier_freq * time).sin()
        })
        .integrate()
        // .take_every(samples_per_symbol)
        .collect();

    dot_plot!(q_01, i_01, "/tmp/bpsk_iq_01.png");
    dot_plot!(q_02, i_02, "/tmp/bpsk_iq_02.png");
    dot_plot!(q_03, i_03, "/tmp/bpsk_iq_03.png");
    dot_plot!(i_01, q_01, "/tmp/bpsk_qi_01.png");
    dot_plot!(i_02, q_02, "/tmp/bpsk_qi_02.png");
    dot_plot!(i_03, q_03, "/tmp/bpsk_qi_03.png");
}
