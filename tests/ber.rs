#![allow(unused_variables, non_upper_case_globals)]
use komunikilo::bpsk::{
    rx_baseband_bpsk_signal, rx_bpsk_signal, tx_baseband_bpsk_signal, tx_bpsk_signal,
};
use komunikilo::cdma::{rx_cdma_bpsk_signal, tx_cdma_bpsk_signal};
use komunikilo::fm::tx_fsk;
use komunikilo::hadamard::HadamardMatrix;
use komunikilo::qpsk::{
    rx_baseband_qpsk_signal, rx_qpsk_signal, tx_baseband_qpsk_signal, tx_qpsk_signal,
};
use komunikilo::{awgn_complex, erfc, linspace, Bit};
use num::complex::Complex;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
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
            let sigma: f64 = (1f64 / (2f64 * i)).sqrt();
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
            let sigma = (1f64 / i).sqrt();
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
    let n_scale = samp_rate as f64 / carrier_freq;

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
        .map(|&i| {
            let sigma = not_inf((n_scale / (2f64 * i)).sqrt());
            let awgn_noise = Normal::new(0f64, sigma).unwrap();

            let noisy_signal = bpsk_tx
                .iter()
                .cloned()
                .zip(awgn_noise.sample_iter(rand::thread_rng()))
                .map(|(symb, noise)| symb + noise);

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
    let n_scale = samp_rate as f64 / carrier_freq;

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
    let n_scale = samp_rate as f64 / carrier_freq;

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

#[test]
#[ignore]
fn all_method_ber() -> PyResult<()> {
    // Simulation parameters.
    let num_bits = 40000; //100_000; //1_000_000; //4000; // How many bits to transmit overall.
    let samp_rate = 128_000; // Clock rate for both RX and TX.
    let symbol_rate = 1000; // Rate symbols come out the things.
    let carrier_freq = 2500f64;

    // Test data.
    let mut rng = rand::thread_rng();
    let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();
    let h = HadamardMatrix::new(64);
    let key = h.key(2);

    // An x-axis for plotting Eb/N0.
    let xmin = f64::MIN_POSITIVE;
    let xmax = 15f64;
    let x: Vec<f64> = linspace(xmin, xmax, 50).collect();

    // let fc_sr = carrier_freq / symbol_rate as f64;
    // let sigmas: Vec<f64> = x.iter().map(|s| fc_sr / (s).sqrt()).collect();
    let fs_fc = samp_rate as f64 / carrier_freq;
    let sigmas: Vec<f64> = x
        .iter()
        .map(|s| not_inf((fs_fc / (2f64 * s)).sqrt()))
        .collect();

    // Tx output.
    let fsk_tx: Vec<f64> = tx_fsk(
        data_bits.iter().cloned(),
        samp_rate,
        symbol_rate,
        carrier_freq,
    )
    .collect();
    let bpsk_tx: Vec<f64> = tx_bpsk_signal(
        data_bits.iter().cloned(),
        samp_rate,
        symbol_rate,
        carrier_freq,
    )
    .collect();
    let qpsk_tx: Vec<f64> = tx_qpsk_signal(
        data_bits.iter().cloned(),
        samp_rate,
        symbol_rate,
        carrier_freq,
    )
    .collect();
    let cdma_tx: Vec<f64> = tx_cdma_bpsk_signal(
        data_bits.iter().cloned(),
        samp_rate,
        symbol_rate,
        carrier_freq,
        key,
    )
    .collect();

    // Container for the Eb/N0.
    let bpsk_bers: Vec<f64> = sigmas
        .par_iter()
        .map(|&sigma| {
            let normal = Normal::new(0f64, sigma).unwrap();
            let noisy_signal = bpsk_tx
                .iter()
                .cloned()
                .zip(normal.sample_iter(rand::thread_rng()))
                .map(|(symb, noise)| symb + noise);
            let rx = rx_bpsk_signal(noisy_signal, samp_rate, symbol_rate, carrier_freq);

            rx.zip(data_bits.iter())
                .map(|(rx, &tx)| if rx == tx { 0f64 } else { 1f64 })
                .sum::<f64>()
                / num_bits as f64
        })
        .collect();

    let qpsk_bers: Vec<f64> = sigmas
        .par_iter()
        .map(|&sigma| {
            let normal = Normal::new(0f64, sigma).unwrap();
            let noisy_signal = qpsk_tx
                .iter()
                .cloned()
                .zip(normal.sample_iter(rand::thread_rng()))
                .map(|(symb, noise)| symb + noise);
            let rx = rx_qpsk_signal(noisy_signal, samp_rate, symbol_rate, carrier_freq);

            rx.zip(data_bits.iter())
                .map(|(rx, &tx)| if rx == tx { 0f64 } else { 1f64 })
                .sum::<f64>()
                / num_bits as f64
        })
        .collect();

    let cdma_bers: Vec<f64> = sigmas
        .par_iter()
        .map(|&sigma| {
            let normal = Normal::new(0f64, sigma).unwrap();
            let noisy_signal = cdma_tx
                .iter()
                .cloned()
                .zip(normal.sample_iter(rand::thread_rng()))
                .map(|(symb, noise)| symb + noise);
            let rx = rx_cdma_bpsk_signal(noisy_signal, samp_rate, symbol_rate, carrier_freq, key);

            rx.zip(data_bits.iter())
                .map(|(rx, &tx)| if rx == tx { 0f64 } else { 1f64 })
                .sum::<f64>()
                / num_bits as f64
        })
        .collect();

    let t_step: f64 = 1f64 / (samp_rate as f64);

    let theory_bers: Vec<f64> = x.iter().map(|&x| ber_bpsk(x)).collect();
    // Plot the BERS for BPSK, QPSK, FSK, CDMA/BPSK, etc...
    Python::with_gil(|py| {
        let plt = py.import_bound("matplotlib.pyplot")?;
        let np = py.import_bound("numpy")?;
        let locals = [("np", np), ("plt", plt)].into_py_dict_bound(py);

        locals.set_item("bpsk_bers", bpsk_bers)?;
        locals.set_item("theory_bers", theory_bers)?;
        locals.set_item("qpsk_bers", qpsk_bers)?;
        locals.set_item("cdma_bers", cdma_bers)?;
        locals.set_item("dt", t_step)?;

        // let x = py.eval_bound("lambda s, dt: [dt * i for i in range(len(s))]", None, None)?;
        locals.set_item("x", x)?;

        let (fig, axes): (&PyAny, &PyAny) = py
            .eval_bound("plt.subplots(1)", None, Some(&locals))?
            .extract()?;
        locals.set_item("fig", fig)?;
        locals.set_item("axes", axes)?;

        for line in [
            "axes.plot(x, bpsk_bers, label='BPSK: (2.5kHz, 1kHz Data Rate)')",
            "axes.plot(x, qpsk_bers, label='QPSK: (2.5kHz, 1kHz Data Rate)')",
            "axes.plot(x, cdma_bers, label='CDMA: 64kHz Chip Rate')",
            "axes.plot(x, theory_bers, label='1/2 erfc(sqrt(Eb/N0))')",
            "axes.set_yscale('log')",
            // "axes.set_ylim(ymin=0, ymax=0.5)",
            // "axes.set_yscale('log')",
            "axes.legend()",
            "fig.set_size_inches(16, 9)",
            // "plt.show()",
            "fig.savefig('/tmp/all_bers.png', dpi=300)",
        ] {
            py.eval_bound(line, None, Some(&locals))?;
        }

        Ok(())
    })
}
