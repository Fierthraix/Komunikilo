#![allow(unused_variables, non_upper_case_globals)]
use komunikilo::{
    bpsk::{rx_baseband_bpsk_signal, rx_bpsk_signal, tx_baseband_bpsk_signal, tx_bpsk_signal},
    cdma::{rx_cdma_bpsk_signal, tx_cdma_bpsk_signal},
    db, erfc,
    fm::tx_fsk,
    hadamard::HadamardMatrix,
    linspace,
    ofdm::{rx_baseband_ofdm_signal, tx_baseband_ofdm_signal},
    qpsk::{rx_baseband_qpsk_signal, rx_qpsk_signal, tx_baseband_qpsk_signal, tx_qpsk_signal},
    undb, Bit,
};

use num::complex::Complex;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use rstest::*;

use util::not_inf;

#[macro_use]
mod util;

fn ber_bpsk(eb_nos: &[f64]) -> Vec<f64> {
    eb_nos
        .iter()
        .map(|eb_no| 0.5 * erfc(eb_no.sqrt()))
        .collect()
}

fn ber_qpsk(eb_nos: &[f64]) -> Vec<f64> {
    eb_nos
        .iter()
        .map(|eb_no| 0.5 * erfc(eb_no.sqrt()) - 0.25 * erfc(eb_no.sqrt()).powi(2))
        .collect()
}

const SAMP_RATE: usize = 80_000; // Clock rate for both RX and TX.
const SYMBOL_RATE: usize = 1000; // Rate symbols come out the things.
const CARRIER_FREQ: f64 = 2500f64;

// const NUM_BITS: usize = 160_000;
const NUM_BITS: usize = 9002;

fn get_data(num_bits: usize) -> Vec<Bit> {
    let mut rng = rand::thread_rng();
    (0..num_bits).map(|_| rng.gen::<Bit>()).collect()
}

#[fixture]
fn data_bits() -> Vec<Bit> {
    get_data(NUM_BITS)
}

#[fixture]
fn snrs() -> Vec<f64> {
    linspace(-26f64, 6f64, 100).map(undb).collect::<Vec<f64>>()
    // let xmin = f64::MIN_POSITIVE;
    // let xmax = 15f64;
    // linspace(xmin, xmax, 100).collect()
}

macro_rules! calculate_bers {
    ($data:expr, $tx_sig:expr, $rx:expr, $snrs:expr) => {{
        let bers: Vec<f64> = $snrs
            .par_iter()
            .map(|&snr| {
                let eb: f64 = $tx_sig.iter().cloned().map(|s_i| s_i.powi(2)).sum::<f64>()
                    / $data.len() as f64;

                let n0 = not_inf((2f64 * eb / snr).sqrt());
                let awgn_noise = Normal::new(0f64, n0 / 2f64).unwrap();

                let noisy_signal = $tx_sig
                    .iter()
                    .cloned()
                    .zip(awgn_noise.sample_iter(rand::thread_rng()))
                    .map(|(symb, noise)| symb + noise);

                $rx(noisy_signal)
                    .zip($data.iter())
                    .map(|(rx, &tx)| if rx == tx { 0f64 } else { 1f64 })
                    .sum::<f64>()
                    / $data.len() as f64
            })
            .collect();
        bers
    }};
}

macro_rules! calculate_bers_baseband {
    ($data:expr, $tx_sig:expr, $rx:expr, $snrs:expr) => {{
        let bers: Vec<f64> = $snrs
            .par_iter()
            .map(|&snr| {
                let eb: f64 = $tx_sig
                    .iter()
                    .cloned()
                    .map(|s_i| s_i.norm_sqr())
                    .sum::<f64>()
                    / $data.len() as f64;

                let n0 = not_inf((eb / (2f64 * snr)).sqrt());
                let awgn_noise = Normal::new(0f64, n0).unwrap();

                let noisy_signal = $tx_sig
                    .iter()
                    .cloned()
                    .zip(awgn_noise.sample_iter(rand::thread_rng()))
                    .map(|(symb, noise)| symb + noise);

                $rx(noisy_signal)
                    .zip($data.iter())
                    .map(|(rx, &tx)| if rx == tx { 0f64 } else { 1f64 })
                    .sum::<f64>()
                    / $data.len() as f64
            })
            .collect();
        bers
    }};
}

#[rstest]
fn baseband_bpsk_works(snrs: Vec<f64>, data_bits: Vec<Bit>) {
    // Transmit the signal.
    let bpsk_tx: Vec<Complex<f64>> = tx_baseband_bpsk_signal(data_bits.iter().cloned()).collect();

    let y = calculate_bers_baseband!(data_bits, bpsk_tx, rx_baseband_bpsk_signal, snrs);
    let y_theory: Vec<f64> = ber_bpsk(&snrs);
    let snrs_db: Vec<f64> = snrs.iter().cloned().map(db).collect();

    // ber_plot!(snrs_db, y, y_theory, "/tmp/ber_baseband_bpsk.png");
    ber_plot!(snrs, y, y_theory, "/tmp/ber_baseband_bpsk.png");

    let bpsk_rx: Vec<Bit> = rx_baseband_bpsk_signal(bpsk_tx.iter().cloned()).collect();
    assert_eq!(data_bits, bpsk_rx);
}

#[rstest]
fn baseband_qpsk_works(snrs: Vec<f64>, data_bits: Vec<Bit>) {
    // Transmit the signal.
    let qpsk_tx: Vec<Complex<f64>> = tx_baseband_qpsk_signal(data_bits.iter().cloned()).collect();

    let y = calculate_bers_baseband!(data_bits, qpsk_tx, rx_baseband_qpsk_signal, snrs);

    let y_theory: Vec<f64> = ber_qpsk(&snrs);

    ber_plot!(snrs, y, y_theory, "/tmp/ber_baseband_qpsk.png");

    let qpsk_rx: Vec<Bit> = rx_baseband_qpsk_signal(qpsk_tx.iter().cloned()).collect();
    assert_eq!(data_bits, qpsk_rx);
}

#[rstest]
fn bpsk_works(snrs: Vec<f64>, data_bits: Vec<Bit>) {
    let tx_sig: Vec<f64> = tx_bpsk_signal(
        data_bits.iter().cloned(),
        SAMP_RATE,
        SYMBOL_RATE,
        CARRIER_FREQ,
    )
    .collect();

    fn rx<I: Iterator<Item = f64>>(signal: I) -> impl Iterator<Item = Bit> {
        rx_bpsk_signal(signal.into_iter(), SAMP_RATE, SYMBOL_RATE, CARRIER_FREQ)
    }

    let y = calculate_bers!(data_bits, tx_sig, rx, snrs);

    let y_theory: Vec<f64> = ber_bpsk(&snrs);

    ber_plot!(snrs, y, y_theory, "/tmp/ber_bpsk.png");
}

#[rstest]
fn qpsk_works(snrs: Vec<f64>, data_bits: Vec<Bit>) {
    // Tx output.
    let qpsk_tx: Vec<f64> = tx_qpsk_signal(
        data_bits.iter().cloned(),
        SAMP_RATE,
        SYMBOL_RATE,
        CARRIER_FREQ,
    )
    .collect();

    fn rx<I: Iterator<Item = f64>>(signal: I) -> impl Iterator<Item = Bit> {
        rx_qpsk_signal(signal.into_iter(), SAMP_RATE, SYMBOL_RATE, CARRIER_FREQ)
    }

    let y = calculate_bers!(data_bits, qpsk_tx, rx, snrs);

    let y_theory: Vec<f64> = ber_qpsk(&snrs);

    ber_plot!(snrs, y, y_theory, "/tmp/ber_qpsk.png");
    plot!(snrs, y, y_theory, "/tmp/ber_qpsk_unlog.png");
}

#[rstest]
fn cdma_works(snrs: Vec<f64>, data_bits: Vec<Bit>) {
    // Spreading Code
    let h = HadamardMatrix::new(8);
    let key = h.key(2);

    // Tx output.
    let cdma_tx: Vec<f64> = tx_cdma_bpsk_signal(
        data_bits.iter().cloned(),
        SAMP_RATE,
        SYMBOL_RATE,
        CARRIER_FREQ,
        key,
    )
    .collect();

    // TODO: FIXME: Resolve lifetime issues around keys to fix this macro!
    /*
    fn rx<I: Iterator<Item = f64>>(signal: I) -> impl Iterator<Item = Bit> {
    let rx = |signal: &dyn Iterator<Item = f64>| -> impl Iterator<Item = Bit> {
        rx_cdma_bpsk_signal(
            signal.into_iter(),
            SAMP_RATE,
            SYMBOL_RATE,
            CARRIER_FREQ,
            key,
        )
        }
    };
        */

    // let y = calculate_bers!(data_bits, cdma_tx, rx, snrs);
    let eb: f64 =
        cdma_tx.iter().cloned().map(|s_i| s_i.powi(2)).sum::<f64>() / data_bits.len() as f64;

    // Container for the Eb/N0.
    let y: Vec<f64> = snrs
        .par_iter()
        .map(|&snr| {
            let n0 = not_inf((2f64 * eb / snr).sqrt());

            let normal = Normal::new(0f64, n0 / 2f64).unwrap();
            let noisy_signal = cdma_tx
                .iter()
                .cloned()
                .zip(normal.sample_iter(rand::thread_rng()))
                .map(|(symb, noise)| symb + noise);
            let rx = rx_cdma_bpsk_signal(noisy_signal, SAMP_RATE, SYMBOL_RATE, CARRIER_FREQ, key);

            rx.zip(data_bits.iter())
                .map(|(rx, &tx)| if rx == tx { 0f64 } else { 1f64 })
                .sum::<f64>()
                / data_bits.len() as f64
        })
        .collect();

    let y_theory: Vec<f64> = ber_qpsk(&snrs);

    ber_plot!(snrs, y, y_theory, "/tmp/ber_cdma.png");
    plot!(snrs, y, y_theory, "/tmp/ber_cdma_unlog.png");
}

#[rstest]
fn baseband_ofdm_works(snrs: Vec<f64>, data_bits: Vec<Bit>) {
    let subcarriers = 64;
    let pilots = 12;
    let tx_sig: Vec<Complex<f64>> = tx_baseband_ofdm_signal(
        tx_baseband_bpsk_signal(data_bits.iter().cloned()),
        subcarriers,
        pilots,
    )
    .collect();

    fn rx<I: Iterator<Item = Complex<f64>>>(signal: I) -> impl Iterator<Item = Bit> {
        let subcarriers = 64;
        let pilots = 12;
        rx_baseband_bpsk_signal(rx_baseband_ofdm_signal(signal, subcarriers, pilots))
    }

    let y = calculate_bers_baseband!(data_bits, tx_sig, rx, snrs);

    let y_theory: Vec<f64> = ber_bpsk(&snrs);

    ber_plot!(snrs, y, y_theory, "/tmp/ber_baseband_ofdm.png");
}

#[rstest]
#[ignore]
fn all_method_ber(snrs: Vec<f64>) -> PyResult<()> {
    // Simulation parameters.
    let num_bits = 40000; //100_000; //1_000_000; //4000; // How many bits to transmit overall.
    let samp_rate_ = 128_000; // Clock rate for both RX and TX.
    let symbol_rate_ = 1000; // Rate symbols come out the things.
    let carrier_freq_ = 2500f64;

    // Test data.
    let data_bits: Vec<Bit> = get_data(num_bits);
    let h = HadamardMatrix::new(64);
    let key = h.key(2);

    let eb = |signal: &[f64]| -> f64 {
        signal.iter().map(|&s_i| s_i.powi(2)).sum::<f64>() / data_bits.len() as f64
    };

    let sigmas = |eb: f64, snrs: &[f64]| -> Vec<f64> {
        snrs.iter()
            .map(|&snr| (2f64 * eb / snr).sqrt() / 2f64)
            .collect()
    };

    // Tx output.
    let fsk_tx: Vec<f64> = tx_fsk(
        data_bits.iter().cloned(),
        samp_rate_,
        symbol_rate_,
        carrier_freq_,
    )
    .collect();
    let bpsk_tx: Vec<f64> = tx_bpsk_signal(
        data_bits.iter().cloned(),
        samp_rate_,
        symbol_rate_,
        carrier_freq_,
    )
    .collect();
    let qpsk_tx: Vec<f64> = tx_qpsk_signal(
        data_bits.iter().cloned(),
        samp_rate_,
        symbol_rate_,
        carrier_freq_,
    )
    .collect();
    let cdma_tx: Vec<f64> = tx_cdma_bpsk_signal(
        data_bits.iter().cloned(),
        samp_rate_,
        symbol_rate_,
        carrier_freq_,
        key,
    )
    .collect();

    // Container for the Eb/N0.
    let bpsk_bers: Vec<f64> = sigmas(eb(&bpsk_tx), &snrs)
        .par_iter()
        .map(|&sigma| {
            let normal = Normal::new(0f64, sigma).unwrap();
            let noisy_signal = bpsk_tx
                .iter()
                .cloned()
                .zip(normal.sample_iter(rand::thread_rng()))
                .map(|(symb, noise)| symb + noise);
            let rx = rx_bpsk_signal(noisy_signal, samp_rate_, symbol_rate_, carrier_freq_);

            rx.zip(data_bits.iter())
                .map(|(rx, &tx)| if rx == tx { 0f64 } else { 1f64 })
                .sum::<f64>()
                / num_bits as f64
        })
        .collect();

    let qpsk_bers: Vec<f64> = sigmas(eb(&qpsk_tx), &snrs)
        .par_iter()
        .map(|&sigma| {
            let normal = Normal::new(0f64, sigma).unwrap();
            let noisy_signal = qpsk_tx
                .iter()
                .cloned()
                .zip(normal.sample_iter(rand::thread_rng()))
                .map(|(symb, noise)| symb + noise);
            let rx = rx_qpsk_signal(noisy_signal, samp_rate_, symbol_rate_, carrier_freq_);

            rx.zip(data_bits.iter())
                .map(|(rx, &tx)| if rx == tx { 0f64 } else { 1f64 })
                .sum::<f64>()
                / num_bits as f64
        })
        .collect();

    let cdma_bers: Vec<f64> = sigmas(eb(&cdma_tx), &snrs)
        .par_iter()
        .map(|&sigma| {
            let normal = Normal::new(0f64, sigma).unwrap();
            let noisy_signal = cdma_tx
                .iter()
                .cloned()
                .zip(normal.sample_iter(rand::thread_rng()))
                .map(|(symb, noise)| symb + noise);
            let rx =
                rx_cdma_bpsk_signal(noisy_signal, samp_rate_, symbol_rate_, carrier_freq_, key);

            rx.zip(data_bits.iter())
                .map(|(rx, &tx)| if rx == tx { 0f64 } else { 1f64 })
                .sum::<f64>()
                / num_bits as f64
        })
        .collect();

    let t_step: f64 = 1f64 / (samp_rate_ as f64);

    let theory_bers: Vec<f64> = ber_bpsk(&snrs);
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
        locals.set_item("x", snrs)?;

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
