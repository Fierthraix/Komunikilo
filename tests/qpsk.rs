#![allow(non_snake_case)]
use komunikilo::qpsk::{rx_qpsk_signal, tx_baseband_qpsk_signal, tx_qpsk_signal};
use komunikilo::{awgn, iter::Iter, Bit};
use num::Complex;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use sci_rs::signal::filter::{design::*, sosfiltfilt_dyn};
use std::f64::consts::PI;
use welch_sde::{Build, PowerSpectrum, SpectralDensity};

#[macro_use]
mod util;

#[test]
fn qpsk_graphs() {
    let data: Vec<Bit> = [
        false, false, true, false, false, true, false, true, true, true,
    ]
    .into_iter()
    .inflate(2)
    .collect();

    let samp_rate = 4_410_000;
    let fc = 2000_f64;
    let symb_rate = 900;

    let tx: Vec<f64> = tx_qpsk_signal(data.iter().cloned(), samp_rate, symb_rate, fc).collect();
    let rx: Vec<Bit> = rx_qpsk_signal(tx.iter().cloned(), samp_rate, symb_rate, fc).collect();

    assert_eq!(rx, data);

    let sigma = 2f64;
    let noisy_signal: Vec<f64> = awgn(tx.iter().cloned(), sigma).collect();

    let psd: SpectralDensity<f64> = SpectralDensity::builder(&tx, fc).build();
    let sd = psd.periodogram();
    plot!(sd.frequency(), (*sd).to_vec(), "/tmp/qpsk_specdens.png");
    let psd: SpectralDensity<f64> = SpectralDensity::builder(&noisy_signal, fc).build();
    let sd = psd.periodogram();
    plot!(
        sd.frequency(),
        (*sd).to_vec(),
        "/tmp/qpsk_specdens_dirty.png"
    );

    let psd: PowerSpectrum<f64> = PowerSpectrum::builder(&tx).build();
    let sd = psd.periodogram();
    plot!(sd.frequency(), (*sd).to_vec(), "/tmp/qpsk_pwrspctrm.png");
    let psd: PowerSpectrum<f64> = PowerSpectrum::builder(&noisy_signal).build();
    let sd = psd.periodogram();
    plot!(
        sd.frequency(),
        (*sd).to_vec(),
        "/tmp/qpsk_pwrspctrm_dirty.png"
    );

    // I/Q Data
    /*
    let mut fftp = RealFftPlanner::<f64>::new();
    let fft = fftp.plan_fft_forward(tx.len());

    let mut spectrum: Vec<Complex<f64>> = fft.make_output_vec();
    let mut sig = tx.clone();
    // assert_eq!(sig.len(), 8192);
    fft.process(&mut sig, &mut spectrum).unwrap();

    let I: Vec<f64> = spectrum.iter().map(|&c_i| c_i.re).collect();
    let Q: Vec<f64> = spectrum.iter().map(|&c_i| c_i.im).collect();
    */

    let samples_per_symbol = samp_rate / symb_rate;
    let filter: Vec<f64> = (0..samples_per_symbol).map(|_| 1f64).collect();
    /*
    let I2: Vec<f64> = tx
        .iter()
        .enumerate()
        .map(|(i, &s_i)| s_i * (2f64 * PI * fc * (i as f64 / samp_rate as f64)).cos())
        .convolve(filter.clone())
        .take_every(samples_per_symbol)
        .skip(2)
        .collect();
    let Q2: Vec<f64> = tx
        .iter()
        .enumerate()
        .map(|(i, &s_i)| s_i * -(2f64 * PI * fc * (i as f64 / samp_rate as f64)).sin())
        .convolve(filter)
        .take_every(samples_per_symbol)
        .skip(2)
        .collect();
    */
    let I2: Vec<f64> = komunikilo::iter::Iter::chunks(
        tx.iter()
            .cloned()
            .enumerate()
            .map(|(i, s_i)| s_i * (2f64 * PI * fc * (i as f64 / samp_rate as f64)).cos()),
        samples_per_symbol,
    )
    .map(|chunk| chunk.into_iter().convolve(filter.clone()).last().unwrap())
    .collect();

    let Q2: Vec<f64> = komunikilo::iter::Iter::chunks(
        tx.iter()
            .cloned()
            .enumerate()
            .map(|(i, s_i)| s_i * -(2f64 * PI * fc * (i as f64 / samp_rate as f64)).sin()),
        samples_per_symbol,
    )
    .map(|chunk| chunk.into_iter().convolve(filter.clone()).last().unwrap())
    .collect();

    dot_plot!(I2, Q2, "/tmp/qpsk_IQ.png");

    let I3: Vec<f64> = tx
        .iter()
        .enumerate()
        .map(move |(idx, sample)| {
            let time = idx as f64 / samp_rate as f64;
            sample * (2_f64 * PI * fc * time).cos()
        })
        .integrate_and_dump(samples_per_symbol)
        .collect();

    let Q3: Vec<f64> = tx
        .iter()
        .enumerate()
        .map(move |(idx, sample)| {
            let time = idx as f64 / samp_rate as f64;
            // let ii = sample * (2_f64 * PI * fc * time).cos();
            sample * -(2_f64 * PI * fc * time).sin()
        })
        .integrate_and_dump(samples_per_symbol)
        .collect();

    dot_plot!(I3, Q3, "/tmp/qpsk_IQ2.png");
}

#[test]
fn qpsk_from_baseband() -> PyResult<()> {
    let data: Vec<Bit> = [
        false, false, true, false, false, true, false, true, true, true,
    ]
    .into_iter()
    .inflate(2)
    .collect();

    let samp_rate = 44_000;
    let fc = 1_000f64;
    let symb_rate = 400;

    let tx_normal: Vec<f64> =
        tx_qpsk_signal(data.iter().cloned(), samp_rate, symb_rate, fc).collect();
    // let rx: Vec<Bit> = rx_qpsk_signal(tx.iter().cloned(), samp_rate, symb_rate, fc).collect();

    let manual: Vec<f64> = tx_baseband_qpsk_signal(data.iter().cloned())
        .inflate(samp_rate / symb_rate)
        .enumerate()
        .map(|(i, s_i)| {
            let time = i as f64 / samp_rate as f64;
            let w0 = 2f64 * PI * fc * 4f64;
            // let w0 = 2f64 * PI * fc;

            s_i.re * (w0 * time).cos() - s_i.im * (w0 * time).sin() / 2f64.sqrt()
        })
        .collect();

    let x: Vec<f64> = (0..(data.len() * samp_rate / symb_rate) / 2)
        .map(|i| i as f64 / samp_rate as f64)
        .collect();
    assert_eq!(x.len(), tx_normal.len());

    let rx_qpsk_from_baseband_i: Vec<f64> = tx_normal
        .iter()
        .cloned()
        .enumerate()
        .map(|(idx, s_i)| s_i * 1f64 * (2f64 * PI * fc * (idx as f64 / samp_rate as f64)).cos())
        .collect();
    let rx_qpsk_from_baseband_q: Vec<f64> = tx_normal
        .iter()
        .cloned()
        .enumerate()
        .map(|(idx, s_i)| s_i * -1f64 * (2f64 * PI * fc * (idx as f64 / samp_rate as f64)).sin())
        .collect();

    let filter = {
        let bandpass_filter = butter_dyn(
            2,
            [fc].to_vec(),
            Some(FilterBandType::Lowpass),
            Some(false),
            Some(FilterOutputType::Sos),
            Some(samp_rate as f64),
        );
        let DigitalFilter::Sos(sos) = bandpass_filter else {
            panic!("Failed to design filter");
        };
        sos
    };
    let filtered_rx_i = sosfiltfilt_dyn(rx_qpsk_from_baseband_i.iter().cloned(), &filter.sos);
    let filtered_rx_q = sosfiltfilt_dyn(rx_qpsk_from_baseband_q.iter().cloned(), &filter.sos);

    let filter: Vec<f64> = (0..samp_rate / symb_rate).map(|_| 1f64).collect();
    let rx_convolved_i: Vec<f64> = filtered_rx_i // rx_qpsk_from_baseband_i
        .iter()
        .cloned()
        .convolve(filter.clone())
        .take_every(samp_rate / symb_rate)
        .collect();
    let rx_convolved_q: Vec<f64> = filtered_rx_q // rx_qpsk_from_baseband_q
        .iter()
        .cloned()
        .convolve(filter.clone())
        .take_every(samp_rate / symb_rate)
        .collect();

    plot!(
        x,
        rx_qpsk_from_baseband_i,
        "/tmp/qpsk_manual_rx_i_multiplied.png"
    );
    plot!(
        x,
        filtered_rx_i,
        "/tmp/qpsk_manual_rx_i_multiplied_and_filtered.png"
    );
    plot!(
        x,
        rx_qpsk_from_baseband_q,
        "/tmp/qpsk_manual_rx_q_multiplied.png"
    );
    plot!(
        x,
        filtered_rx_q,
        "/tmp/qpsk_manual_rx_q_multiplied_and_filtered.png"
    );
    plot!(
        (0..rx_convolved_i.len()).collect::<Vec<usize>>(),
        rx_convolved_i,
        "/tmp/qpsk_manual_rx_q_multiplied_and_filtered_and_convolved.png"
    );
    plot!(
        (0..rx_convolved_q.len()).collect::<Vec<usize>>(),
        rx_convolved_q,
        "/tmp/qpsk_manual_rx_q_multiplied_and_filtered_and_convolved.png"
    );

    let I_Q_symbols: Vec<Complex<f64>> = filtered_rx_i
        .iter()
        .cloned()
        .zip(filtered_rx_q.iter().cloned())
        .take_every(samp_rate / symb_rate / 4)
        .map(|(b_i, b_q)| Complex::new(b_i, b_q))
        .collect();

    Python::with_gil(|py| {
        let locals = init_matplotlib!(py);
        // for (key, val) in [("rx_q", filtered_rx_q), ("rx_i", filtered_rx_i)] {
        for (key, val) in [("rx_q", rx_convolved_q), ("rx_i", rx_convolved_i)] {
            locals.set_item(key, val).unwrap();
        }

        for line in [
            "plt.plot(rx_i, rx_q, '.')",
            "plt.savefig('/tmp/asdf-01.png')",
        ] {
            py.eval_bound(line, None, Some(&locals)).unwrap();
        }
    });

    println!("{:?}", I_Q_symbols);
    // assert!(false);

    /*
    manual
        .iter()
        .zip(tx_normal.iter())
        .for_each(|(&m_i, &s_i)| assert!((m_i - s_i).abs() < 10e-3));
    */
    plot!(x, tx_normal, manual, "/tmp/qpsk_manual.png");
    Ok(())
}
