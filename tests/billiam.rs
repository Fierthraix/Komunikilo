#![allow(non_upper_case_globals, unused_variables, unused_macros, dead_code)]
use assert_approx_eq::assert_approx_eq;
use komunikilo::bpsk::tx_bpsk_signal;
use komunikilo::cdma::{tx_cdma_bpsk_signal, tx_cdma_qpsk_signal};
use komunikilo::fsk::tx_bfsk_signal;
use komunikilo::hadamard::HadamardMatrix;
use komunikilo::ofdm::tx_ofdm_qpsk_signal;
use komunikilo::qpsk::tx_qpsk_signal;
use komunikilo::{avg_energy, awgn, erfc, linspace, Bit};
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use rand::Rng;
use rayon::prelude::*;
use util::fit_erfc;

#[macro_use]
mod util;

#[test]
#[ignore]
fn radiometer_pd_new() {
    /*
     * NOTE: $E_B = signal.map(|s_i| s_i.powi(2)).sum / (samp_freq * num_bits)$
     */
    fn get_pds(
        snrs: &[f64],
        attempts_var: usize,
        num_bits: usize,
        samp_rate: usize,
        tx: &[f64],
    ) -> Vec<f64> {
        let e_b: f64 = tx.iter().cloned().map(|s_i| s_i.powi(2)).sum::<f64>()
            / ((samp_rate * num_bits) as f64);
        let p_ds: Vec<f64> = snrs
            .par_iter()
            .map(|&snr| {
                let n_0: f64 = e_b / snr;

                let mut energies = Vec::with_capacity(attempts_var);
                for _ in 0..attempts_var {
                    let chan_sig: Vec<f64> = awgn(tx.iter().cloned(), n_0).collect();
                    energies.push(avg_energy(&chan_sig));
                }

                let p_d: f64 = energies
                    .iter()
                    .map(|&nrg| if nrg > (e_b / n_0) { 0 } else { 1 })
                    .sum::<usize>() as f64
                    / attempts as f64;
                p_d
            })
            .collect();
        p_ds
    }

    const fs: usize = 43200;
    const sr: usize = 900;
    const fc: f64 = 1800f64;

    const nb: usize = 1024;
    const attempts: usize = 1000; //1000;

    const fl: f64 = 1600f64;
    const fh: f64 = 1800f64;

    let mut rng = rand::thread_rng();
    let data: Vec<Bit> = (0..nb).map(|_| rng.gen::<Bit>()).collect();

    let h = HadamardMatrix::new(16);
    let key = h.key(2);

    let bpsk_snrs: Vec<f64> = linspace(0.9999, 1.0001, 250 /*500*/).collect();

    let bpsk_tx: Vec<f64> = tx_bpsk_signal(data.iter().cloned(), fs, sr, fc).collect();
    let bpsk_p_ds = get_pds(&bpsk_snrs, attempts, nb, fs, &bpsk_tx);
    dot_plot!(bpsk_snrs, bpsk_p_ds, "/tmp/bpsk_covertness_radiometer.png");

    let cdma_bpsk_tx: Vec<f64> =
        tx_cdma_bpsk_signal(data.iter().cloned(), fs, sr, fc, key).collect();
    let cdma_bpsk_p_ds = get_pds(&bpsk_snrs, attempts, nb, fs, &cdma_bpsk_tx);
    dot_plot!(
        bpsk_snrs,
        cdma_bpsk_p_ds,
        "/tmp/cdma_bpsk_covertness_radiometer.png"
    );

    let qpsk_tx: Vec<f64> = tx_qpsk_signal(data.iter().cloned(), fs, sr, fc).collect();
    let qpsk_p_ds = get_pds(&bpsk_snrs, attempts, nb, fs, &qpsk_tx);
    dot_plot!(bpsk_snrs, qpsk_p_ds, "/tmp/qpsk_covertness_radiometer.png");

    let cdma_qpsk_tx: Vec<f64> =
        tx_cdma_qpsk_signal(data.iter().cloned(), fs, sr, fc, key).collect();
    let cdma_qpsk_p_ds = get_pds(&bpsk_snrs, attempts, nb, fs, &cdma_qpsk_tx);
    dot_plot!(
        bpsk_snrs,
        cdma_qpsk_p_ds,
        "/tmp/cdma_qpsk_covertness_radiometer.png"
    );

    // let fsk_snrs: Vec<f64> = linspace(0.74, 0.75, 250).collect();
    let fsk_snrs: Vec<f64> = linspace(0.1, 0.9, 250).collect();
    // let fsk_snrs: Vec<f64> = linspace(0.275, 10., 100).collect();
    let fsk_tx: Vec<f64> = tx_bfsk_signal(data.iter().cloned(), fs, sr, fl, fh).collect();
    let fsk_p_ds = get_pds(&fsk_snrs, attempts, nb, fs, &fsk_tx);
    dot_plot!(fsk_snrs, fsk_p_ds, "/tmp/fsk_covertness_radiometer.png");

    let num_subcarriers = 64;

    let samp_rate = 2_000_000;
    let symbol_rate = samp_rate / num_subcarriers;

    let carrier_freq = 1e3;

    // let ofdm_snrs: Vec<f64> = linspace(0.0245, 0.025, 250).collect();
    // let ofdm_snrs: Vec<f64> = linspace(0.0245, 1.1, 1000).collect();
    let ofdm_snrs: Vec<f64> = linspace(0.001, 0.035, 100).collect();
    let ofdm_signal: Vec<f64> = tx_ofdm_qpsk_signal(
        data.iter().cloned(),
        64,
        12,
        samp_rate,
        symbol_rate,
        carrier_freq,
    )
    .collect();
    let ofdm_p_ds = get_pds(&ofdm_snrs, attempts * 10, nb, fs, &ofdm_signal);
    dot_plot!(ofdm_snrs, ofdm_p_ds, "/tmp/ofdm_covertness_radiometer.png");

    Python::with_gil(|py| {
        let locals = init_matplotlib!(py);

        locals.set_item("snrs", bpsk_snrs).unwrap();
        // locals.set_item("fsk_p_ds", fsk_p_ds).unwrap();
        locals.set_item("bpsk_p_ds", bpsk_p_ds).unwrap();
        locals.set_item("qpsk_p_ds", qpsk_p_ds).unwrap();
        locals.set_item("cdma_bpsk_p_ds", cdma_bpsk_p_ds).unwrap();
        locals.set_item("cdma_qpsk_p_ds", cdma_qpsk_p_ds).unwrap();

        let (fig, axes): (&PyAny, &PyAny) = py
            .eval_bound("plt.subplots(1)", None, Some(&locals))
            .unwrap()
            .extract()
            .unwrap();
        locals.set_item("fig", fig).unwrap();
        locals.set_item("axes", axes).unwrap();
        py.eval_bound("fig.set_size_inches(12, 7)", None, Some(&locals))
            .unwrap();
        for line in [
            // "axes.plot(snrs, fsk_p_ds, label='FSK', marker='.', linestyle='None')",
            "axes.plot(snrs, bpsk_p_ds, label='BPSK', marker='1', linestyle='None')",
            "axes.plot(snrs, qpsk_p_ds, label='QPSK', marker='+', linestyle='None')",
            "axes.plot(snrs, cdma_bpsk_p_ds, label='CDMA-BPSK', marker='2', linestyle='None')",
            "axes.plot(snrs, cdma_qpsk_p_ds, label='CDMA-QPSK', marker='x', linestyle='None')",
            "axes.legend(loc='best')",
            "axes.set_ylabel('Probablility of Detection')",
            "axes.set_xlabel('Eb/N0')",
            &format!("fig.savefig('{}')", "/tmp/probability_detection_psk.png"),
            "plt.close('all')",
        ] {
            py.eval_bound(line, None, Some(&locals)).unwrap();
        }
    })

    /*
    let coeffs_fsk = fit_erfc(&fsk_snrs, &fsk_p_ds);
    let coeffs_bpsk = fit_erfc(&bpsk_snrs, &bpsk_p_ds);
    let coeffs_qpsk = fit_erfc(&bpsk_snrs, &qpsk_p_ds);
    let coeffs_cdma_bpsk = fit_erfc(&bpsk_snrs, &cdma_bpsk_p_ds);
    let coeffs_cdma_qpsk = fit_erfc(&bpsk_snrs, &cdma_qpsk_p_ds);

    fn my_erfc(x: &[f64], q: (f64, f64, f64, f64)) -> Vec<f64> {
        x.iter()
            .map(|x| q.0 * erfc((x - q.2) * q.3) + q.1)
            .collect()
    }

    let ebs: Vec<f64> = linspace(0.001, 2.5, 1000).collect();

    let fsk_erfc: Vec<f64> = my_erfc(&ebs, coeffs_fsk);
    let bpsk_erfc: Vec<f64> = my_erfc(&ebs, coeffs_bpsk);
    let qpsk_erfc: Vec<f64> = my_erfc(&ebs, coeffs_qpsk);
    let cdma_bpsk_erfc: Vec<f64> = my_erfc(&ebs, coeffs_cdma_bpsk);
    let cdma_qpsk_erfc: Vec<f64> = my_erfc(&ebs, coeffs_cdma_qpsk);

    Python::with_gil(|py| {
        let locals = init_matplotlib!(py);

        locals.set_item("snrs", bpsk_snrs).unwrap();
        locals.set_item("fsk_erfc", fsk_erfc).unwrap();
        locals.set_item("bpsk_erfc", bpsk_erfc).unwrap();
        locals.set_item("qpsk_erfc", qpsk_erfc).unwrap();
        locals.set_item("cdma_bpsk_erfc", cdma_bpsk_erfc).unwrap();
        locals.set_item("cdma_qpsk_erfc", cdma_qpsk_erfc).unwrap();

        let (fig, axes): (&PyAny, &PyAny) = py
            .eval_bound("plt.subplots(1)", None, Some(&locals))
            .unwrap()
            .extract()
            .unwrap();
        locals.set_item("fig", fig).unwrap();
        locals.set_item("axes", axes).unwrap();
        py.eval_bound("fig.set_size_inches(16, 9)", None, Some(&locals))
            .unwrap();
        for line in [
            // "axes.plot(snrs, fsk_erfc)",
            "axes.plot(snrs, bpsk_erfc)",
            "axes.plot(snrs, qpsk_erfc)",
            "axes.plot(snrs, cdma_bpsk_erfc)",
            "axes.plot(snrs, cdma_qpsk_erfc)",
            &format!("fig.savefig('{}')", "/tmp/qwer.png"),
            "plt.close('all')",
        ] {
            py.eval_bound(line, None, Some(&locals)).unwrap();
        }
    })
    */
}

#[test]
#[ignore]
fn radiometer_pd() {
    fn get_pds(
        ebs: &[f64],
        attempts_var: usize,
        num_bits: usize,
        tx: fn(Vec<Bit>, f64) -> Vec<f64>,
    ) -> Vec<f64> {
        let p_ds: Vec<f64> = ebs
            .par_iter()
            .map(|&eb| {
                let mut energies = Vec::with_capacity(attempts_var);
                for _ in 0..attempts_var {
                    let mut rng = rand::thread_rng();
                    let data: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();
                    let chan_sig: Vec<f64> = awgn(tx(data, eb).into_iter(), 1f64).collect();
                    energies.push(avg_energy(&chan_sig));
                }

                let p_d: f64 = energies
                    .iter()
                    .map(|&nrg| if nrg > eb { 0 } else { 1 })
                    .sum::<usize>() as f64
                    / attempts as f64;
                p_d
            })
            .collect();
        p_ds
    }

    const fs: usize = 43200;
    const sr: usize = 900;
    const fc: f64 = 1800f64;

    const nb: usize = 1000;
    const attempts: usize = 100; //1000;

    const fl: f64 = 1600f64;
    const fh: f64 = 1800f64;

    // let bpsk_ebs: Vec<f64> = linspace(1.275, 1.4, 100).collect();
    let bpsk_ebs: Vec<f64> = linspace(1.9, 2.1, 75).collect();
    let bpsk_tx =
        |x: Vec<Bit>, e: f64| -> Vec<f64> { tx_bpsk_signal(x.into_iter(), fs, sr, fc).collect() };
    let bpsk_p_ds = get_pds(&bpsk_ebs, attempts, nb, bpsk_tx);
    dot_plot!(bpsk_ebs, bpsk_p_ds, "/tmp/bpsk_covertness_radiometer.png");

    let qpsk_ebs: Vec<f64> = linspace(1.9, 2.1, 150).collect();
    let qpsk_tx =
        |x: Vec<Bit>, e: f64| -> Vec<f64> { tx_qpsk_signal(x.into_iter(), fs, sr, fc).collect() };
    let qpsk_p_ds = get_pds(&qpsk_ebs, attempts, nb, qpsk_tx);
    dot_plot!(qpsk_ebs, qpsk_p_ds, "/tmp/qpsk_covertness_radiometer.png");

    let fsk_ebs: Vec<f64> = linspace(1.5, 2.0, 100).collect();
    // let fsk_ebs: Vec<f64> = linspace(0.275, 10., 100).collect();
    let fsk_tx = |x: Vec<Bit>, e: f64| -> Vec<f64> {
        tx_bfsk_signal(x.into_iter(), fs, sr, fl, fh).collect()
    };
    let fsk_p_ds = get_pds(&fsk_ebs, attempts, nb, fsk_tx);
    dot_plot!(fsk_ebs, fsk_p_ds, "/tmp/fsk_covertness_radiometer.png");

    let cdma_bpsk_ebs: Vec<f64> = linspace(1.9, 2.1, 75).collect();
    let cdma_bpsk_tx = |x: Vec<Bit>, e: f64| -> Vec<f64> {
        let h = HadamardMatrix::new(16);
        let key = h.key(2);
        tx_cdma_bpsk_signal(x.into_iter(), fs, sr, fc, key).collect()
    };
    let cdma_bpsk_p_ds = get_pds(&cdma_bpsk_ebs, attempts, nb, cdma_bpsk_tx);
    dot_plot!(
        cdma_bpsk_ebs,
        cdma_bpsk_p_ds,
        "/tmp/cdma_bpsk_covertness_radiometer.png"
    );

    // let cdma_qpsk_ebs: Vec<f64> = linspace(1.9, 2.1, 75).collect();
    let cdma_qpsk_ebs: Vec<f64> = linspace(1.25, 1.5, 175).collect();
    let cdma_qpsk_tx = |x: Vec<Bit>, e: f64| -> Vec<f64> {
        let h = HadamardMatrix::new(16);
        let key = h.key(2);
        tx_cdma_qpsk_signal(x.into_iter(), fs, sr, fc, key).collect()
    };
    let cdma_qpsk_p_ds = get_pds(&cdma_qpsk_ebs, attempts, nb, cdma_qpsk_tx);
    dot_plot!(
        cdma_qpsk_ebs,
        cdma_qpsk_p_ds,
        "/tmp/cdma_qpsk_covertness_radiometer.png"
    );

    fn my_erfc(x: &[f64], q: (f64, f64, f64, f64)) -> Vec<f64> {
        x.iter()
            .map(|x| q.0 * erfc((x - q.2) * q.3) + q.1)
            .collect()
    }

    let coeffs_fsk = fit_erfc(&fsk_ebs, &fsk_p_ds);
    let coeffs_bpsk = fit_erfc(&bpsk_ebs, &bpsk_p_ds);
    let coeffs_qpsk = fit_erfc(&qpsk_ebs, &qpsk_p_ds);
    let coeffs_cdma_bpsk = fit_erfc(&cdma_bpsk_ebs, &cdma_bpsk_p_ds);
    let coeffs_cdma_qpsk = fit_erfc(&cdma_qpsk_ebs, &cdma_qpsk_p_ds);

    let ebs: Vec<f64> = linspace(0.001, 2.5, 1000).collect();

    let fsk_erfc: Vec<f64> = my_erfc(&ebs, coeffs_fsk);
    let bpsk_erfc: Vec<f64> = my_erfc(&ebs, coeffs_bpsk);
    let qpsk_erfc: Vec<f64> = my_erfc(&ebs, coeffs_qpsk);
    let cdma_bpsk_erfc: Vec<f64> = my_erfc(&ebs, coeffs_cdma_bpsk);
    let cdma_qpsk_erfc: Vec<f64> = my_erfc(&ebs, coeffs_cdma_qpsk);

    Python::with_gil(|py| {
        let locals = init_matplotlib!(py);

        locals.set_item("ebs", ebs).unwrap();
        locals.set_item("fsk_erfc", fsk_erfc).unwrap();
        locals.set_item("bpsk_erfc", bpsk_erfc).unwrap();
        locals.set_item("qpsk_erfc", qpsk_erfc).unwrap();
        locals.set_item("cdma_bpsk_erfc", cdma_bpsk_erfc).unwrap();
        locals.set_item("cdma_qpsk_erfc", cdma_qpsk_erfc).unwrap();

        let (fig, axes): (&PyAny, &PyAny) = py
            .eval_bound("plt.subplots(1)", None, Some(&locals))
            .unwrap()
            .extract()
            .unwrap();
        locals.set_item("fig", fig).unwrap();
        locals.set_item("axes", axes).unwrap();
        py.eval_bound("fig.set_size_inches(16, 9)", None, Some(&locals))
            .unwrap();
        for line in [
            "axes.plot(ebs, fsk_erfc)",
            "axes.plot(ebs, bpsk_erfc)",
            "axes.plot(ebs, qpsk_erfc)",
            "axes.plot(ebs, cdma_bpsk_erfc)",
            "axes.plot(ebs, cdma_qpsk_erfc)",
            &format!("fig.savefig('{}')", "/tmp/qwer.png"),
            "plt.close('all')",
        ] {
            py.eval_bound(line, None, Some(&locals)).unwrap();
        }
    })
}

#[test]
#[ignore]
fn radiometer_pmd_cdma() {
    let samp_rate = 43200; // Clock rate for both RX and TX.
    let symbol_rate = 900; // Rate symbols come out the things.
    let carrier_freq = 1800_f64;

    let num_bits: usize = 1000;

    // let tx: Vec<f64> =
    //     tx_bpsk_signal(data.iter().cloned(), samp_rate, symbol_rate, carrier_freq).collect();

    let ebs: Vec<f64> = linspace(1.85, 2.15, 75).collect();
    // let ebs: Vec<f64> = linspace(0.001, 0.25, /*2.025,*/ 150).collect();
    // let ebs: Vec<f64> = linspace(0.0, 5.0, /*2.025,*/ 150).collect();

    let attempts: usize = 1000;
    // let (cdma_p_mds, cdma_bers): (Vec<f64>, Vec<f64>) = ebs
    let cdma_p_mds: Vec<f64> = ebs
        .par_iter()
        .map(|&eb| {
            // For each SNR value (Eb/N0):
            // let energies/*: Vec<f64>*/ = (0..attempts)  // For each try of the SNR:
            let h = HadamardMatrix::new(16);
            let key = h.key(2);

            let mut energies = Vec::with_capacity(attempts);
            // let mut bers = Vec::with_capacity(attempts);
            for _ in 0..attempts {
                let mut rng = rand::thread_rng();
                let data: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect(); // Make new random data.

                // Scale it, and add AWGN noise of N0 = 1.
                let chan_sig: Vec<f64> = awgn(
                    tx_cdma_bpsk_signal(
                        data.iter().cloned(),
                        samp_rate,
                        symbol_rate,
                        carrier_freq,
                        key,
                    ),
                    1f64,
                )
                .collect();
                // Calculate the average energy.
                // let rx: Vec<Bit> = rx_cdma_bpsk_signal(
                //     chan_sig.iter().cloned(),
                //     samp_rate,
                //     symbol_rate,
                //     carrier_freq,
                //     key,
                // )
                // .collect();
                energies.push(avg_energy(&chan_sig));
                // bers.push(ber(&data, &rx));
            }

            let p_md: f64 = energies
                .iter()
                .map(|&nrg| if nrg > eb { 1 } else { 0 })
                .sum::<usize>() as f64
                / attempts as f64;

            // let ber = bers.iter().sum::<f64>() / bers.len() as f64;

            // (p_md, ber)
            p_md
        })
        .collect();
    // .unzip();
    dot_plot!(ebs, cdma_p_mds, "/tmp/cdma_covertness_radiometer.png");
    // dot_plot!(ebs, cdma_bers, "/tmp/cdma_covertness_radiometer_ber.png");
}

#[test]
#[ignore]
fn radiometer_pmd_cdma_qpsk() {
    let samp_rate = 43200; // Clock rate for both RX and TX.
    let symbol_rate = 900; // Rate symbols come out the things.
    let carrier_freq = 1800_f64;

    let num_bits: usize = 1000;

    let ebs: Vec<f64> = linspace(1.85, 2.15, 75).collect();
    // let ebs: Vec<f64> = linspace(0.001, 0.25, /*2.025,*/ 150).collect();
    // let ebs: Vec<f64> = linspace(0.0, 5.0, /*2.025,*/ 150).collect();

    let attempts: usize = 1000;
    let cdma_p_mds: Vec<f64> = ebs
        .par_iter()
        .map(|&eb| {
            let h = HadamardMatrix::new(16);
            let key = h.key(2);

            let mut energies = Vec::with_capacity(attempts);
            for _ in 0..attempts {
                let mut rng = rand::thread_rng();
                let data: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect(); // Make new random data.

                // Scale it, and add AWGN noise of N0 = 1.
                let chan_sig: Vec<f64> = awgn(
                    tx_cdma_qpsk_signal(
                        data.iter().cloned(),
                        samp_rate,
                        symbol_rate,
                        carrier_freq,
                        key,
                    ),
                    1f64,
                )
                .collect();
                energies.push(avg_energy(&chan_sig));
            }

            let p_md: f64 = energies
                .iter()
                .map(|&nrg| if nrg > eb { 1 } else { 0 })
                .sum::<usize>() as f64
                / attempts as f64;
            p_md
        })
        .collect();
    dot_plot!(ebs, cdma_p_mds, "/tmp/cdma_qpsk_covertness_radiometer.png");

    Python::with_gil(|py| {
        let pickle = py.import_bound("pickle")?;
        let pathlib = py.import_bound("pathlib")?;
        let locals = [("pickle", pickle), ("pathlib", pathlib)].into_py_dict_bound(py);

        locals.set_item("ebs", ebs.clone())?;
        locals.set_item("p_mds", cdma_p_mds.clone())?;

        for line in [
            "pickle.dump(ebs, pathlib.Path('/tmp/cdma_qpsk_snrs.pkl').open('wb'))",
            "pickle.dump(p_mds, pathlib.Path('/tmp/cdma_qpsk_p_mds.pkl').open('wb'))",
        ] {
            py.eval_bound(line, None, Some(&locals)).unwrap();
        }

        PyResult::Ok(())
    })
    .unwrap();

    let abcd = fit_erfc(&ebs, &cdma_p_mds);
    assert_approx_eq!(abcd.0, 0.5, 0.5);
    assert_approx_eq!(abcd.1, 0.01, 0.5);
    assert_approx_eq!(abcd.2, 2.0, 0.5);
    assert_approx_eq!(abcd.3, 22.2, 0.5);
}
