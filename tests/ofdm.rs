use komunikilo::ofdm::{rx_baseband_ofdm_signal, tx_baseband_ofdm_signal, tx_ofdm_qpsk_signal};
use komunikilo::qpsk::{rx_baseband_qpsk_signal, tx_baseband_qpsk_signal};
use komunikilo::Bit;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use rand::Rng;
use rustfft::num_complex::{Complex, ComplexFloat};
use welch_sde::{Build, PowerSpectrum, SpectralDensity};

#[macro_use]
mod util;

#[test]
fn ofdm_qpsk_plot() {
    let num_subcarriers = 64;
    let num_pilots = 12;

    let samp_rate = 20_000_000;

    let symbol_rate = samp_rate / num_subcarriers;

    let num_ofdm_symbols = 500;
    let num_qpsk_symbols = num_ofdm_symbols * num_subcarriers;
    let num_bits = 2 * num_qpsk_symbols;

    let carrier_freq = 1e3;

    let mut rng = rand::thread_rng();
    let data: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

    let ofdm_signal: Vec<f64> = tx_ofdm_qpsk_signal(
        data.iter().cloned(),
        num_subcarriers,
        num_pilots,
        samp_rate,
        symbol_rate,
        carrier_freq,
    )
    .collect();

    let x: Vec<f64> = (0..ofdm_signal.len())
        .map(|idx| idx as f64 / samp_rate as f64)
        .collect();

    plot!(x[..10_000], ofdm_signal[..10_000], "/tmp/ofdm_qpsk.png");

    let psd: SpectralDensity<f64> = SpectralDensity::builder(&ofdm_signal, carrier_freq).build();
    let sd = psd.periodogram();
    plot!(
        sd.frequency(),
        (*sd).to_vec(),
        "/tmp/ofdm_qpsk_specdens.png"
    );

    let psd: PowerSpectrum<f64> = PowerSpectrum::builder(&ofdm_signal).build();
    let sd = psd.periodogram();
    plot!(
        sd.frequency(),
        (*sd).to_vec(),
        "/tmp/ofdm_qpsk_pwrspctrm.png"
    );
}

#[test]
#[ignore]
fn py_version() -> PyResult<()> {
    let num_bits = 2080;

    let mut rng = rand::thread_rng();
    let data: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

    let scs = 64;
    let pilots = 12;

    let tx_sig: Vec<Complex<f64>> =
        tx_baseband_ofdm_signal(tx_baseband_qpsk_signal(data.iter().cloned()), scs, pilots)
            .collect();

    let rx_dat: Vec<Bit> =
        rx_baseband_qpsk_signal(rx_baseband_ofdm_signal(tx_sig.iter().cloned(), scs, pilots))
            .collect();

    Python::with_gil(|py| {
        let pathlib = py.import_bound("pathlib")?;
        let importlib = py.import_bound("importlib")?;
        let plt = py.import_bound("matplotlib.pyplot")?;
        let np = py.import_bound("numpy")?;

        let locals = [
            ("importlib", importlib),
            ("pathlib", pathlib),
            ("np", np),
            ("plt", plt),
        ]
        .into_py_dict_bound(py);

        locals.set_item("tx_sig_rs", tx_sig.clone())?;

        let mod_path = py.eval_bound(
            "pathlib.Path.home() / 'projects' / 'comms_py' / 'ofdm_trx5.py'",
            None,
            Some(&locals),
        )?;
        locals.set_item("mod_path", mod_path)?;

        let comms_py = py.eval_bound(
            "importlib.machinery.SourceFileLoader(mod_path.name, str(mod_path)).load_module()",
            None,
            Some(&locals),
        )?;
        locals.set_item("comms_py", comms_py)?;

        locals.set_item("data", data.clone())?;
        let py_tx_sig_re: Vec<f64> = py
            .eval_bound(
                "list(np.array(comms_py.tx_ofdm(data)).real)",
                None,
                Some(&locals),
            )?
            .extract()?;
        let py_tx_sig_im: Vec<f64> /*&pyo3::PyAny*/ = py.eval_bound(
            "list(np.array(comms_py.tx_ofdm(data)).imag)",
            None,
            Some(&locals),
        )?.extract()?;

        let py_rx_dat: Vec<bool> = py
            .eval_bound(
                "list(map(bool, comms_py.rx_ofdm(comms_py.tx_ofdm(data))))",
                None,
                Some(&locals),
            )?
            .extract()?;

        let py_rx_dat_rs_py: Vec<bool> = py
            .eval_bound(
                "list(map(bool, comms_py.rx_ofdm(tx_sig_rs)))",
                None,
                Some(&locals),
            )?
            .extract()?;

        let py_tx_sig: Vec<Complex<f64>> = py_tx_sig_re
            .iter()
            .zip(py_tx_sig_im.iter())
            .map(|(&re, &im)| Complex::new(re, im))
            .collect();

        let t: Vec<f64> = (0..tx_sig.len())
            .map(|i| i as f64 /* sample_rate as f64*/)
            .collect();
        plot!(t, tx_sig, py_tx_sig, "/tmp/ofdm_py_vs_rs.png");

        let diff: Vec<f64> = py_tx_sig
            .iter()
            .zip(tx_sig.iter())
            .map(|(&ai, &bi)| (ai - bi).abs())
            .collect();
        plot!(t, diff, "/tmp/ofdm_py_vs_rs_diff.png");
        plot!(t, tx_sig, "/tmp/ofdm_tx_rs.png");
        plot!(t, py_tx_sig, "/tmp/ofdm_tx_py.png");

        assert_eq!(tx_sig.len(), py_tx_sig.len());
        assert_eq!(rx_dat.len(), py_rx_dat.len());

        assert_eq!(Complex::new(1f64, 1f64).abs(), std::f64::consts::SQRT_2);

        let tx_err: usize = tx_sig
            .iter()
            .zip(py_tx_sig.iter())
            .map(|(&i, &j)| if i == j { 0 } else { 1 })
            .sum();
        let rx_err: usize = rx_dat
            .iter()
            .zip(py_rx_dat.iter())
            .map(|(&i, &j)| if i == j { 0 } else { 1 })
            .sum();
        println!(
            "TX ERRs : {} ({}%)",
            tx_err,
            tx_err as f64 / tx_sig.len() as f64 * 100f64
        );
        println!(
            "RX ERRs : {} ({}%)",
            rx_err,
            rx_err as f64 / rx_dat.len() as f64 * 100f64
        );

        // assert_eq!(tx_sig, py_tx_sig);
        assert_eq!(rx_dat, py_rx_dat);
        assert_eq!(data, rx_dat);
        assert_eq!(data, py_rx_dat);
        assert_eq!(data, py_rx_dat_rs_py);

        PyResult::Ok(())
    })?;
    Ok(())
}
