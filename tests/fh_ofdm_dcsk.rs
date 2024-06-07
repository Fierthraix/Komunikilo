use komunikilo::cdma::tx_cdma_bpsk_signal;
use komunikilo::fh_ofdm_dcsk::{tx_baseband_fh_ofdm_dcsk_signal, tx_fh_ofdm_dcsk_signal};
use komunikilo::hadamard::HadamardMatrix;
use komunikilo::iter::Iter;
use komunikilo::{bit_to_nrz, Bit};
use num::Complex;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use rand::Rng;

#[macro_use]
mod util;

#[test]
fn baseband_plot() {
    let mut rng = rand::thread_rng();
    let num_bits = 50 * 9002;
    let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

    let tx: Vec<Complex<f64>> =
        tx_baseband_fh_ofdm_dcsk_signal(data_bits.iter().cloned()).collect();

    iq_plot!(tx, "/tmp/fh_ofdm_dcsk_baseband_IQ.png");
}

#[test]
fn python_plotz() -> PyResult<()> {
    let data: Vec<Bit> = vec![true, true, false, false, true, false, true];
    /*
    let data: Vec<Bit> = data
        .iter()
        .cloned()
        .chain(data.iter().cloned())
        .chain(data.iter().cloned())
        .chain(data.iter().cloned())
        .chain(data.iter().cloned())
        .chain(data.iter().cloned())
        .collect();
    */
    let samp_rate = 80_000; // Clock rate for both RX and TX.
    let symbol_rate = 1000; // Rate symbols come out the things.
    let carrier_freq = 2500_f64;
    let low_freq = 1000f64;
    let high_freq = 4000f64;

    let data_tx: Vec<f64> = data
        .iter()
        .cloned()
        .map(bit_to_nrz)
        .inflate(samp_rate / symbol_rate)
        .collect();

    let fh_ofdm_dcsk_tx: Vec<f64> = tx_fh_ofdm_dcsk_signal(
        data.iter().cloned(),
        low_freq,
        high_freq,
        8,
        samp_rate,
        symbol_rate,
    )
    .collect();

    // Compare to CDMA
    let h = HadamardMatrix::new(8);
    let key = h.key(2);
    let cdma_tx: Vec<f64> = tx_cdma_bpsk_signal(
        data.iter().cloned(),
        samp_rate,
        symbol_rate,
        carrier_freq,
        &key,
    )
    .collect();

    let t_step: f64 = 1f64 / (samp_rate as f64);

    Python::with_gil(|py| {
        let plt = py.import_bound("matplotlib.pyplot")?;
        let np = py.import_bound("numpy")?;
        let locals = [("np", np), ("plt", plt)].into_py_dict_bound(py);

        locals.set_item("data_tx", data_tx)?;
        locals.set_item("cdma_tx", cdma_tx)?;
        locals.set_item("fh_ofdm_dcsk_tx", fh_ofdm_dcsk_tx)?;
        locals.set_item("dt", t_step)?;

        let x = py.eval_bound("lambda s, dt: [dt * i for i in range(len(s))]", None, None)?;
        locals.set_item("x", x)?;

        let (fig, axes): (&PyAny, &PyAny) = py
            .eval_bound("plt.subplots(4)", None, Some(&locals))?
            .extract()?;
        locals.set_item("fig", fig)?;
        locals.set_item("axes", axes)?;

        for line in [
            "axes[0].plot(x(data_tx, dt), data_tx, label='DATA: [1 1 0 0 1]')",
            "axes[1].plot(x(cdma_tx, dt), cdma_tx, label='CDMA: (2.5kHz, 1kHz Data Rate)')",
            "axes[2].plot(x(fh_ofdm_dcsk_tx, dt), fh_ofdm_dcsk_tx, label='fh_ofdm_dcsk: 8kHz Chip Rate')",
        ] {
            py.eval_bound(line, None, Some(&locals))?;
        }

        locals.set_item("samp_rate", samp_rate)?;
        locals.set_item("freq", carrier_freq)?;

        for line in [
            "plt.psd(cdma_tx, Fs=samp_rate, Fc=freq)",
            "plt.psd(fh_ofdm_dcsk_tx, Fs=samp_rate, Fc=freq)",
            "[x.legend() for x in axes[:-1]]",
            "fig.set_size_inches(16, 9)",
            // "plt.show()",
            "fig.savefig('/tmp/fh_ofdm_dcsk_works.png', dpi=300)",
        ] {
            py.eval_bound(line, None, Some(&locals))?;
        }

        Ok(())
    })
}
