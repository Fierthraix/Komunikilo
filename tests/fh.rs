use komunikilo::fh::tx_fh_bpsk_signal;
use komunikilo::Bit;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;

#[macro_use]
mod util;

const DATA: [Bit; 8] = [true, true, false, false, true, false, true, false];

#[test]
fn fh_works() {
    let sample_rate = 128_000; // Clock rate for both RX and TX.
    let symbol_rate = 100; // Rate symbols come out the things.
    let carrier_freq = 1e3;

    let low = 2e3;
    let high = 3e3;
    let num_freqs = 8;

    let fh_signal: Vec<f64> = tx_fh_bpsk_signal(
        DATA.iter().cloned(),
        low,
        high,
        num_freqs,
        sample_rate,
        symbol_rate,
    )
    .collect();

    let time: Vec<f64> = (0..fh_signal.len())
        .map(|i| i as f64 / sample_rate as f64)
        .collect();

    plot!(time, fh_signal, "/tmp/fh_bpsk_signal.png");

    Python::with_gil(|py| {
        let plt = py.import_bound("matplotlib.pyplot")?;
        let np = py.import_bound("numpy")?;
        let locals = [("np", np), ("plt", plt)].into_py_dict_bound(py);

        locals.set_item("fh_signal", fh_signal)?;
        locals.set_item("time", time)?;

        let (fig, axes): (&PyAny, &PyAny) = py
            .eval_bound("plt.subplots(1)", None, Some(&locals))?
            .extract()?;
        locals.set_item("fig", fig)?;
        locals.set_item("axes", axes)?;

        /*
        for line in [
            "axes[0].plot(x(data_tx, dt), data_tx, label='DATA: [1 1 0 0 1]')",
            "axes[1].plot(x(bpsk_tx, dt), bpsk_tx, label='BPSK: (2.5kHz, 1kHz Data Rate)')",
            "axes[2].plot(x(cdma_tx, dt), cdma_tx, label='CDMA: 8kHz Chip Rate')",
        ] {
            py.eval_bound(line, None, Some(&locals))?;
        }
        */

        locals.set_item("samp_rate", sample_rate)?;
        locals.set_item("freq", carrier_freq)?;

        for line in [
            "plt.psd(fh_signal, Fs=samp_rate) #, Fc=freq)",
            // "plt.psd(fh_signal[:int(samp_rate/100)], Fs=samp_rate) #, Fc=freq)",
            // "plt.psd(fh_signal[int(samp_rate/100):2*int(samp_rate/100)], Fs=samp_rate, Fc=freq)",
            // "plt.psd(cdma_tx, Fs=samp_rate, Fc=freq)",
            // "[x.legend() for x in axes[:-1]]",
            "fig.set_size_inches(16, 9)",
            // "plt.show()",
            "fig.savefig('/tmp/fh_works.png', dpi=300)",
        ] {
            py.eval_bound(line, None, Some(&locals))?;
        }

        PyResult::Ok(())
    })
    .unwrap();
}
