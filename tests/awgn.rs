use itertools::iproduct;
use komunikilo::{avg_energy, awgn, linspace};
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use std::collections::HashMap;
use std::iter::repeat;

#[macro_use]
mod util;

#[test]
fn check_n0_and_energy_expectation() {
    /*
     * GOAL: Plot the `avg_energy` of AWGN at different sampling rates and N0 values.
     */

    let sample_rates = [44100, 100_000, 2_000_000, 10_000_000];

    let seconds = 1;

    let n0s: Vec<f64> = linspace(0.01, 20f64, 150).collect();

    let mut nrgs: HashMap<usize, Vec<f64>> = {
        let mut nrgs = HashMap::new();
        for samp_rate in sample_rates {
            nrgs.insert(samp_rate, vec![]);
        }
        nrgs
    };

    for (sample_rate, n0) in iproduct!(sample_rates, n0s.clone()) {
        let sig: Vec<f64> = awgn(repeat(0f64).take(seconds * sample_rate), n0).collect();
        let nrg = avg_energy(&sig).sqrt();
        if let Some(v) = nrgs.get_mut(&sample_rate) {
            v.push(nrg);
        }
    }

    for (samp_rate, nrg_vec) in nrgs.iter() {
        dot_plot!(n0s.clone(), nrg_vec, format!("/tmp/awgn_{}.png", samp_rate));
    }

    /*
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
