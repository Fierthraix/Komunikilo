use komunikilo::bpf::{bandpass, butterpass};
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use std::f64::consts::PI;

#[macro_use]
mod util;

const FFT_LEN: usize = 8192;

#[test]
fn it_works() {
    let sample_rate = 10_000;
    let t_step = 1f64 / sample_rate as f64;
    let num_samples = FFT_LEN * 5;

    let time: Vec<f64> = (0..num_samples).map(|i| i as f64 * t_step).collect();

    let f1 = 100f64;
    let f2 = 50f64;
    let f3 = 150f64;

    let s1: Vec<f64> = time.iter().map(|&t| (2f64 * PI * f1 * t).cos()).collect();

    let signal: Vec<f64> = time
        .iter()
        .map(|&t| {
            [f1, f2, f3]
                .iter()
                .map(|f_i| (2f64 * PI * f_i * t).cos())
                .sum::<f64>()
        })
        .collect();

    let eps = 20f64;
    const LEN: usize = 16000;

    let fft_filtered_signal: Vec<f64> = bandpass(
        f1 - eps,
        f1 + eps,
        signal.iter().cloned(),
        sample_rate,
        FFT_LEN,
    )
    .collect();

    let butter_filtered_signal: Vec<f64> =
        butterpass(f1 - eps, f1 + eps, signal.iter().cloned(), sample_rate).collect();

    let regression = |x1: &[f64], x2: &[f64]| -> Vec<f64> {
        x1.iter()
            .zip(x2.iter())
            .map(|(&x1i, &x2i)| (x1i - x2i).abs())
            .collect()
    };

    let fft_least_squares: f64 = regression(&s1, &fft_filtered_signal)
        .iter()
        .map(|si| si.powi(2))
        .sum();
    let butter_least_squares: f64 = regression(&s1, &butter_filtered_signal)
        .iter()
        .map(|si| si.powi(2))
        .sum();

    assert!(fft_least_squares < butter_least_squares);

    plot!(time[..500], signal[..500], "/tmp/bpf_signal.png");

    Python::with_gil(|py| {
        let locals = init_matplotlib!(py);

        let fft_regres: Vec<f64> = regression(&s1, &fft_filtered_signal);
        let butter_regres: Vec<f64> = regression(&s1, &butter_filtered_signal);

        locals.set_item("time", &time[..LEN]).unwrap();
        locals.set_item("s1", &s1[..LEN]).unwrap();
        locals
            .set_item("fft_filtered_signal", &fft_filtered_signal[..LEN])
            .unwrap();
        locals
            .set_item("butter_filtered_signal", &butter_filtered_signal[..LEN])
            .unwrap();
        locals.set_item("fft_regres", &fft_regres[..LEN]).unwrap();
        locals
            .set_item("butter_regres", &butter_regres[..LEN])
            .unwrap();

        let (fig, axes): (&PyAny, &PyAny) = py
            .eval_bound("plt.subplots(2)", None, Some(&locals))
            .unwrap()
            .extract()
            .unwrap();
        locals.set_item("fig", fig).unwrap();
        locals.set_item("axes", axes).unwrap();
        py.eval_bound("fig.set_size_inches(16, 9)", None, Some(&locals))
            .unwrap();
        for line in [
            "axes[0].plot(time, s1, label='Orignal Signal')",
            "axes[0].plot(time, fft_filtered_signal, label='FFT_Filtered')",
            "axes[0].plot(time, butter_filtered_signal, label='Butterworth Filter')",
            "axes[0].legend()",
            "axes[1].plot(time, fft_regres, label='FFT Regression')",
            "axes[1].plot(time, butter_regres, label='Butterworth Regression')",
            "axes[1].legend()",
            &format!(
                "fig.savefig('{}')",
                "/tmp/bpf_butter_and_fft_signal_before_and_after.png"
            ),
            "plt.close('all')",
        ] {
            py.eval_bound(line, None, Some(&locals)).unwrap();
        }
    })
}
