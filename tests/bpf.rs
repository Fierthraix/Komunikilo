use komunikilo::bpf::bandpass;
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
    let filtered_signal: Vec<f64> =
        bandpass(f1 - eps, f1 + eps, signal.iter().cloned(), sample_rate).collect();

    plot!(time[..500], signal[..500], "/tmp/bpf_before.png");
    plot!(time[..500], filtered_signal[..500], "/tmp/bpf_after.png");
    plot!(
        time[..1000],
        s1[..1000],
        filtered_signal[..1000],
        "/tmp/bpf_signal_before_and_after.png"
    );
}
