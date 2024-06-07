use crate::linspace;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;

fn normaltest(signal: &[f64]) -> f64 {
    Python::with_gil(|py| {
        let scipy_stats = py.import_bound("scipy.stats")?;
        let normtest: f64 = scipy_stats
            .getattr("normaltest")?
            .call1((Vec::from(signal),))?
            .getattr("pvalue")?
            .extract()?;

        Ok::<f64, PyErr>(normtest)
    })
    .unwrap()
}

fn normal_detector(signal: &[f64], alpha: f64) -> bool {
    normaltest(signal) > alpha
}

fn energy_detector(signal: &[f64], n0: f64) -> bool {
    let energy: f64 = signal.iter().map(|&s_i| s_i.powi(2)).sum::<f64>() / signal.len() as f64;
    energy.sqrt() > n0
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use std::f64::consts::PI;
    extern crate rand;
    extern crate rand_distr;
    use crate::willie::tests::rand_distr::{Distribution, Normal};

    #[test]
    fn normtest_awgn() {
        let num_samples = 100_000;
        let awgn_signal: Vec<f64> = Normal::new(0f64, 5f64)
            .unwrap()
            .sample_iter(rand::thread_rng())
            .take(num_samples)
            .collect();
        let norm_test: f64 = normaltest(&awgn_signal);
        assert!(norm_test > 0.05);
    }

    #[test]
    fn normtest_sine() {
        let freq = 1000f64;
        let num_samples = 10_000;
        let sine_wave: Vec<f64> = (0..num_samples)
            .map(|i| (2f64 * PI * freq * i as f64).sin())
            .collect();
        let sine_test: f64 = normaltest(&sine_wave);
        assert_approx_eq!(sine_test, 0f64);
    }

    #[test]
    fn energy_detector_test_awgn() {
        let num_samples = 100_000;
        let n0 = 6f64;
        let awgn_signal: Vec<f64> = Normal::new(0f64, n0)
            .unwrap()
            .sample_iter(rand::thread_rng())
            .take(num_samples)
            .collect();

        let energy: f64 = (awgn_signal.iter().map(|&s_i| s_i.powi(2)).sum::<f64>()
            / awgn_signal.len() as f64)
            .sqrt();
        assert_approx_eq!(energy, n0, 1f64);
    }
}
