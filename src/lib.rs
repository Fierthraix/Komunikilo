use num::complex::Complex;
use rand::rngs::ThreadRng;
use rand_distr::{Distribution, Normal};

pub mod bpsk;
pub mod cdma;
mod chunks;
mod convolution;
mod costas;
mod filters;
mod hadamard;
pub mod inflate;
pub mod qpsk;

pub type Bit = bool;

pub fn db(x: f64) -> f64 {
    10f64 * x.log10()
}

pub fn undb(x: f64) -> f64 {
    10f64.powf(x / 10f64)
}

pub fn linspace(start: f64, stop: f64, num: usize) -> impl Iterator<Item = f64> {
    let step = (stop - start) / (num as f64);
    (0..num).map(move |i| start + step * (i as f64))
}

pub fn bit_to_nrz(bit: Bit) -> f64 {
    if bit {
        1_f64
    } else {
        -1_f64
    }
}

pub fn erf(x: f64) -> f64 {
    let t: f64 = 1f64 / (1f64 + 0.5 * x.abs());
    let tau = t
        * (-x.powi(2) - 1.26551223
            + 1.00002368 * t
            + 0.37409196 * t.powi(2)
            + 0.09678418 * t.powi(3)
            - 0.18628806 * t.powi(4)
            + 0.27886807 * t.powi(5)
            - 1.13520398 * t.powi(6)
            + 1.48851587 * t.powi(7)
            - 0.82215223 * t.powi(8)
            + 0.17087277 * t.powi(9))
        .exp();
    if x >= 0f64 {
        1f64 - tau
    } else {
        tau - 1f64
    }
}

pub fn erfc(x: f64) -> f64 {
    1f64 - erf(x)
}

/*
pub fn ber<I, T>(tx: I, rx: I) -> f64
where
    I: Iterator<Item = T>,
    T: Eq,
{
    let num_samples = tx.len();
    let num_errors = tx
        .zip(rx)
        .enumerate()
        .map(|(idx, (tx, rx))| if tx == rx { 1 } else { 0 })
        .sum();
}
*/

pub fn awgn_complex<I>(signal: I, sigma: f64) -> impl Iterator<Item = Complex<f64>>
where
    I: Iterator<Item = Complex<f64>>,
{
    signal
        .zip(
            Normal::new(0f64, sigma)
                .unwrap()
                .sample_iter(rand::thread_rng()),
        )
        .map(|(sample, noise)| sample + noise)
}

pub fn awgn<I>(signal: I, sigma: f64) -> impl Iterator<Item = f64>
where
    I: Iterator<Item = f64>,
{
    signal
        .zip(
            Normal::new(0f64, sigma)
                .unwrap()
                .sample_iter(rand::thread_rng()),
        )
        .map(|(sample, noise)| sample + noise)
}

pub struct Awgn {
    rng: ThreadRng,
    dist: Normal<f64>,
}

impl Awgn {
    pub fn new(sigma: f64) -> Self {
        Self {
            rng: rand::thread_rng(),
            dist: Normal::new(0_f64, sigma).unwrap(),
        }
    }
    //fn noise(&mut self) -> f64 {
    //self.dist.sample_iter
    //}
    pub fn awgn<'a, I>(&'a mut self, signal: I) -> impl Iterator<Item = Complex<f64>> + '_
    where
        I: Iterator<Item = Complex<f64>> + 'a,
    {
        signal
            .zip(self.dist.sample_iter(&mut self.rng))
            .map(|(sample, noise)| sample + noise)
    }
}

pub fn energy<I>(signal: I, sample_rate: f64) -> f64
where
    I: Iterator<Item = f64>,
{
    signal.map(|sample| sample.powi(2)).sum::<f64>() / sample_rate
}

pub fn power<I>(signal: I, sample_rate: f64) -> f64
where
    I: Iterator<Item = f64>,
{
    let (count, power) = signal.enumerate().fold((0, 0f64), |acc, (idx, sample)| {
        (idx, acc.1 + sample.powi(2))
    });

    power / (count as f64) / sample_rate
}

#[cfg(test)]
mod tests {

    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use std::f64::consts::PI;

    #[test]
    fn it_works() {
        let fs = 44100;
        let baud = 900;
        let nbits = 4000;
        let _f0 = 1800;
        let ns = fs / baud;
        let _n = nbits * ns;
    }

    #[test]
    fn energy_test() {
        let fc: f64 = 100f64;

        let start = 0f64;
        let stop = 1f64;
        let steps: usize = 1000;
        let sample_rate: f64 = steps as f64 / (stop - start) as f64;
        let t: Vec<f64> = linspace(start, stop, steps).collect();
        let sinx: Vec<f64> = t.iter().map(|&t| (2f64 * PI * fc * t).sin()).collect();
        let _sinx2: Vec<f64> = t
            .iter()
            .map(|&t| (2f64 * PI * fc * t).sin().powi(2))
            .collect();

        let e_sinx = energy(sinx.iter().cloned(), sample_rate);
        let p_sinx = power(sinx.iter().cloned(), sample_rate);
        assert_approx_eq!(e_sinx, 0.5);
        assert_approx_eq!(p_sinx, 5e-4);
    }
}
