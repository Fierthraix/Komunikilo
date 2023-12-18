use crate::iter::Iter;
use num::complex::Complex;
use rand::rngs::ThreadRng;
use rand_distr::{Distribution, Normal};

mod bch;
pub mod bpsk;
pub mod cdma;
pub mod chaos;
mod costas;
pub mod csk;
pub mod dcsk;
mod fh_ofdm_dcsk;
mod filters;
pub mod fm;
pub mod hadamard;
pub mod iter;
pub mod qpsk;
mod turbo;
mod util;

pub type Bit = bool;

fn bool_to_u8(bools: &[bool]) -> u8 {
    let mut out: u8 = 0x0;
    for i in 0..std::cmp::min(bools.len(), 8) {
        if bools[i] {
            out |= 1 << i
        }
    }
    out
}

fn u8_to_bools(data: u8) -> [bool; 8] {
    let mut out: [bool; 8] = [false; 8];
    let mut data: u8 = data;
    for i in 0..8 {
        out[i] = (data & (1 << i)) != 0
    }
    out
}

fn bools_to_u8s<I>(bools: I) -> impl Iterator<Item = u8>
where
    I: Iterator<Item = Bit>,
{
    bools.chunks(8).map(|chunk| {
        // Pad the last chunk, if neccessary.
        if chunk.len() != 8 {
            let mut last_chunk = Vec::with_capacity(8);
            last_chunk.extend_from_slice(&chunk);

            while !last_chunk.len() < 8 {
                last_chunk.push(false);
            }
            bool_to_u8(&last_chunk)
        } else {
            bool_to_u8(&chunk)
        }
    })
}

fn u8s_to_bools<I>(bytes: I) -> impl Iterator<Item = bool>
where
    I: Iterator<Item = u8>,
{
    bytes.flat_map(u8_to_bools)
}

fn is_int(num: f64) -> bool {
    num == (num as u64) as f64
}

#[inline]
pub fn db(x: f64) -> f64 {
    10f64 * x.log10()
}

#[inline]
pub fn undb(x: f64) -> f64 {
    10f64.powf(x / 10f64)
}

#[inline]
pub fn linspace(start: f64, stop: f64, num: usize) -> impl Iterator<Item = f64> {
    let step = (stop - start) / (num as f64);
    (0..num).map(move |i| start + step * (i as f64))
}

#[inline]
pub fn bit_to_nrz(bit: Bit) -> f64 {
    if bit {
        1_f64
    } else {
        -1_f64
    }
}

#[inline]
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

#[inline]
pub fn erfc(x: f64) -> f64 {
    1f64 - erf(x)
}

macro_rules! ber {
    ($tx:expr, $rx:expr) => {
        let len = std::cmp::min(tx, rx);
        let total = tx
            .iter()
            .zip(rx.iter())
            .map(|(ti, ri)| if t1 == r1 { 0 } else { 1 })
            .sum();
        (total as f64) / (len as f64)
    };
}

#[inline]
pub fn ber(tx: &[Bit], rx: &[Bit]) -> f64 {
    let len: usize = std::cmp::min(tx.len(), rx.len());
    let errors: usize = tx
        .iter()
        .zip(rx.iter())
        .map(|(&ti, &ri)| if ti == ri { 0 } else { 1 })
        .sum();
    (errors as f64) / (len as f64)
}

/*
pub fn ber<'a, I, T>(tx: I, rx: I) -> f64
where
    I: Iterator<Item = &'a T> + ExactSizeIterator,
    T: Eq + 'a,
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

pub fn awgn<I: Iterator<Item = f64>>(signal: I, sigma: f64) -> impl Iterator<Item = f64> {
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
    pub fn awgn<'a, I>(&'a mut self, signal: I) -> impl Iterator<Item = Complex<f64>> + '_
    where
        I: Iterator<Item = Complex<f64>> + 'a,
    {
        signal
            .zip(self.dist.sample_iter(&mut self.rng))
            .map(|(sample, noise)| sample + noise)
    }
}

pub fn energy<I: Iterator<Item = f64>>(signal: I, sample_rate: f64) -> f64 {
    signal.map(|sample| sample.powi(2)).sum::<f64>() / sample_rate
}

pub fn power<I: Iterator<Item = f64>>(signal: I, sample_rate: f64) -> f64 {
    let (count, power) = signal.enumerate().fold((0, 0f64), |acc, (idx, sample)| {
        (idx, acc.1 + sample.powi(2))
    });

    power / (count as f64) / sample_rate
}

#[cfg(test)]
mod tests {

    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use rand::Rng;
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
    fn bitstream_conversions() {
        let mut rng = rand::thread_rng();
        let num_bits = 33; // Ensure there will be padding.
        let start_data: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

        let u8s: Vec<u8> = bools_to_u8s(start_data.iter().cloned()).collect();
        assert_eq!(u8s.len(), 40 / 8); // Check for padding as well...

        let bits: Vec<Bit> = u8s_to_bools(u8s.iter().cloned()).collect();
        assert_eq!(start_data, bits[..num_bits])
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
