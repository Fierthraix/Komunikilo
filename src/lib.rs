#![allow(unused_macros, dead_code)]
use std::f64::consts::PI;

use num_complex::Complex;
use numpy::ndarray::Axis;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use smallvec::SmallVec;

mod bch;
pub mod bpf;
pub mod bpsk;
pub mod cdma;
pub mod chaos;
mod costas;
pub mod csk;
pub mod dcsk;
pub mod fh;
pub mod fh_css;
pub mod fh_ofdm_dcsk;
mod filters;
pub mod fm;
pub mod fsk;
pub mod hadamard;
pub mod iter;
pub mod ofdm;
pub mod qpsk;
pub mod ssca;
mod util;
mod willie;

use crate::bpsk::{rx_bpsk_signal, tx_baseband_bpsk_signal, tx_bpsk_signal};
use crate::cdma::tx_cdma_bpsk_signal;
use crate::csk::tx_baseband_csk;
use crate::dcsk::tx_baseband_dcsk;
use crate::fh_css::{linear_chirp, tx_fh_css_signal};
use crate::fh_ofdm_dcsk::tx_fh_ofdm_dcsk_signal_2;
use crate::fsk::tx_bfsk_signal;
use crate::hadamard::HadamardMatrix;
use crate::iter::Iter;
use crate::ofdm::tx_ofdm_qpsk_signal;
use crate::ssca::{ssca_base, ssca_mapped};

pub type Bit = bool;

fn bool_to_u8(bools: &[bool]) -> u8 {
    let mut out: u8 = 0x0;
    for (i, &bool) in bools.iter().enumerate().take(std::cmp::min(bools.len(), 8)) {
        if bool {
            out |= 1 << i
        }
    }
    out
}

fn u8_to_bools(data: u8) -> [bool; 8] {
    let mut out: [bool; 8] = [false; 8];
    let data: u8 = data;
    for (i, bool) in out.iter_mut().enumerate() {
        *bool = (data & (1 << i)) != 0
    }
    out
}

fn bools_to_u8s<I: Iterator<Item = Bit>>(bools: I) -> impl Iterator<Item = u8> {
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

fn u8s_to_bools<I: Iterator<Item = u8>>(bytes: I) -> impl Iterator<Item = bool> {
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

fn fftshift<T: Clone>(x: &[T]) -> Vec<T> {
    let mut v = Vec::with_capacity(x.len());
    let pivot = x.len().div_ceil(2);
    v.extend_from_slice(&x[pivot..]);
    v.extend_from_slice(&x[..pivot]);
    v
}

fn fftshift_mut<T: Copy>(x: &mut [T]) {
    let pivot = x.len().div_ceil(2);
    let v: Vec<T> = Vec::from(&x[..pivot]);
    for (i, j) in (pivot..x.len()).enumerate() {
        x[i] = x[j];
    }
    for (i, item) in (0..pivot).zip(v.into_iter()) {
        x[i] = item;
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

pub fn hamming_window(length: usize) -> Vec<f64> {
    (0..length)
        .map(|i| 0.54 - 0.46 * (2f64 * PI * i as f64 / length as f64).cos())
        .collect()
}

pub fn hamming_window_complex(length: usize) -> Vec<Complex<f64>> {
    (0..length)
        .map(|i| {
            Complex::new(
                0.54 - 0.46 * (2f64 * PI * i as f64 / length as f64).cos(),
                0f64,
            )
        })
        .collect()
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

pub fn awgn_complex<I: Iterator<Item = Complex<f64>>>(
    signal: I,
    sigma: f64,
) -> impl Iterator<Item = Complex<f64>> {
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

#[inline]
/// Calculates the energy per sample.
pub fn avg_energy(signal: &[f64]) -> f64 {
    signal.iter().map(|&sample| sample.powi(2)).sum::<f64>() / signal.len() as f64
}

#[pyfunction]
#[pyo3(name = "avg_energy")]
pub fn avg_energy_py(signal: Vec<f64>) -> f64 {
    signal.iter().map(|&sample| sample.powi(2)).sum::<f64>() / signal.len() as f64
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
        let sample_rate: f64 = steps as f64 / (stop - start);
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

#[pyfunction]
fn dcs_detector(signal: Vec<f64>) -> f64 {
    let np = 64;
    let n = 4096;
    let sx = ssca_base(signal, n, np);

    let top = sx.map(Complex::<f64>::norm_sqr).sum_axis(Axis(0));
    let bot = sx.row(0).map(Complex::<f64>::norm_sqr);

    let lambda = (top / bot)
        .into_iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    10f64 * lambda.log10()
}

#[pyfunction]
fn dcs_detector_sxf(sxf: PyReadonlyArray2<'_, Complex<f64>>) -> f64 {
    let sx = sxf.as_array();

    let top = sx.map(Complex::<f64>::norm_sqr).sum_axis(Axis(0));
    let bot = sx.row(0).map(Complex::<f64>::norm_sqr);

    let lambda = (top / bot)
        .into_iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    10f64 * lambda.log10()
}

#[pyfunction]
fn max_cut_detector(signal: Vec<f64>) -> f64 {
    let np = 64;
    let n = 4096;
    let lambda: f64 = ssca_base(signal, n, np)
        .iter()
        .map(|&s_i| s_i.norm_sqr())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    10f64 * lambda.log10()
}

#[pyfunction]
fn max_cut_detector_sxf(sxf: PyReadonlyArray2<'_, Complex<f64>>) -> f64 {
    let lambda = sxf
        .as_array()
        .iter()
        .map(|s_i| s_i.norm_sqr())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    10f64 * lambda.log10()
}

#[pyfunction]
#[pyo3(name = "ssca", signature=(s, n=4096, np=64, map_output=false))]
pub fn ssca_py(
    py: Python<'_>,
    s: Vec<f64>,
    n: usize,
    np: usize,
    map_output: bool,
) -> Bound<'_, PyArray2<Complex<f64>>> {
    if map_output {
        ssca_mapped(s, n, np).into_pyarray_bound(py)
    } else {
        ssca_base(s, n, np).into_pyarray_bound(py)
    }
}

#[pyfunction]
pub fn random_data(py: Python<'_>, num_bits: usize) -> Bound<'_, PyArray1<Bit>> {
    let mut rng = rand::thread_rng();
    PyArray1::from_iter_bound(py, (0..num_bits).map(|_| rng.gen::<Bit>()))
}

#[pyfunction]
fn energy_detector(signal: Vec<f64>) -> f64 {
    10f64 * (signal.into_iter().map(|s_i| s_i.powi(2)).sum::<f64>()).log10()
}

#[pyfunction]
#[pyo3(name = "awgn")]
fn awgn_py(signal: Vec<f64>, sigma: f64) -> Vec<f64> {
    signal
        .into_iter()
        .zip(
            Normal::new(0f64, sigma)
                .unwrap()
                .sample_iter(rand::thread_rng()),
        )
        .map(|(sample, noise)| sample + noise)
        .collect()
}

#[pyfunction]
fn pure_awgn(size: usize, sigma: f64) -> Vec<f64> {
    Normal::new(0f64, sigma)
        .unwrap()
        .sample_iter(rand::thread_rng())
        .take(size)
        .collect()
}

#[pyfunction]
fn chirp(chirp_rate: f64, sample_rate: usize, f0: f64, f1: f64) -> Vec<f64> {
    linear_chirp(chirp_rate, sample_rate, f0, f1).collect()
}

#[pymodule]
#[pyo3(name = "komunikilo")]
fn module_with_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    /// Transmit a baseband signal.
    #[pyfunction]
    fn tx_baseband_bpsk(message: Vec<Bit>) -> Vec<Complex<f64>> {
        tx_baseband_bpsk_signal(message.into_iter()).collect()
    }

    /// Transmit a BPSK signal.
    #[pyfunction]
    fn tx_bpsk(
        message: Vec<Bit>,
        sample_rate: usize,
        symbol_rate: usize,
        carrier_freq: f64,
    ) -> Vec<f64> {
        tx_bpsk_signal(message.into_iter(), sample_rate, symbol_rate, carrier_freq).collect()
    }

    #[pyfunction]
    fn rx_bpsk(
        signal: Vec<f64>,
        sample_rate: usize,
        symbol_rate: usize,
        carrier_freq: f64,
    ) -> Vec<Bit> {
        rx_bpsk_signal(signal.into_iter(), sample_rate, symbol_rate, carrier_freq).collect()
    }

    /// Transmit an OFDM QPSK signal.
    #[pyfunction]
    fn tx_ofdm(
        message: Vec<Bit>,
        subcarriers: usize,
        pilots: usize,
        sample_rate: usize,
        symbol_rate: usize,
        carrier_freq: f64,
    ) -> Vec<f64> {
        tx_ofdm_qpsk_signal(
            message.into_iter(),
            subcarriers,
            pilots,
            sample_rate,
            symbol_rate,
            carrier_freq,
        )
        .collect()
    }

    /// Transmit a CDMA BPSK signal.
    #[pyfunction]
    fn tx_cdma_bpsk(
        message: Vec<Bit>,
        sample_rate: usize,
        symbol_rate: usize,
        carrier_freq: f64,
    ) -> Vec<f64> {
        let walsh_codes = HadamardMatrix::new(16);
        let key: Vec<Bit> = walsh_codes.key(0).clone();
        tx_cdma_bpsk_signal(
            message.into_iter(),
            sample_rate,
            symbol_rate,
            carrier_freq,
            &key,
        )
        .collect()
    }

    #[pyfunction]
    fn tx_bfsk(
        message: Vec<Bit>,
        sample_rate: usize,
        symbol_rate: usize,
        freq_low: f64,
        freq_high: f64,
    ) -> Vec<f64> {
        tx_bfsk_signal(
            message.into_iter(),
            sample_rate,
            symbol_rate,
            freq_low,
            freq_high,
        )
        .collect()
    }

    #[pyfunction]
    fn tx_fh_css(
        message: Vec<Bit>,
        sample_rate: usize,
        symbol_rate: usize,
        freq_low: f64,
        freq_high: f64,
        num_freqs: usize,
    ) -> Vec<f64> {
        tx_fh_css_signal(
            message.into_iter(),
            sample_rate,
            symbol_rate,
            freq_low,
            freq_high,
            num_freqs,
        )
        .collect()
    }

    #[pyfunction]
    fn tx_csk(message: Vec<Bit>, sample_rate: usize, symbol_rate: usize) -> Vec<f64> {
        assert!(is_int(sample_rate as f64 / symbol_rate as f64));
        let samples_per_symbol: usize = sample_rate / symbol_rate;
        tx_baseband_csk(message.into_iter())
            .inflate(samples_per_symbol)
            .collect()
    }

    #[pyfunction]
    fn tx_dcsk(message: Vec<Bit>, sample_rate: usize, symbol_rate: usize) -> Vec<f64> {
        assert!(is_int(sample_rate as f64 / symbol_rate as f64));
        let samples_per_symbol: usize = sample_rate / symbol_rate;
        tx_baseband_dcsk(message.into_iter())
            .inflate(samples_per_symbol)
            .collect()
    }

    #[pyfunction]
    fn tx_fh_ofdm_dcsk(
        message: Vec<Bit>,
        sample_rate: usize,
        symbol_rate: usize,
        carrier_freq: f64,
    ) -> Vec<f64> {
        let res =
            tx_fh_ofdm_dcsk_signal_2(message.into_iter(), sample_rate, symbol_rate, carrier_freq)
                .collect();
        println!("RUST: FHOFDMDCSK symbols ready for python");
        res
    }

    m.add_function(wrap_pyfunction!(tx_baseband_bpsk, m)?)?;
    m.add_function(wrap_pyfunction!(tx_bpsk, m)?)?;
    m.add_function(wrap_pyfunction!(rx_bpsk, m)?)?;
    m.add_function(wrap_pyfunction!(tx_cdma_bpsk, m)?)?;
    m.add_function(wrap_pyfunction!(tx_ofdm, m)?)?;
    m.add_function(wrap_pyfunction!(tx_csk, m)?)?;
    m.add_function(wrap_pyfunction!(tx_dcsk, m)?)?;
    m.add_function(wrap_pyfunction!(tx_bfsk, m)?)?;
    m.add_function(wrap_pyfunction!(tx_fh_css, m)?)?;
    m.add_function(wrap_pyfunction!(tx_fh_ofdm_dcsk, m)?)?;
    m.add_function(wrap_pyfunction!(random_data, m)?)?;
    m.add_function(wrap_pyfunction!(awgn_py, m)?)?;
    m.add_function(wrap_pyfunction!(pure_awgn, m)?)?;
    m.add_function(wrap_pyfunction!(chirp, m)?)?;
    m.add_function(wrap_pyfunction!(energy_detector, m)?)?;
    m.add_function(wrap_pyfunction!(avg_energy_py, m)?)?;
    m.add_function(wrap_pyfunction!(ssca_py, m)?)?;
    m.add_function(wrap_pyfunction!(max_cut_detector, m)?)?;
    m.add_function(wrap_pyfunction!(max_cut_detector_sxf, m)?)?;
    m.add_function(wrap_pyfunction!(dcs_detector, m)?)?;
    m.add_function(wrap_pyfunction!(dcs_detector_sxf, m)?)?;
    Ok(())
}
