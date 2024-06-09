use crate::{fh::HopTable, is_int, linspace, Bit};
use num::traits::Inv;
use std::f64::consts::PI;

const SEED: u64 = 64;

pub fn linear_chirp(
    chirp_rate: f64,
    sample_rate: usize,
    f0: f64,
    f1: f64,
) -> impl Iterator<Item = f64> {
    let chirp_period: f64 = chirp_rate.inv();
    let k: f64 = (f1 - f0) / chirp_period;
    let samps_per_chirp: usize = (sample_rate as f64 / chirp_rate) as usize;

    linspace(0f64, chirp_period, samps_per_chirp)
        .map(move |t_i| (2f64 * PI * (f0 + k * t_i / 2f64) * t_i).cos())
}

pub fn linear_upchirp_centered(
    chirp_rate: f64,
    center_freq: f64,
    bandwidth: f64,
    sample_rate: usize,
) -> impl Iterator<Item = f64> {
    let f0: f64 = center_freq - bandwidth / 2f64;
    let f1: f64 = center_freq + bandwidth / 2f64;
    linear_chirp(chirp_rate, sample_rate, f0, f1)
}
pub fn linear_downchirp_centered(
    chirp_rate: f64,
    center_freq: f64,
    bandwidth: f64,
    sample_rate: usize,
) -> impl Iterator<Item = f64> {
    let f0: f64 = center_freq - bandwidth / 2f64;
    let f1: f64 = center_freq + bandwidth / 2f64;
    linear_chirp(chirp_rate, sample_rate, f0, f1)
}

pub fn tx_fh_css_signal<I: Iterator<Item = Bit>>(
    message: I,
    sample_rate: usize,
    symbol_rate: usize,
    low_freq: f64,
    high_freq: f64,
    num_freqs: usize,
) -> impl Iterator<Item = f64> {
    assert!(sample_rate / 2 >= high_freq as usize);
    assert!(is_int(sample_rate as f64 / symbol_rate as f64));

    let bandwidth: f64 = (high_freq - low_freq) / num_freqs as f64;

    message
        .zip(HopTable::new(low_freq, high_freq, num_freqs, SEED))
        .flat_map(move |(bit, hop_freq)| {
            let (f0, f1) = if bit {
                (hop_freq - bandwidth / 2f64, hop_freq + bandwidth / 2f64)
            } else {
                (hop_freq + bandwidth / 2f64, hop_freq - bandwidth / 2f64)
            };
            linear_chirp(symbol_rate as f64, sample_rate, f0, f1)
        })
}
