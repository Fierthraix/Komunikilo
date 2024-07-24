use crate::{is_int, iter::Iter, Bit};
use std::f64::consts::PI;

pub fn tx_bfsk_signal<I: Iterator<Item = Bit>>(
    message: I,
    sample_rate: usize,
    symbol_rate: usize,
    freq_low: f64,
    freq_high: f64,
) -> impl Iterator<Item = f64> {
    let samples_per_symbol: usize = sample_rate / symbol_rate;
    assert!(sample_rate / 2 >= freq_high as usize);
    assert!(is_int(sample_rate as f64 / symbol_rate as f64));

    message
        .map(move |bit| if bit { freq_high } else { freq_low })
        .inflate(samples_per_symbol)
        .enumerate()
        .map(move |(idx, freq)| {
            let time = idx as f64 / sample_rate as f64;
            (2f64 * PI * freq * time).cos()
        })
}

pub fn tx_mfsk_signal<I: Iterator<Item = Bit>>(
    message: I,
    sample_rate: usize,
    symbol_rate: usize,
    freq_low: f64,
    freq_high: f64,
    _num_freqs: usize,
) -> impl Iterator<Item = f64> {
    let samples_per_symbol: usize = sample_rate / symbol_rate;
    assert!(sample_rate / 2 >= freq_high as usize);
    assert!(is_int(sample_rate as f64 / symbol_rate as f64));
    // TODO: FIXME: evenly divide the space by number of freqs,

    message
        .map(move |bit| if bit { freq_high } else { freq_low })
        .inflate(samples_per_symbol)
        .enumerate()
        .map(move |(idx, freq)| {
            let time = idx as f64 / sample_rate as f64;
            (2f64 * PI * freq * time).cos()
        })
}

pub fn tx_fsk_signal<I: Iterator<Item = Bit>>(
    message: I,
    sample_rate: usize,
    symbol_rate: usize,
    freqs: &[f64],
) -> impl Iterator<Item = f64> {
    let samples_per_symbol: usize = sample_rate / symbol_rate;
    // TODO: FIXME: Read from list of freqs.
    let freq_high = freqs[0];
    let freq_low = freqs[1];
    assert!(sample_rate / 2 >= freq_high as usize);
    assert!(is_int(sample_rate as f64 / symbol_rate as f64));
    message
        .map(move |bit| if bit { freq_high } else { freq_low })
        .inflate(samples_per_symbol)
        .enumerate()
        .map(move |(idx, freq)| {
            let time = idx as f64 / sample_rate as f64;
            (2f64 * PI * freq * time).cos()
        })
}
fn rx_bfsk_signal<I: Iterator<Item = f64>>(
    _signal: I,
    _sample_rate: usize,
    _symbol_rate: usize,
    _freq_low: f64,
    _freq_high: f64,
) -> impl Iterator<Item = Bit> {
    [true].into_iter()
}
