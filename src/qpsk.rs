use crate::convolution::convolve2;
use crate::{bit_to_nrz, inflate::InflateIt, Bit};
use itertools::Itertools;
use num::complex::Complex;
use std::f64::consts::PI;

pub fn tx_baseband_qpsk_signal<I: Iterator<Item = Bit>>(
    message: I,
) -> impl Iterator<Item = Complex<f64>> {
    message.tuples().map(|(bit1, bit2)| match (bit1, bit2) {
        (true, true) => Complex::new(2f64.sqrt(), 2f64.sqrt()),
        (true, false) => Complex::new(2f64.sqrt(), -(2f64.sqrt())),
        (false, true) => Complex::new(-(2f64.sqrt()), 2f64.sqrt()),
        (false, false) => Complex::new(-(2f64.sqrt()), -(2f64.sqrt())),
    })
}

pub fn rx_baseband_qpsk_signal<I: Iterator<Item = Complex<f64>>>(
    message: I,
) -> impl Iterator<Item = Bit> {
    message.flat_map(|sample| [sample.re >= 0f64, sample.im >= 0f64].into_iter())
}

pub fn tx_qpsk_signal<I: Iterator<Item = Bit>>(
    message: I,
    sample_rate: usize,
    symbol_rate: usize,
    carrier_freq: f64,
    start_time: f64,
) -> impl Iterator<Item = f64> {
    let samples_per_symbol: usize = sample_rate / symbol_rate;
    let t_step: f64 = 1_f64 / (samples_per_symbol as f64);
    message
        .tuples()
        .inflate(samples_per_symbol)
        .enumerate()
        .map(move |(idx, (bit1, bit2))| {
            // println!("{}: {}  {}", idx, bit1, bit2);
            let time = (idx as f64) * t_step + start_time;
            let i_t = bit_to_nrz(bit1) * (2f64 * PI * carrier_freq * time).cos();
            let q_t = -bit_to_nrz(bit2) * (2f64 * PI * carrier_freq * time).sin();

            i_t + q_t
        })
}

pub fn rx_qpsk_signal<I: Iterator<Item = f64>>(
    message: I,
    sample_rate: usize,
    symbol_rate: usize,
    carrier_freq: f64,
    start_time: f64,
) -> impl Iterator<Item = Bit> {
    let samples_per_symbol: usize = sample_rate / symbol_rate;
    let t_step: f64 = 1_f64 / (samples_per_symbol as f64);
    let filter: Vec<f64> = (0..samples_per_symbol).map(|_| 1f64).collect();

    // Split into two branches for I and Q, and output two bits at once.
    let real_demod = message.enumerate().map(move |(idx, sample)| {
        let time = start_time + (idx as f64) * t_step;
        let ii = sample * (2_f64 * PI * carrier_freq * time).cos();
        let qi = sample * -(2_f64 * PI * carrier_freq * time).sin();

        (ii, qi)
        // vec![ii, qi]
    });

    convolve2(real_demod, filter)
        // nonvolve(2, real_demod, filter)
        .enumerate()
        .filter_map(move |(i, val)| {
            if i % samples_per_symbol == 0 {
                Some(val)
            } else {
                None
            }
        })
        .flat_map(|(val1, val2)| [val1 >= 0f64, val2 >= 0f64].into_iter())
        // .flat_map(|valz| valz.iter().map(|val| val >= 0f64))
        .skip(2)
}

#[cfg(test)]
mod tests {

    use super::*;
    extern crate itertools;
    extern crate plotpy;
    extern crate rand;
    extern crate rand_distr;
    use crate::qpsk::tests::rand::Rng;

    #[test]
    fn baseband_qpsk() {
        let mut rng = rand::thread_rng();
        let num_bits = 9002;
        let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

        let qpsk_tx: Vec<Complex<f64>> =
            tx_baseband_qpsk_signal(data_bits.iter().cloned()).collect();

        let qpsk_rx: Vec<Bit> = rx_baseband_qpsk_signal(qpsk_tx.iter().cloned()).collect();

        assert_eq!(data_bits, qpsk_rx);
    }

    #[test]
    fn full_qpsk() {
        // let num_bits = 9002; // How many bits to transmit overall.
        let num_bits = 20; // How many bits to transmit overall.
        let samp_rate = 44100; // Clock rate for both RX and TX.
        let symbol_rate = 900; // Rate symbols come out the things.
        let carrier_freq = 1800_f64;

        let mut rng = rand::thread_rng();
        let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

        let qpsk_tx: Vec<f64> = tx_qpsk_signal(
            data_bits.iter().cloned(),
            samp_rate,
            symbol_rate,
            carrier_freq,
            0f64,
        )
        .collect();

        let qpsk_rx: Vec<Bit> = rx_qpsk_signal(
            qpsk_tx.iter().cloned(),
            samp_rate,
            symbol_rate,
            carrier_freq,
            0f64,
        )
        .collect();

        assert_eq!(data_bits, qpsk_rx);
    }
}
