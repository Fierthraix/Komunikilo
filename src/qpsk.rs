use crate::convolution::convolve2;
use crate::{bit_to_nrz, inflate, Bit};
use itertools::Itertools;
use num::complex::Complex;
use std::f64::consts::PI;

pub fn tx_baseband_qpsk_signal<I>(message: I) -> impl Iterator<Item = Complex<f64>>
where
    I: Iterator<Item = Bit>,
{
    message.tuples().map(|(bit1, bit2)| match (bit1, bit2) {
        (true, true) => Complex::new(2f64.sqrt(), 2f64.sqrt()),
        (true, false) => Complex::new(2f64.sqrt(), -(2f64.sqrt())),
        (false, true) => Complex::new(-(2f64.sqrt()), 2f64.sqrt()),
        (false, false) => Complex::new(-(2f64.sqrt()), -(2f64.sqrt())),
    })
}

pub fn rx_baseband_qpsk_signal<I>(message: I) -> impl Iterator<Item = Bit>
where
    I: Iterator<Item = Complex<f64>>,
{
    message.flat_map(|sample| [sample.re >= 0f64, sample.im >= 0f64].into_iter())
}

pub fn tx_qpsk_signal<I>(
    message: I,
    sample_rate: usize,
    symbol_rate: usize,
    carrier_freq: f64,
    start_time: f64,
) -> impl Iterator<Item = Complex<f64>>
where
    I: Iterator<Item = Bit>,
{
    let samples_per_symbol: usize = sample_rate / symbol_rate;
    let t_step: f64 = 1_f64 / (samples_per_symbol as f64);
    inflate(message.tuples(), samples_per_symbol)
        .enumerate()
        .map(move |(idx, (bit1, bit2))| {
            println!("{}: {}  {}", idx, bit1, bit2);
            let time = (idx as f64) * t_step + start_time;
            let i_t = bit_to_nrz(bit1) * (2f64 * PI * carrier_freq * time).cos();
            let q_t = -bit_to_nrz(bit2) * (2f64 * PI * carrier_freq * time).sin();

            Complex::<f64>::new(i_t + q_t, 0f64)
        })
}

pub fn rx_qpsk_signal<I>(
    message: I,
    sample_rate: usize,
    symbol_rate: usize,
    carrier_freq: f64,
    start_time: f64,
) -> impl Iterator<Item = Bit>
where
    I: Iterator<Item = Complex<f64>>,
{
    let samples_per_symbol: usize = sample_rate / symbol_rate;
    let t_step: f64 = 1_f64 / (samples_per_symbol as f64);
    let filter: Vec<f64> = (0..samples_per_symbol).map(|_| 1f64).collect();

    // Split into two branches for I and Q, and output two bits at once.
    let real_demod = message.enumerate().map(move |(idx, sample)| {
        let time = start_time + (idx as f64) * t_step;
        let ii = sample * (2_f64 * PI * carrier_freq * time).cos();
        let qi = sample * -(2_f64 * PI * carrier_freq * time).sin();

        (ii.re, qi.re)
    });

    convolve2(real_demod, filter)
        .enumerate()
        .filter_map(move |(i, val)| {
            if i % samples_per_symbol == 0 {
                Some(val)
            } else {
                None
            }
        })
        .flat_map(|(val1, val2)| [val1 >= 0f64, val2 >= 0f64].into_iter())
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
            tx_baseband_qpsk_signal(data_bits.clone().into_iter()).collect();

        let qpsk_rx: Vec<Bit> = rx_baseband_qpsk_signal(qpsk_tx.clone().into_iter()).collect();

        assert_eq!(data_bits, qpsk_rx);
    }
}
