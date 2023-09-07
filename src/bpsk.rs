use crate::convolution::ConvolveExt;
use crate::{bit_to_nrz, inflate, Bit};
use num::complex::Complex;
use std::f64::consts::PI;

pub fn tx_baseband_bpsk_signal<I>(message: I) -> impl Iterator<Item = Complex<f64>>
where
    I: Iterator<Item = Bit>,
{
    message.map(|bit| {
        if bit {
            Complex::<f64>::new(1f64, 0f64)
        } else {
            Complex::<f64>::new(-1f64, 0f64)
        }
    })
}

pub fn rx_baseband_bpsk_signal<I>(message: I) -> impl Iterator<Item = Bit>
where
    I: Iterator<Item = Complex<f64>>,
{
    message.map(|sample| sample.re >= 0f64)
}

pub fn tx_bpsk_signal<I>(
    message: I,
    sample_rate: usize,
    symbol_rate: usize,
    carrier_freq: f64,
    start_time: f64,
) -> impl Iterator<Item = f64>
where
    I: Iterator<Item = Bit>,
{
    let samples_per_symbol: usize = sample_rate / symbol_rate;
    let t_step: f64 = 1_f64 / (samples_per_symbol as f64);

    inflate(message.map(bit_to_nrz), samples_per_symbol)
        .enumerate()
        .map(move |(idx, msg_val)| {
            let time = start_time + (idx as f64) * t_step;
            msg_val * (2_f64 * PI * carrier_freq * time).cos()
        })
}

pub fn rx_bpsk_signal<I>(
    message: I,
    sample_rate: usize,
    symbol_rate: usize,
    carrier_freq: f64,
    start_time: f64,
) -> impl Iterator<Item = Bit>
where
    I: Iterator<Item = f64>,
{
    let samples_per_symbol: usize = sample_rate / symbol_rate;
    let t_step: f64 = 1_f64 / (samples_per_symbol as f64);
    let filter: Vec<f64> = (0..samples_per_symbol).map(|_| 1f64).collect();
    let real_demod = message.enumerate().map(move |(idx, sample)| {
        let time = start_time + (idx as f64) * t_step;
        sample * (2_f64 * PI * carrier_freq * time).cos()
    });
    real_demod
        .convolve(filter)
        .enumerate()
        .filter_map(move |(i, val)| {
            if i % samples_per_symbol == 0 {
                Some(val)
            } else {
                None
            }
        })
        .map(|thresh_val| thresh_val > 0f64)
        .skip(1)
    // Take every `samples_per_symbol`th output from this threshold detector.
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate itertools;
    extern crate plotpy;
    extern crate rand;
    extern crate rand_distr;
    use crate::bpsk::tests::rand::Rng;
    use plotpy::{Curve, Plot};

    #[test]
    fn baseband() {
        let mut rng = rand::thread_rng();
        let num_bits = 9001;
        let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

        let bpsk_tx: Vec<Complex<f64>> =
            tx_baseband_bpsk_signal(data_bits.iter().cloned()).collect();
        let bpsk_rx: Vec<Bit> = rx_baseband_bpsk_signal(bpsk_tx.iter().cloned()).collect();
        assert_eq!(data_bits, bpsk_rx);
    }

    #[test]
    fn full_bpsk() {
        // Simulation parameters.
        let num_bits = 4000; // How many bits to transmit overall.
                             // Input parameters.
        let samp_rate = 44100; // Clock rate for both RX and TX.
        let symbol_rate = 900; // Rate symbols come out the things.
        let carrier_freq = 1800_f64;
        // Derived Parameters.
        let samples_per_symbol = samp_rate / symbol_rate;
        let _num_samples = num_bits * samples_per_symbol;
        //let t_step: f64 = 1_f64 / (samples_per_symbol as f64);

        // Test proper.
        let mut rng = rand::thread_rng();
        let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

        // Tx output.
        let bpsk_tx: Vec<f64> = tx_bpsk_signal(
            data_bits.iter().cloned(),
            samp_rate,
            symbol_rate,
            carrier_freq,
            0_f64,
        )
        .collect();

        let bpsk_rx: Vec<Bit> = rx_bpsk_signal(
            bpsk_tx.iter().cloned(),
            samp_rate,
            symbol_rate,
            carrier_freq,
            0_f64,
        )
        .collect();

        let samples_per_symbol: usize = samp_rate / symbol_rate;
        let t_step: f64 = 1_f64 / (samples_per_symbol as f64);

        let x =
            |y: &Vec<f64>| -> Vec<f64> { (0..y.len()).map(|idx| idx as f64 * t_step).collect() };

        let mut curve_tx = Curve::new();
        curve_tx.draw(&x(&bpsk_tx), &bpsk_tx);

        let mut plot_tx = Plot::new();
        plot_tx.add(&curve_tx);

        // TODO: Add AWGN to graphic.
        plot_tx.save("/tmp/bpsk_tx.png").unwrap();

        assert_eq!(bpsk_rx, data_bits);
    }
}
