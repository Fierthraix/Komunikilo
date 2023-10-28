use crate::chunks::ChunkIt;
use crate::inflate::InflateIt;
use crate::integrate::IntegrateIt;
use crate::{bit_to_nrz, Bit};
use std::f64::consts::PI;

pub fn tx_fsk<I: Iterator<Item = Bit>>(
    message: I,
    sample_rate: usize,
    symbol_rate: usize,
    carrier_freq: f64,
    start_time: f64,
) -> impl Iterator<Item = f64> {
    let samples_per_symbol: usize = sample_rate / symbol_rate;
    let t_step: f64 = 1_f64 / (samples_per_symbol as f64);

    let sep: f64 = 0.05 * carrier_freq;

    message
        .map(bit_to_nrz)
        .inflate(samples_per_symbol)
        .enumerate()
        .map(move |(idx, sig)| {
            let time = start_time + (idx as f64) * t_step;
            (2_f64 * PI * (carrier_freq + sig * (sep / 2f64)) * time).cos()
        })
}

pub fn rx_fsk<I: Iterator<Item = f64>>(
    signal: I,
    sample_rate: usize,
    symbol_rate: usize,
    carrier_freq: f64,
    start_time: f64,
) -> impl Iterator<Item = Bit> {
    let samples_per_symbol: usize = sample_rate / symbol_rate;
    let t_step: f64 = 1_f64 / (samples_per_symbol as f64);

    let sep: f64 = 0.05 * carrier_freq;
    signal
        .chunks(samples_per_symbol)
        .enumerate()
        .map(move |(idx, chunk)| {
            let time = start_time + (idx as f64) * t_step * (samples_per_symbol as f64);

            let rez = chunk
                .iter()
                .enumerate()
                .map(|(jdx, &s_i)| {
                    let t_i = time + (jdx as f64) * t_step;
                    [
                        s_i * (2f64 * PI * (carrier_freq + (sep / 2f64)) * t_i).cos(),
                        s_i * (2f64 * PI * (carrier_freq + (sep / 2f64)) * t_i).sin(),
                        s_i * (2f64 * PI * (carrier_freq - (sep / 2f64)) * t_i).cos(),
                        s_i * (2f64 * PI * (carrier_freq - (sep / 2f64)) * t_i).sin(),
                    ]
                })
                .fold((0f64, 0f64, 0f64, 0f64), |acc, v| {
                    (acc.0 + v[0], acc.1 + v[1], acc.2 + v[2], acc.3 + v[3])
                });

            (rez.0.powi(2) + rez.1.powi(2)) - (rez.2.powi(2) + rez.3.powi(2)) >= 0f64
        })
}

pub fn tx_fm<I: Iterator<Item = f64>>(
    signal: I,
    sample_rate: usize,
    symbol_rate: usize,
    carrier_freq: f64,
    start_time: f64,
) -> impl Iterator<Item = f64> {
    let k = 0.05;
    let samples_per_symbol: usize = sample_rate / symbol_rate;
    let t_step: f64 = 1_f64 / (samples_per_symbol as f64);

    signal
        .inflate(samples_per_symbol)
        .integrate()
        .enumerate()
        .map(move |(idx, msg_cumsum)| {
            let time = start_time + (idx as f64) * t_step;
            (2_f64 * PI * carrier_freq * time + k * msg_cumsum).cos()
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate rand;
    extern crate rand_distr;
    use crate::fm::tests::rand::Rng;

    #[test]
    fn fsk() {
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
        let fsk_tx: Vec<f64> = tx_fsk(
            data_bits.iter().cloned(),
            samp_rate,
            symbol_rate,
            carrier_freq,
            0_f64,
        )
        .collect();

        let fsk_rx: Vec<Bit> = rx_fsk(
            fsk_tx.iter().cloned(),
            samp_rate,
            symbol_rate,
            carrier_freq,
            0_f64,
        )
        .collect();

        assert_eq!(fsk_rx, data_bits);
    }
}
