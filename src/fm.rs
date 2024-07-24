use crate::{bit_to_nrz, iter::Iter, Bit};
use std::f64::consts::PI;

pub fn tx_fsk<I: Iterator<Item = Bit>>(
    message: I,
    sample_rate: usize,
    symbol_rate: usize,
    carrier_freq: f64,
) -> impl Iterator<Item = f64> {
    let samples_per_symbol: usize = sample_rate / symbol_rate;

    let sep: f64 = 0.05 * carrier_freq;

    message
        .map(bit_to_nrz)
        .inflate(samples_per_symbol)
        .enumerate()
        .map(move |(idx, sig)| {
            let time = idx as f64 / sample_rate as f64;
            (2_f64 * PI * (carrier_freq + sig * (sep / 2f64)) * time).cos()
        })
}

pub fn rx_fsk<I: Iterator<Item = f64>>(
    signal: I,
    sample_rate: usize,
    symbol_rate: usize,
    carrier_freq: f64,
) -> impl Iterator<Item = Bit> {
    let samples_per_symbol: usize = sample_rate / symbol_rate;

    let sep: f64 = 0.05 * carrier_freq;
    signal
        .chunks(samples_per_symbol)
        .enumerate()
        .map(move |(idx, chunk)| {
            let time = idx as f64 / sample_rate as f64;

            let (s0, s1, s2, s3) = chunk
                .iter()
                .enumerate()
                .map(|(jdx, &s_i)| {
                    let t_i = time + jdx as f64 / sample_rate as f64;
                    [
                        s_i * (2f64 * PI * (carrier_freq + (sep / 2f64)) * t_i).cos(),
                        s_i * (2f64 * PI * (carrier_freq + (sep / 2f64)) * t_i).sin(),
                        s_i * (2f64 * PI * (carrier_freq - (sep / 2f64)) * t_i).cos(),
                        s_i * (2f64 * PI * (carrier_freq - (sep / 2f64)) * t_i).sin(),
                    ]
                })
                .fold(
                    (0f64, 0f64, 0f64, 0f64),
                    |(a0, a1, a2, a3), [v0, v1, v2, v3]| (a0 + v0, a1 + v1, a2 + v2, a3 + v3),
                );

            (s0.powi(2) + s1.powi(2)) - (s2.powi(2) + s3.powi(2)) >= 0f64
        })
}

pub fn tx_fm<I: Iterator<Item = f64>>(
    signal: I,
    sample_rate: usize,
    symbol_rate: usize,
    carrier_freq: f64,
) -> impl Iterator<Item = f64> {
    let k = 0.05;
    let samples_per_symbol: usize = sample_rate / symbol_rate;

    signal
        .inflate(samples_per_symbol)
        .integrate()
        .enumerate()
        .map(move |(idx, msg_cumsum)| {
            let time = idx as f64 / sample_rate as f64;
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

        // Test proper.
        let mut rng = rand::thread_rng();
        let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

        // Tx output.
        let fsk_tx: Vec<f64> = tx_fsk(
            data_bits.iter().cloned(),
            samp_rate,
            symbol_rate,
            carrier_freq,
        )
        .collect();

        let fsk_rx: Vec<Bit> =
            rx_fsk(fsk_tx.iter().cloned(), samp_rate, symbol_rate, carrier_freq).collect();

        assert_eq!(fsk_rx, data_bits);
    }
}
