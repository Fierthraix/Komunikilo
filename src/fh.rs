use crate::bpsk::{rx_bpsk_signal, tx_bpsk_signal};
use crate::iter::Iter;
use crate::{bit_to_nrz, is_int, linspace, Bit};
use std::f64::consts::PI;

use rand::prelude::*;

const SEED: u64 = 64;

/// Hop Table
/// As an iterator, it is constantly shuffling the hop-pattern.
pub(crate) struct HopTable {
    rng: StdRng,
    hopping_table: Vec<f64>,
    // filters: Vec<SosFormatFilter<f64>>,
    idx: usize,
}

impl HopTable {
    pub fn new(
        low: f64,
        high: f64,
        num_freqs: usize,
        // base_freq: f64,
        // sample_rate: usize,
        seed: u64,
    ) -> Self {
        let hopping_table: Vec<f64> = linspace(low, high, num_freqs).collect();

        Self {
            rng: StdRng::seed_from_u64(seed),
            hopping_table,
            // filters,
            idx: 0,
        }
    }
}

impl Iterator for HopTable {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        if self.idx == 0 {
            self.hopping_table.shuffle(&mut self.rng);
            self.idx += 1;
            Some(self.hopping_table[self.idx - 1])
        } else if self.idx == self.hopping_table.len() - 1 {
            self.idx = 0;
            Some(self.hopping_table[self.hopping_table.len() - 1])
        } else {
            self.idx += 1;
            Some(self.hopping_table[self.idx - 1])
        }
    }
}

/// One hop per bit.
pub fn tx_fh_bpsk_signal<I: Iterator<Item = Bit>>(
    message: I,
    low_freq: f64,
    high_freq: f64,
    num_freqs: usize,
    sample_rate: usize,
    symbol_rate: usize,
) -> impl Iterator<Item = f64> {
    assert!(sample_rate / 2 >= high_freq as usize);
    assert!(is_int(sample_rate as f64 / symbol_rate as f64));

    message
        .zip(HopTable::new(low_freq, high_freq, num_freqs, SEED))
        .flat_map(move |(msg_val, hop_freq)| {
            tx_bpsk_signal([msg_val].into_iter(), sample_rate, symbol_rate, hop_freq)
        })
}

pub fn rx_fh_bpsk_signal<I: Iterator<Item = f64>>(
    signal: I,
    low_freq: f64,
    high_freq: f64,
    num_freqs: usize,
    sample_rate: usize,
    symbol_rate: usize,
) -> impl Iterator<Item = Bit> {
    let samples_per_symbol: usize = sample_rate / symbol_rate;
    assert!(sample_rate / 2 >= high_freq as usize);
    assert!(is_int(sample_rate as f64 / symbol_rate as f64));

    signal
        .chunks(samples_per_symbol)
        .zip(HopTable::new(low_freq, high_freq, num_freqs, SEED))
        .flat_map(move |(signal_chunk, hop_freq)| {
            rx_bpsk_signal(signal_chunk.into_iter(), sample_rate, symbol_rate, hop_freq)
        })
}

pub fn tx_fh_bpsk_signal_old<I: Iterator<Item = Bit>>(
    message: I,
    low_freq: f64,
    high_freq: f64,
    num_freqs: usize,
    sample_rate: usize,
    symbol_rate: usize,
    carrier_freq: f64,
) -> impl Iterator<Item = f64> {
    let samples_per_symbol: usize = sample_rate / symbol_rate;
    let t_step: f64 = 1_f64 / (samples_per_symbol as f64);
    assert!(sample_rate / 2 >= carrier_freq as usize);
    assert!(is_int(sample_rate as f64 / symbol_rate as f64));
    // TODO: assert to check hopped freqs are within range.

    message
        .map(bit_to_nrz)
        .zip(HopTable::new(low_freq, high_freq, num_freqs, SEED))
        .enumerate()
        .flat_map(move |(idx, (msg_val, hop_freq))| {
            let time = (idx * samples_per_symbol) as f64 * t_step;

            // Generate symbol's worth of data:
            let symbol/*: Vec<f64>*/ = (0..samples_per_symbol)
                .map(move |idx| {
                    let time: f64 = time + t_step * idx as f64;
                    let bpsk_val = msg_val * (2f64 * PI * carrier_freq * time).cos();
                    let freq_hop = (2f64 * PI * hop_freq * time).cos();

                    bpsk_val * freq_hop
                });

            let bandwidth = (high_freq - low_freq) / (num_freqs as f64);
            let sum_freq = hop_freq + carrier_freq;
            let (_low, _high) = (sum_freq - (bandwidth / 2f64), sum_freq + (bandwidth / 2f64));
            /*
            let filter = {
                let bandpass_filter = butter_dyn(
                    20,
                    [low, high].to_vec(),
                    Some(FilterBandType::Bandpass),
                    Some(false),
                    Some(FilterOutputType::Sos),
                    Some(sample_rate as f64),
                );
                let DigitalFilter::Sos(sos) = bandpass_filter else {
                    panic!("Failed to design filter");
                };
                sos
            };
            sosfiltfilt_dyn(symbol, &filter.sos)
            */
            symbol.into_iter()
            // bandpass(
            //     low,
            //     high,
            //     symbol.into_iter(),
            //     sample_rate,
            //     samples_per_symbol,
            // )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iter::Iter;
    #[test]
    fn hop_table() {
        let low = 10e3;
        let high = 20e3;
        let num_freqs = 8;
        let hops: Vec<Vec<f64>> = HopTable::new(low, high, num_freqs, SEED)
            .take(8 * 5)
            .chunks(8)
            .collect();

        let mut freqs: Vec<f64> = linspace(low, high, num_freqs).collect();
        let mut rng = StdRng::seed_from_u64(SEED);
        let mut expected: Vec<Vec<f64>> = vec![];
        for _ in 0..5 {
            freqs.shuffle(&mut rng);
            expected.push(freqs.clone());
        }

        assert_eq!(hops, expected);
    }

    #[test]
    fn fh_bpsk_works() {
        let mut rng = rand::thread_rng();
        // let num_bits = 9002;
        let num_bits = 9002;
        let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

        let sample_rate = 44100; // Clock rate for both RX and TX.
        let symbol_rate = 900; // Rate symbols come out the things.
        let low_freq = 1000f64;
        let high_freq = 2000f64;
        let num_freqs = 16;

        let fh_bpsk_tx: Vec<f64> = tx_fh_bpsk_signal(
            data_bits.iter().cloned(),
            low_freq,
            high_freq,
            num_freqs,
            sample_rate,
            symbol_rate,
        )
        .collect();
        let fh_bpsk_rx: Vec<Bit> = rx_fh_bpsk_signal(
            fh_bpsk_tx.iter().cloned(),
            low_freq,
            high_freq,
            num_freqs,
            sample_rate,
            symbol_rate,
        )
        .collect();
        assert_eq!(data_bits.len(), fh_bpsk_rx.len());

        let ber: f64 = data_bits
            .iter()
            .zip(fh_bpsk_rx.iter())
            .map(|(a, b)| if a == b { 0 } else { 1 })
            .sum::<usize>() as f64
            / data_bits.len() as f64;
        println!("BER: {}", ber);
        assert_eq!(data_bits, fh_bpsk_rx);
    }
}
