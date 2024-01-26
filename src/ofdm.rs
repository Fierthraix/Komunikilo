use crate::iter::Iter;
use crate::qpsk::{rx_baseband_qpsk_signal, tx_baseband_qpsk_signal};
use crate::{is_int, Bit};
use rustfft::num_traits::Zero;
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::PI;

fn fftshift<T: Clone>(x: &[T]) -> Vec<T> {
    let mut v = Vec::with_capacity(x.len());
    let pivot = x.len().div_ceil(2);
    v.extend_from_slice(&x[pivot..]);
    v.extend_from_slice(&x[..pivot]);
    v
}

pub fn tx_baseband_ofdm_signal<I>(symbols: I) -> impl Iterator<Item = Complex<f64>>
where
    I: Iterator<Item = Complex<f64>>,
{
    let n_subcarriers = 64;
    let n_data_subcarriers = 52;
    let cp_len = n_subcarriers / 4;

    let data_subcarriers: Vec<usize> = [(6..32), (33..59)]
        .iter()
        .flat_map(|i| i.clone().into_iter())
        .collect();

    let mut fftp = FftPlanner::new();
    let fft = fftp.plan_fft_inverse(n_subcarriers);
    let mut fft_scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];

    let ofdm_symbols = symbols
        .chunks(n_data_subcarriers) // S/P
        .flat_map(move |data_chunk| {
            // Insert data symbols into data carriers.
            let mut ofdm_symbol_data = vec![Complex::zero(); n_subcarriers];
            data_subcarriers
                .iter()
                .zip(data_chunk.into_iter())
                .for_each(|(&carrier, datum)| ofdm_symbol_data[carrier] = datum);

            let mut ofdm_symbols = fftshift(&ofdm_symbol_data);
            fft.process_with_scratch(&mut ofdm_symbols, &mut fft_scratch); // IFFT

            let cp = ofdm_symbols[n_subcarriers - cp_len..n_subcarriers]
                .iter()
                .cloned();

            let cp_symbol: Vec<Complex<f64>> = cp
                .chain(ofdm_symbols.iter().cloned())
                .map(|s_i| s_i / (n_subcarriers as f64) /*.sqrt()*/)
                .collect();

            cp_symbol
        });

    ofdm_symbols
}

pub fn rx_baseband_ofdm_signal<I>(message: I) -> impl Iterator<Item = Complex<f64>>
where
    I: Iterator<Item = Complex<f64>>,
{
    let n_subcarriers = 64;
    let n_data_subcarriers = 52;
    let cp_len = n_subcarriers / 4;

    let data_subcarriers: Vec<usize> = [(6..32), (33..59)]
        .iter()
        .flat_map(|i| i.clone().into_iter())
        .collect();

    let mut fftp = FftPlanner::new();

    let fft = fftp.plan_fft_forward(n_subcarriers);
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];

    message
        .chunks(n_subcarriers + cp_len) // S/P
        .flat_map(move |ofdm_symbol_data| {
            // let mut buffer: Vec<Complex<f64>> = fftshift(&ofdm_symbol_data[cp_len..]); // CP Removal
            let mut buffer: Vec<Complex<f64>> = Vec::from(&ofdm_symbol_data[cp_len..]); // CP Removal
            fft.process_with_scratch(&mut buffer, &mut scratch); // IFFT
            let demoded = fftshift(&buffer);

            let mut data_symbols = Vec::with_capacity(n_data_subcarriers);
            data_subcarriers
                .iter()
                .for_each(|&carrier| data_symbols.push(demoded[carrier]));

            // P/S
            data_symbols.into_iter()
        })
}

pub fn tx_ofdm_qpsk_signal<I>(
    message: I,
    sample_rate: usize,
    symbol_rate: usize,
    carrier_freq: f64,
) -> impl Iterator<Item = f64>
where
    I: Iterator<Item = Bit>,
{
    assert!(is_int(sample_rate as f64 / symbol_rate as f64));
    let samples_per_symbol = sample_rate / symbol_rate;

    let ofdm_symbols = tx_baseband_ofdm_signal(tx_baseband_qpsk_signal(message));

    ofdm_symbols
        .inflate(samples_per_symbol)
        .enumerate()
        .map(move |(idx, ofdm_symbol)| {
            let time = idx as f64 / sample_rate as f64;
            // ofdm_symbol.re * (w0 * time).cos() - ofdm_symbol.im * (w0 * time).sin()
            (ofdm_symbol * Complex::new(0f64, 2f64 * PI * carrier_freq * time).exp()).re
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate rand;
    extern crate rand_distr;
    use crate::ofdm::tests::rand::Rng;

    #[test]
    fn test_baseband_qpsk_ofdm() {
        let mut rng = rand::thread_rng();
        let num_bits = 2080;
        let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

        let tx_sig: Vec<Complex<f64>> =
            tx_baseband_ofdm_signal(tx_baseband_qpsk_signal(data_bits.iter().cloned())).collect();

        let rx_bits =
            rx_baseband_qpsk_signal(rx_baseband_ofdm_signal(tx_sig.iter().cloned())).collect_vec();

        assert_eq!(data_bits.len(), rx_bits.len());
        assert_eq!(data_bits, rx_bits);
    }
}
