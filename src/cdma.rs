use crate::bpsk::{rx_bpsk_signal, tx_bpsk_signal};
use crate::{bit_to_nrz, is_int, iter::Iter, Bit};

pub fn tx_baseband_cdma<'a, I>(message: I, key: &'a [Bit]) -> impl Iterator<Item = Bit> + 'a
where
    I: Iterator<Item = Bit> + 'a,
{
    message
        .inflate(key.len()) // Make each bit as long as the key.
        .zip(key.iter().cycle())
        .map(|(bit, key)| bit ^ key) // XOR the bit with the entire key.
}

pub fn rx_baseband_cdma<'a, I>(bitstream: I, key: &'a [Bit]) -> impl Iterator<Item = Bit> + '_
where
    I: Iterator<Item = Bit> + 'a,
{
    // Multiply by key, and take the average.
    bitstream
        .zip(key.iter().cycle())
        .map(|(bit, key)| bit ^ key) // XOR the bit with the entire key.
        .chunks(key.len())
        .map(move |data_bit| {
            // Now take the average of the XOR'd part.
            let trueness: usize = data_bit
                .into_iter()
                .map(|bit| if bit { 1 } else { 0 })
                .sum();

            trueness * 2 > key.len()
        })
}

/// A function where:
/// $ s_k(t) = d_k(t) c_k(t) cos(\omega_c t) $
pub fn tx_cdma_bpsk_signal<'a, I>(
    message: I,
    sample_rate: usize,
    symbol_rate: usize,
    carrier_freq: f64,
    start_time: f64,
    key: &'a [Bit],
) -> impl Iterator<Item = f64> + '_
where
    I: Iterator<Item = Bit> + 'a,
{
    assert!(sample_rate / 2 >= key.len() * symbol_rate);
    assert!(sample_rate / 2 >= carrier_freq as usize);
    assert!(is_int(sample_rate as f64 / symbol_rate as f64));
    assert!(is_int(
        sample_rate as f64 / (symbol_rate as f64 * key.len() as f64)
    ));
    let samples_per_symbol: usize = sample_rate / symbol_rate;
    let samples_per_chip: usize = samples_per_symbol / key.len();

    tx_bpsk_signal(message, sample_rate, symbol_rate, carrier_freq, start_time)
        .zip(key.iter().inflate(samples_per_chip).cycle())
        .map(|(s_i, &keybit)| s_i * bit_to_nrz(keybit))
}

pub fn rx_cdma_bpsk_signal<'a, I>(
    signal: I,
    sample_rate: usize,
    symbol_rate: usize,
    carrier_freq: f64,
    start_time: f64,
    key: &'a [Bit],
) -> impl Iterator<Item = Bit> + '_
where
    I: Iterator<Item = f64> + 'a,
{
    assert!(sample_rate / 2 >= key.len() * symbol_rate);
    assert!(sample_rate / 2 >= carrier_freq as usize);
    assert!(is_int(sample_rate as f64 / symbol_rate as f64));
    assert!(is_int(
        sample_rate as f64 / (symbol_rate as f64 * key.len() as f64)
    ));
    let samples_per_symbol: usize = sample_rate / symbol_rate;
    let samples_per_chip: usize = samples_per_symbol / key.len();
    let chip_xored_signal = signal
        .zip(key.iter().inflate(samples_per_chip).cycle())
        .map(|(s_i, &keybit)| s_i * bit_to_nrz(keybit));

    rx_bpsk_signal(
        chip_xored_signal,
        sample_rate,
        symbol_rate,
        carrier_freq,
        start_time,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate rand;
    extern crate rand_distr;
    use crate::cdma::tests::rand::Rng;
    use crate::hadamard::HadamardMatrix;
    use rstest::rstest;

    #[rstest]
    #[case(2)]
    #[case(4)]
    #[case(8)]
    #[case(16)]
    #[case(256)]
    fn baseband_cdma(#[case] matrix_size: usize) {
        let num_bits = 16960;

        let walsh_codes = HadamardMatrix::new(matrix_size);
        let key: Vec<Bit> = walsh_codes.key(0).clone();

        // Data bits.
        let mut rng = rand::thread_rng();
        let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

        // TX CDMA.
        let cdma_tx: Vec<Bit> = tx_baseband_cdma(data_bits.clone().into_iter(), &key).collect();

        let cdma_rx: Vec<Bit> =
            rx_baseband_cdma(cdma_tx.clone().into_iter(), &key.clone()).collect();

        assert_eq!(data_bits, cdma_rx);
    }

    #[rstest]
    #[case(2)]
    #[case(4)]
    #[case(8)]
    #[case(16)]
    #[case(32)]
    #[case(64)]
    fn bpsk_cdma_single_user(#[case] matrix_size: usize) {
        // Simulation parameters.
        let num_bits = 4000; // How many bits to transmit overall.
                             // Input parameters.
        let samp_rate = 128_000; // Clock rate for both RX and TX.
        let symbol_rate = 1000; // Rate symbols come out the things.
        let carrier_freq = 2500_f64;

        let mut rng = rand::thread_rng();
        let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

        let walsh_codes = HadamardMatrix::new(matrix_size);
        let key: Vec<Bit> = walsh_codes.key(0).clone();

        // Tx output.
        let cdma_tx: Vec<f64> = tx_cdma_bpsk_signal(
            data_bits.iter().cloned(),
            samp_rate,
            symbol_rate,
            carrier_freq,
            0_f64,
            &key,
        )
        .collect();

        let cdma_rx: Vec<Bit> = rx_cdma_bpsk_signal(
            cdma_tx.iter().cloned(),
            samp_rate,
            symbol_rate,
            carrier_freq,
            0_f64,
            &key,
        )
        .collect();

        assert_eq!(data_bits, cdma_rx);
    }

    #[rstest]
    #[case(4, 3)]
    #[case(8, 7)]
    #[case(16, 15)]
    #[case(32, 31)]
    #[case(64, 63)]
    fn bpsk_cdma_multi_user(#[case] matrix_size: usize, #[case] num_users: usize) {
        assert!(matrix_size >= num_users);

        // Simulation parameters.
        let num_bits = 1_000; // How many bits to transmit overall.
                              // Input parameters.
        let samp_rate = match matrix_size {
            32 | 64 => 256_000,
            _ => 80_000,
        };
        let symbol_rate = 1000; // Rate symbols come out the things.
        let carrier_freq = 2500_f64;
        let num_samples = num_bits * samp_rate / symbol_rate;

        let mut rng = rand::thread_rng();
        // The data each user will transmit.
        let datas: Vec<Vec<Bit>> = (0..num_users)
            .map(|_| (0..num_bits).map(|_| rng.gen::<Bit>()).collect())
            .collect();

        let walsh_codes = HadamardMatrix::new(matrix_size);
        let keys: Vec<Vec<Bit>> = (0..num_users)
            .map(|idx| walsh_codes.key(idx).clone())
            .collect();

        // Tx output.
        let channel: Vec<f64> = datas
            .iter()
            .zip(keys.iter())
            .map(|(&ref data, &ref key)| {
                tx_cdma_bpsk_signal(
                    data.iter().cloned(),
                    samp_rate,
                    symbol_rate,
                    carrier_freq,
                    0_f64,
                    &key,
                )
            })
            .fold(vec![0f64; num_samples], |mut acc, tx| {
                acc.iter_mut().zip(tx).for_each(|(s_i, tx_i)| *s_i += tx_i);
                acc
            });

        let rxs: Vec<Vec<Bit>> = keys
            .iter()
            .map(|key| {
                rx_cdma_bpsk_signal(
                    channel.iter().cloned(),
                    samp_rate,
                    symbol_rate,
                    carrier_freq,
                    0_f64,
                    key,
                )
                .collect()
            })
            .collect();

        assert_eq!(rxs, datas);
    }
}
