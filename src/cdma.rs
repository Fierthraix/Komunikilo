use crate::bpsk::{rx_bpsk_signal, tx_bpsk_signal};
use crate::{iter::Iter, Bit};

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
    tx_bpsk_signal(
        tx_baseband_cdma(message, key),
        sample_rate,
        symbol_rate * key.len(),
        carrier_freq,
        start_time,
    )
}

pub fn rx_cdma_bpsk_signal<'a, I>(
    message: I,
    sample_rate: usize,
    symbol_rate: usize,
    carrier_freq: f64,
    start_time: f64,
    key: &'a [Bit],
) -> impl Iterator<Item = Bit> + '_
where
    I: Iterator<Item = f64> + 'a,
{
    rx_baseband_cdma(
        rx_bpsk_signal(
            message,
            sample_rate,
            symbol_rate * key.len(),
            carrier_freq,
            start_time,
        ),
        key,
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
    //#[case(16)]
    //#[case(256)]
    fn bpsk_cdma(#[case] matrix_size: usize) {
        // Simulation parameters.
        let num_bits = 4000; //16; //4000; // How many bits to transmit overall.
                             // Input parameters.
        let samp_rate = 44100; // Clock rate for both RX and TX.
        let symbol_rate = 900; // Rate symbols come out the things.
        let carrier_freq = 1800_f64;
        // Derived Parameters.
        let samples_per_symbol = samp_rate / symbol_rate;
        let _num_samples = num_bits * samples_per_symbol;
        //let t_step: f64 = 1_f64 / (samples_per_symbol as f64);

        let mut rng = rand::thread_rng();
        let data_bits_1: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();
        let data_bits_2: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

        let walsh_codes = HadamardMatrix::new(matrix_size);
        let key_1: Vec<Bit> = walsh_codes.key(0).clone();
        let key_2: Vec<Bit> = walsh_codes.key(1).clone();

        // Tx output.
        let cdma_tx_1/*: Vec<f64>*/ = tx_cdma_bpsk_signal(
            data_bits_1.iter().cloned(),
            samp_rate,
            symbol_rate,
            carrier_freq,
            0_f64,
            &key_1,
        );
        // .collect();

        let cdma_tx_2/*: Vec<f64>*/ = tx_cdma_bpsk_signal(
            data_bits_2.iter().cloned(),
            samp_rate,
            symbol_rate,
            carrier_freq,
            0_f64,
            &key_2,
        );
        // .collect();

        let channel: Vec<f64> = cdma_tx_1.zip(cdma_tx_2).map(|(s1, s2)| s1 + s2).collect();

        let cdma_rx_1: Vec<Bit> = rx_cdma_bpsk_signal(
            channel.iter().cloned(),
            samp_rate,
            symbol_rate,
            carrier_freq,
            0_f64,
            &key_1,
        )
        .collect();

        let cdma_rx_2: Vec<Bit> = rx_cdma_bpsk_signal(
            channel.iter().cloned(),
            samp_rate,
            symbol_rate,
            carrier_freq,
            0_f64,
            &key_2,
        )
        .collect();

        let ber = |tx: &[Bit], rx: &[Bit]| -> f64 {
            let len: usize = std::cmp::min(tx.len(), rx.len());
            let errors: usize = tx
                .iter()
                .zip(rx.iter())
                .map(|(&ti, &ri)| if ti == ri { 0 } else { 1 })
                .sum();
            (errors as f64) / (len as f64)
        };

        // assert_eq!(data_bits_1, cdma_rx_1,);
        // assert_eq!(data_bits_2, cdma_rx_2,);
        assert!(ber(&data_bits_1, &cdma_rx_1) < 0.5);
        assert!(ber(&data_bits_2, &cdma_rx_2) < 0.5);
    }
}
