use crate::{bit_to_nrz, chaos::LogisticMap, Bit};
use itertools::Itertools;

pub fn tx_baseband_dcsk<I: Iterator<Item = Bit>>(message: I) -> impl Iterator<Item = f64> {
    message
        .zip(LogisticMap::new(3.9, 0.1))
        .flat_map(|(bit, reference)| [reference, reference * bit_to_nrz(bit)].into_iter())
}

pub fn rx_baseband_dcsk<I: Iterator<Item = f64>>(message: I) -> impl Iterator<Item = Bit> {
    message
        .tuples()
        .map(|(reference, information)| reference * information > 0f64)
}

#[cfg(test)]
mod tests {

    use super::*;
    extern crate rand;
    use crate::dcsk::tests::rand::Rng;

    #[test]
    fn baseband_dcsk() {
        let mut rng = rand::thread_rng();
        let num_bits = 9001;
        let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

        let dcsk_tx: Vec<f64> = tx_baseband_dcsk(data_bits.iter().cloned()).collect();
        let dcsk_rx: Vec<Bit> = rx_baseband_dcsk(dcsk_tx.iter().cloned()).collect();
        assert_eq!(data_bits, dcsk_rx);
    }
}
