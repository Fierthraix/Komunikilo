use crate::Bit;
use itertools::Itertools;
use num::complex::Complex;

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
    // I guess I have to do the whole constellation.
    message.flat_map(|sample| [sample.re >= 0f64, sample.im >= 0f64].into_iter())
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

        let qpsk_rx: Vec<Bit> =
            rx_baseband_qpsk_signal(qpsk_tx.clone().into_iter()).collect::<Vec<_>>();
        assert_eq!(data_bits, qpsk_rx);
    }
}
