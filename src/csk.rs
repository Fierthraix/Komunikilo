use crate::{chaos::LogisticMap, Bit};

pub fn tx_baseband_csk<I: Iterator<Item = Bit>>(message: I) -> impl Iterator<Item = f64> {
    let mu = 3.9;
    let x0_1 = 0.1;
    let x0_2 = 0.15;
    message
        .zip(LogisticMap::new(mu, x0_1))
        .zip(LogisticMap::new(mu, x0_2))
        .map(|((bit, chaos_1), chaos_2)| if bit { chaos_1 } else { chaos_2 })
}

fn rx_baseband_csk<I: Iterator<Item = f64>>(message: I) -> impl Iterator<Item = Bit> {
    let mu = 3.9;
    let x0_1 = 0.1;
    let x0_2 = 0.15;
    message
        .zip(LogisticMap::new(mu, x0_1))
        .zip(LogisticMap::new(mu, x0_2))
        .map(|((sample, chaos_1), chaos_2)| (sample - chaos_1).abs() < (sample - chaos_2).abs())
}

#[cfg(test)]
mod tests {

    use super::*;
    extern crate rand;
    use crate::Rng;

    #[test]
    fn baseband_csk() {
        let mut rng = rand::thread_rng();
        let num_bits = 9001;
        let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

        let csk_tx: Vec<f64> = tx_baseband_csk(data_bits.iter().cloned()).collect();
        let csk_rx: Vec<Bit> = rx_baseband_csk(csk_tx.iter().cloned()).collect();
        assert_eq!(data_bits, csk_rx);
    }
}
