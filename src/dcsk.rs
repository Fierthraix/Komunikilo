use crate::{bit_to_nrz, logistic_map::LogisticMap, Bit};

pub fn tx_baseband_dcsk<I>(message: I) -> impl Iterator<Item = f64>
where
    I: Iterator<Item = Bit>,
{
    message
        .zip(LogisticMap::new(3.9, 0.1))
        .flat_map(|(bit, reference)| [reference, reference * -bit_to_nrz(bit)].into_iter())
}

pub fn rx_baseband_dcsk<I>(message: I) -> impl Iterator<Item = Bit>
where
    I: Iterator<Item = f64>,
{
    struct RxDcsk<I>
    where
        I: Iterator<Item = f64>,
    {
        source: I,
    }

    impl<I> RxDcsk<I>
    where
        I: Iterator<Item = f64>,
    {
        fn new(source: I) -> RxDcsk<I> {
            Self { source }
        }
    }

    impl<I> Iterator for RxDcsk<I>
    where
        I: Iterator<Item = f64>,
    {
        type Item = Bit;

        fn next(&mut self) -> Option<Bit> {
            let reference: f64 = self.source.next()?;
            let information: f64 = self.source.next()?;
            Some(reference * information < 0f64)
        }
    }

    trait RxDcskExt: Iterator {
        fn dcsk_rx(self) -> RxDcsk<Self>
        where
            Self: Iterator<Item = f64> + Sized,
        {
            RxDcsk::new(self)
        }
    }

    impl<I: Iterator> RxDcskExt for I {}

    message.dcsk_rx()
}

#[cfg(test)]
mod tests {

    use super::*;
    extern crate rand;
    use crate::dcsk::tests::rand::Rng;

    #[test]
    fn baseband_dcsk() {
        let mut rng = rand::thread_rng();
        let num_bits = 10;
        let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

        let dcsk_tx: Vec<f64> = tx_baseband_dcsk(data_bits.iter().cloned()).collect();
        let dcsk_rx: Vec<Bit> = rx_baseband_dcsk(dcsk_tx.iter().cloned()).collect();
        assert_eq!(data_bits, dcsk_rx);
    }
}
