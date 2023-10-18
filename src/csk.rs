use crate::{logistic_map, Bit};

fn tx_baseband_csk<I>(message: I) -> impl Iterator<Item = f64>
where
    I: Iterator<Item = Bit>,
{
    // Multiply each branch by either one or the other chaos generator.

    struct Csk<I>
    where
        I: Iterator<Item = Bit>,
    {
        source: I,
        curr_a: f64,
        curr_b: f64,
    }

    impl<I> Csk<I>
    where
        I: Iterator<Item = Bit>,
    {
        fn new(source: I) -> Csk<I> {
            Self {
                source,
                curr_a: 0.1,
                curr_b: 0.15,
            }
        }
    }

    impl<I> Iterator for Csk<I>
    where
        I: Iterator<Item = Bit>,
    {
        type Item = f64;

        fn next(&mut self) -> Option<f64> {
            self.curr_a = logistic_map(3.9, self.curr_a);
            self.curr_b = logistic_map(3.9, self.curr_b);
            match self.source.next() {
                Some(bit) => {
                    if bit {
                        Some(self.curr_a)
                    } else {
                        Some(self.curr_b)
                    }
                }
                None => None,
            }
        }
    }

    trait CskExt: Iterator {
        fn csk_tx(self) -> Csk<Self>
        where
            Self: Iterator<Item = Bit> + Sized,
        {
            Csk::new(self)
        }
    }

    impl<I: Iterator> CskExt for I {}

    message.csk_tx()
}

fn rx_baseband_csk<I>(message: I) -> impl Iterator<Item = Bit>
where
    I: Iterator<Item = f64>,
{
    // Multiply each branch by either one or the other chaos generator.

    struct Csk<I>
    where
        I: Iterator<Item = f64>,
    {
        source: I,
        curr_a: f64,
        curr_b: f64,
    }

    impl<I> Csk<I>
    where
        I: Iterator<Item = f64>,
    {
        fn new(source: I) -> Csk<I> {
            Self {
                source,
                curr_a: 0.1,
                curr_b: 0.15,
            }
        }
    }

    impl<I> Iterator for Csk<I>
    where
        I: Iterator<Item = f64>,
    {
        type Item = Bit;

        fn next(&mut self) -> Option<Bit> {
            self.curr_a = logistic_map(3.9, self.curr_a);
            self.curr_b = logistic_map(3.9, self.curr_b);
            match self.source.next() {
                Some(num) => Some((num - self.curr_a).abs() < (num - self.curr_b).abs()),
                None => None,
            }
        }
    }

    trait CskExt: Iterator {
        fn csk_rx(self) -> Csk<Self>
        where
            Self: Iterator<Item = f64> + Sized,
        {
            Csk::new(self)
        }
    }

    impl<I: Iterator> CskExt for I {}

    message.csk_rx()
}

#[cfg(test)]
mod tests {

    use super::*;
    extern crate rand;
    use crate::csk::tests::rand::Rng;

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
