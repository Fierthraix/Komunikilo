
macro_rules! test_data {
    ($num_bits:expr) => {{
        let mut rng = rand::thread_rng();
        let data_bits: Vec<Bit> = (0..$num_bits).map(|_| rng.gen::<Bit>()).collect();
        let tx_data = tx(data_bits.iter().cloned());
        let rx_data: Vec<Bit> = rx(tx_data).collect();
        assert_eq!(data_bits, rx_data);
    }};
}

/*
// fn test_data<FnTx, FnRx, ItBit, ItT, T>(num_bits: usize, tx: FnTx, rx: FnRx)
fn test_data<FnTx, FnRx, ItBit, ItT, T>(
    num_bits: usize,
    tx: fn(ItBit) -> impl Iterator<Item = T>,
    rx: FnRx,
) where
    FnTx: Fn() -> ItT,
    FnRx: Fn(ItT) -> ItBit,
    ItBit: Iterator<Item = Bit>,
    ItT: Iterator<Item = T>,
    T: Copy,
{
    let mut rng = rand::thread_rng();
    let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

    let tx_data = tx(data_bits.iter().cloned());
    let rx_data: Vec<Bit> = rx(tx_data).collect();
}
*/
