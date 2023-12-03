use comms::fm::{rx_fsk, tx_fsk};
use comms::{linspace, Bit};
use plotpy::{Curve, Plot};

#[macro_use]
mod util;

#[test]
fn fsk_graphs() {
    let data: Vec<Bit> = [
        false, false, true, false, false, true, false, true, true, true,
    ]
    .into();

    let samp_rate = 44100;
    let fc = 1800_f64;
    let symb_rate = 900;

    let tx: Vec<f64> = tx_fsk(data.iter().cloned(), samp_rate, symb_rate, fc).collect();
    let rx: Vec<Bit> = rx_fsk(tx.iter().cloned(), samp_rate, symb_rate, fc).collect();

    let x: Vec<f64> = linspace(0f64, 10f64, tx.len()).collect();

    plot!(x, tx, "/tmp/fsk_tx.png");

    assert_eq!(rx, data);
}
