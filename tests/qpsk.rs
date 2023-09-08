use comms::qpsk::{
    rx_baseband_qpsk_signal, rx_qpsk_signal, tx_baseband_qpsk_signal, tx_qpsk_signal,
};
use comms::{bit_to_nrz, inflate, linspace, Bit};
use plotpy::{Curve, Plot};

#[macro_use]
mod util;

#[test]
fn qpsk_graphs() {
    let data: Vec<Bit> = inflate(
        [
            false, false, true, false, false, true, false, true, true, true,
        ]
        .into_iter(),
        2,
    )
    .collect();

    // let (i_b, q_b) =
    let samp_rate = 44100;
    // let fc = 100;
    let fc = 1800_f64;
    let symb_rate = 900;
    let tx: Vec<f64> =
        tx_qpsk_signal(data.iter().cloned(), samp_rate, symb_rate, fc as f64, 0f64).collect();

    let rx: Vec<Bit> = rx_qpsk_signal(tx.iter().cloned(), samp_rate, symb_rate, fc, 0f64).collect();

    let xtx: Vec<f64> = linspace(0f64, 1f64, tx.len()).collect();
    let xrx: Vec<f64> = linspace(0f64, 1f64, rx.len()).collect();
    let xd: Vec<f64> = linspace(0f64, data.len() as f64, data.len()).collect();

    let d_t: Vec<f64> = data.iter().cloned().map(bit_to_nrz).collect();
    plot!(xd, d_t, "/tmp/d_t.png");

    let rx_t: Vec<f64> = rx.iter().cloned().map(bit_to_nrz).collect();
    plot!(xrx, rx_t, "/tmp/rx_t.png");

    plot!(xtx, tx, "/tmp/tx_t.png");

    assert_eq!(rx, data);
}
