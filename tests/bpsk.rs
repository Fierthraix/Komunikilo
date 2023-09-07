use comms::bpsk::{
    rx_baseband_bpsk_signal, rx_bpsk_signal, tx_baseband_bpsk_signal, tx_bpsk_signal,
};
use comms::Bit;
use plotpy::{Curve, Plot};

#[test]
fn bpsk_graphs() {
    let data: Vec<Bit> = vec![
        false, false, true, false, false, true, false, true, true, true,
    ];

    // Simulation parameters.
    let samp_rate = 44100; // Clock rate for both RX and TX.
    let symbol_rate = 900; // Rate symbols come out the things.
    let carrier_freq = 1800_f64;

    let tx: Vec<f64> = tx_bpsk_signal(
        data.clone().into_iter(),
        samp_rate,
        symbol_rate,
        carrier_freq,
        0f64,
    )
    .map(|num| num.re)
    .collect();

    let t: Vec<f64> = (0..tx.len())
        .into_iter()
        .map(|idx| {
            let time_step = symbol_rate as f64 / samp_rate as f64;
            idx as f64 * time_step
        })
        .collect();

    let mut c1 = Curve::new();
    let mut p1 = Plot::new();

    c1.draw(&t, &tx);
    p1.add(&c1);

    p1.save("/tmp/bpsk_tx.png").unwrap();
}
