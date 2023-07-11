use comms::bpsk::{rx_baseband_bpsk_signal, tx_baseband_bpsk_signal};
use comms::{awgn, erfc, linspace, Bit};
use num::complex::Complex;
use plotpy::{Curve, Plot};
use rand::Rng;

fn ber_bpsk(eb_no: f64) -> f64 {
    0.5 * erfc(eb_no.sqrt())
}

#[test]
fn a_graph() {
    let xmin = 0f64;
    let xmax = 1f64;
    let x: Vec<f64> = linspace(xmin, xmax, 100).collect();
    let y1: Vec<f64> = x.iter().map(|&x| ber_bpsk(x)).collect();
    let y2: Vec<f64> = x.iter().map(|&x| erfc(x)).collect();

    let mut curve1 = Curve::new();
    let mut curve2 = Curve::new();
    curve1.draw(&x, &y1);
    curve2.draw(&x, &y2);
    let mut plot = Plot::new();
    plot.add(&curve1);
    plot.add(&curve2);
    plot.save("/tmp/asdf.svg").unwrap();
}

#[test]
fn bpsk_works() {
    // Make some data.
    let mut rng = rand::thread_rng();
    let num_bits = 9001;
    let data_bits: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

    // Transmit the signal.
    let bpsk_tx: Vec<Complex<f64>> =
        tx_baseband_bpsk_signal(data_bits.clone().into_iter()).collect();

    // An x-axis for plotting Eb/N0.
    let xmin = f64::MIN_POSITIVE;
    let xmax = 25f64;
    let x: Vec<f64> = linspace(xmin, xmax, 100).collect();

    // Container for the Eb/N0.
    let mut y: Vec<f64> = vec![];

    for &i in &x {
        // Run for a given Eb/N0:
        let ber: f64 = {
            let sigma = (1f64 / i as f64).sqrt();
            let noisy_signal = awgn(bpsk_tx.clone().into_iter(), sigma);
            let rx = rx_baseband_bpsk_signal(noisy_signal);

            rx.zip(data_bits.iter())
                .map(|(rx, &tx)| if rx == tx { 0f64 } else { 1f64 })
                .sum::<f64>()
                / num_bits as f64
        };
        y.push(ber);
    }

    let y_theory: Vec<f64> = x.iter().map(|&x| ber_bpsk(x)).collect();

    let mut curve_practice = Curve::new();
    curve_practice.draw(&x, &y);
    let mut curve_theory = Curve::new();
    curve_theory.draw(&x, &y_theory);

    let mut plot = Plot::new();
    plot.add(&curve_practice);
    plot.add(&curve_theory);

    plot.save("/tmp/my_thing.png").unwrap();

    // Add AWGN:
    // rx_signal: Vec<Complex<f64>> = awgn(, )

    let bpsk_rx: Vec<Bit> =
        rx_baseband_bpsk_signal(bpsk_tx.clone().into_iter()).collect::<Vec<_>>();
    assert_eq!(data_bits, bpsk_rx);
}
