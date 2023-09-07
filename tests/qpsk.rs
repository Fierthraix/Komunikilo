use comms::qpsk::{
    rx_baseband_qpsk_signal, rx_qpsk_signal, tx_baseband_qpsk_signal, tx_qpsk_signal,
};
use comms::{bit_to_nrz, inflate, linspace, Bit};
use num::complex::Complex;
use plotpy::{Curve, Plot};
use std::f64::consts::PI;

#[macro_use]
mod util;

#[test]
fn basic_math() {
    let time_step = 0.01;
    let freq = 1.0;

    let nums = (0..2000).into_iter().map(|idx| {
        let t = idx as f64 * time_step;
        let a = (2_f64 * PI * freq * t).cos();
        let b = (2_f64 * PI * freq * t).sin();

        // (a, b)
        Complex::new(a, b)
    });

    let (a, b): (Vec<f64>, Vec<f64>) = nums.map(|num| (num.re, num.im)).unzip();
    println!("{:?}", a);
    println!("{:?}", b);

    let x: Vec<f64> = (0..a.len()).map(|num| num as f64).collect();

    plot!(x, a, "/tmp/1_t.png");
    plot!(x, b, "/tmp/2_t.png");
    // assert!(false);
}

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

    let mut c = Curve::new();
    c.draw(
        &xd,
        &data.iter().cloned().map(bit_to_nrz).collect::<Vec<_>>(),
    );
    let mut p = Plot::new();
    p.add(&c);
    p.save("/tmp/d_t.png").unwrap();

    let mut c = Curve::new();
    c.draw(
        &xrx,
        &rx.iter().cloned().map(bit_to_nrz).collect::<Vec<_>>(),
    );
    let mut p = Plot::new();
    p.add(&c);
    p.save("/tmp/rx_t.png").unwrap();

    let mut c = Curve::new();
    c.draw(&xtx, &tx);
    let mut p = Plot::new();
    p.add(&c);
    p.save("/tmp/tx_t.png").unwrap();

    assert_eq!(rx, data);
}
