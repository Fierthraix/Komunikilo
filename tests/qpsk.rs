use comms::qpsk::{
    rx_baseband_qpsk_signal, rx_qpsk_signal, tx_baseband_qpsk_signal, tx_qpsk_signal,
};
use comms::{awgn, bit_to_nrz, erfc, inflate, linspace, Bit};
use num::complex::Complex;
use plotpy::{Curve, Plot};
use rand::Rng;
use rayon::prelude::*;
use std::f64::consts::PI;

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

    let mut c1 = Curve::new();
    let mut c2 = Curve::new();
    let mut p1 = Plot::new();
    let mut p2 = Plot::new();

    c1.draw(&x, &a);
    c2.draw(&x, &b);
    p1.add(&c1);
    p2.add(&c2);

    p1.save("/tmp/1_t.png").unwrap();
    p2.save("/tmp/2_t.png").unwrap();

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
    let tx: Vec<Complex<f64>> = tx_qpsk_signal(
        data.clone().into_iter(),
        samp_rate,
        symb_rate,
        fc as f64,
        0f64,
    )
    .collect();

    let rx: Vec<Bit> =
        rx_qpsk_signal(tx.clone().into_iter(), samp_rate, symb_rate, fc, 0f64).collect();

    let (i, q): (Vec<f64>, Vec<f64>) = tx
        .clone()
        .into_iter()
        .map(|cmplx_num| (cmplx_num.re, cmplx_num.im))
        .unzip();

    let x: Vec<f64> = linspace(0f64, 1f64, i.len()).collect();
    let xd: Vec<f64> = linspace(0f64, data.len() as f64, data.len()).collect();

    let mut c = Curve::new();
    c.draw(
        &xd,
        &data.clone().into_iter().map(bit_to_nrz).collect::<Vec<_>>(),
    );
    let mut p = Plot::new();
    p.add(&c);
    p.save("/tmp/d_t.png").unwrap();

    let mut c = Curve::new();
    let xrx: Vec<f64> = linspace(0f64, 1f64, rx.len()).collect();
    c.draw(
        &xrx,
        &rx.clone().into_iter().map(bit_to_nrz).collect::<Vec<_>>(),
    );
    let mut p = Plot::new();
    p.add(&c);
    p.save("/tmp/rx_t.png").unwrap();

    let mut c = Curve::new();
    c.draw(&x, &i);
    let mut p = Plot::new();
    p.add(&c);
    p.save("/tmp/i_t.png").unwrap();

    let mut c = Curve::new();
    c.draw(&x, &q);
    let mut p = Plot::new();
    p.add(&c);
    p.save("/tmp/q_t.png").unwrap();

    // assert!(false);

    assert_eq!(rx, data);
}
