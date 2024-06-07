use assert_approx_eq::assert_approx_eq;
use komunikilo::bpsk::tx_bpsk_signal;
use komunikilo::cdma::{tx_cdma_bpsk_signal, tx_cdma_qpsk_signal};
use komunikilo::fsk::tx_bfsk_signal;
use komunikilo::hadamard::HadamardMatrix;
use komunikilo::iter::Iter;
use komunikilo::qpsk::tx_qpsk_signal;
use komunikilo::{avg_energy, awgn, erfc, linspace, Bit};
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use rand::Rng;

#[macro_use]
mod util;

macro_rules! check_eb {
    ($num_bits:expr, $tx:expr) => {
        // Run the `tx()` function, calculate the power, and divide by `num_bits`.
        $tx.map(|s_i| s_i.powi(2)).sum::<f64>() / $num_bits as f64
    };
}

#[test]
fn check_ebs() {
    let num_bits: usize = 9002;
    let mut rng = rand::thread_rng();
    let data: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect(); // Make new random data.

    const fs: usize = 43200;
    const sr: usize = 90;
    const fc: f64 = 180f64;

    const fl: f64 = 1600f64;
    const fh: f64 = 1800f64;

    // let bpsk_tx =  { tx_bpsk_signal(data.iter().cloned(), fs, sr, fc).collect() };
    let bpsk_tx = tx_bpsk_signal(data.iter().cloned(), fs, sr, fc); //.scale(48f64.sqrt().powi(-1));
    let bpsk_eb = check_eb!(num_bits, bpsk_tx);

    let bpsk_tx_2: Vec<f64> = tx_bpsk_signal(data.iter().cloned(), fs, sr, fc).collect();
    println!("{}", avg_energy(&bpsk_tx_2));
    let bpsk_eb_2 = avg_energy(&bpsk_tx_2) / num_bits as f64;

    let qpsk_tx = tx_qpsk_signal(data.iter().cloned(), fs, sr, fc); //.scale(24f64.sqrt().powi(-1));
    let qpsk_eb = check_eb!(num_bits, qpsk_tx);

    let fsk_tx = tx_bfsk_signal(data.iter().cloned(), fs, sr, fl, fh); //.scale(35.88_f64.sqrt().powi(-1));
    let fsk_eb = check_eb!(num_bits, fsk_tx);

    let h = HadamardMatrix::new(16);
    let key = h.key(2);

    let cdma_bpsk_tx = tx_cdma_bpsk_signal(data.iter().cloned(), fs, sr, fc, key); //.scale(48f64.sqrt().powi(-1));
    let cdma_bpsk_eb = check_eb!(num_bits, cdma_bpsk_tx);

    let cdma_qpsk_tx = tx_cdma_qpsk_signal(data.iter().cloned(), fs, sr, fc, key); //.scale(24f64.sqrt().powi(-1));
    let cdma_qpsk_eb = check_eb!(num_bits, cdma_qpsk_tx);

    println!(" FSK:     {:.2}", fsk_eb);
    println!("BPSK:     {:.2}", bpsk_eb);
    println!("BPSK2:    {:.2}", bpsk_eb_2);
    println!("BPSKCDMA  {:.2}", cdma_bpsk_eb);
    println!("QPSK:     {:.2}", qpsk_eb);
    println!("QPSKCDMA  {:.2}", cdma_qpsk_eb);
    assert!(false);
}
