use komunikilo::qpsk::{rx_qpsk_signal, tx_qpsk_signal};
use komunikilo::{awgn, iter::Iter, Bit};
use plotpy::{Curve, Plot};
use welch_sde::{Build, PowerSpectrum, SpectralDensity};

#[macro_use]
mod util;

#[test]
fn qpsk_graphs() {
    let data: Vec<Bit> = [
        false, false, true, false, false, true, false, true, true, true,
    ]
    .into_iter()
    .inflate(2)
    .collect();

    let samp_rate = 44100;
    let fc = 1800_f64;
    let symb_rate = 900;

    let tx: Vec<f64> = tx_qpsk_signal(data.iter().cloned(), samp_rate, symb_rate, fc).collect();
    let rx: Vec<Bit> = rx_qpsk_signal(tx.iter().cloned(), samp_rate, symb_rate, fc).collect();

    /*
    let xtx: Vec<f64> = linspace(0f64, 1f64, tx.len()).collect();
    let xrx: Vec<f64> = linspace(0f64, 1f64, rx.len()).collect();

    let rx_t: Vec<f64> = rx.iter().cloned().map(bit_to_nrz).collect();
    plot!(xrx, rx_t, "/tmp/rx_t.png");

    plot!(xtx, tx, "/tmp/tx_t.png");
    */

    assert_eq!(rx, data);

    let sigma = 2f64;
    let noisy_signal: Vec<f64> = awgn(tx.iter().cloned(), sigma).collect();

    let psd: SpectralDensity<f64> = SpectralDensity::builder(&tx, fc).build();
    let sd = psd.periodogram();
    plot!(sd.frequency(), (*sd).to_vec(), "/tmp/qpsk_specdens.png");
    let psd: SpectralDensity<f64> = SpectralDensity::builder(&noisy_signal, fc).build();
    let sd = psd.periodogram();
    plot!(
        sd.frequency(),
        (*sd).to_vec(),
        "/tmp/qpsk_specdens_dirty.png"
    );

    let psd: PowerSpectrum<f64> = PowerSpectrum::builder(&tx).build();
    let sd = psd.periodogram();
    plot!(sd.frequency(), (*sd).to_vec(), "/tmp/qpsk_pwrspctrm.png");
    let psd: PowerSpectrum<f64> = PowerSpectrum::builder(&noisy_signal).build();
    let sd = psd.periodogram();
    plot!(
        sd.frequency(),
        (*sd).to_vec(),
        "/tmp/qpsk_pwrspctrm_dirty.png"
    );
}
