use comms::bpsk::{rx_bpsk_signal, tx_bpsk_signal};
use comms::{awgn, Bit};
use plotpy::{Curve, Plot};
use welch_sde::{Build, PowerSpectrum, SpectralDensity};

#[macro_use]
mod util;

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
        data.iter().cloned(),
        samp_rate,
        symbol_rate,
        carrier_freq,
        0f64,
    )
    .collect();

    let rx_clean: Vec<Bit> = rx_bpsk_signal(
        tx.iter().cloned(),
        samp_rate,
        symbol_rate,
        carrier_freq,
        0f64,
    )
    .collect();

    let sigma = 2f64;
    let noisy_signal: Vec<f64> = awgn(tx.iter().cloned(), sigma).collect();

    let rx_dirty: Vec<Bit> = rx_bpsk_signal(
        noisy_signal.iter().cloned(),
        samp_rate,
        symbol_rate,
        carrier_freq,
        0f64,
    )
    .collect();

    let t: Vec<f64> = (0..tx.len())
        .into_iter()
        .map(|idx| {
            let time_step = symbol_rate as f64 / samp_rate as f64;
            idx as f64 * time_step
        })
        .collect();

    plot!(t, tx, "/tmp/bpsk_tx.png");
    plot!(t, tx, noisy_signal, "/tmp/bpsk_tx_awgn.png");

    // plot!(t, rx_clean, rx_dirty, "/tmp/bpsk_rx_awgn.png");
    println!("ERROR: {}", error!(rx_clean, rx_dirty));
    assert!(error!(rx_clean, rx_dirty) <= 0.2);
    assert_eq!(rx_clean, rx_dirty);

    let psd: SpectralDensity<f64> = SpectralDensity::builder(&noisy_signal, carrier_freq).build();
    let sd = psd.periodogram();
    plot!(sd.frequency(), (*sd).to_vec(), "/tmp/bpsk_specdens.png");
    let psd: SpectralDensity<f64> = SpectralDensity::builder(&tx, carrier_freq).build();
    let sd = psd.periodogram();
    plot!(
        sd.frequency(),
        (*sd).to_vec(),
        "/tmp/bpsk_specdens_clean.png"
    );

    let psd: PowerSpectrum<f64> = PowerSpectrum::builder(&noisy_signal).build();
    let sd = psd.periodogram();
    plot!(sd.frequency(), (*sd).to_vec(), "/tmp/bpsk_pwrspctrm.png");
    let psd: PowerSpectrum<f64> = PowerSpectrum::builder(&tx).build();
    let sd = psd.periodogram();
    plot!(
        sd.frequency(),
        (*sd).to_vec(),
        "/tmp/bpsk_pwrspctrm_clean.png"
    );
}
