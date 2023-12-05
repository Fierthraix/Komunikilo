use komunikilo::bpsk::tx_bpsk_signal;
use komunikilo::cdma::{rx_cdma_bpsk_signal, tx_cdma_bpsk_signal};
use komunikilo::hadamard::HadamardMatrix;
use komunikilo::iter::Iter;
use komunikilo::{awgn, bit_to_nrz, Bit};
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use rand::Rng;
use rayon::prelude::*;
use util::save_vector2;
use welch_sde::{Build, PowerSpectrum, SpectralDensity};

#[macro_use]
mod util;

#[test]
fn cdma_graphs() {
    let data: Vec<Bit> = vec![true, true, false, false, true];

    let h = HadamardMatrix::new(32);
    let key = h.key(2);

    // Simulation parameters.
    let samp_rate = 128_000; // Clock rate for both RX and TX.
    let symbol_rate = 1000; // Rate symbols come out the things.
    let carrier_freq = 2500f64;

    let tx: Vec<f64> = tx_cdma_bpsk_signal(
        data.iter().cloned(),
        samp_rate,
        symbol_rate,
        carrier_freq,
        key,
    )
    .collect();

    let rx_clean: Vec<Bit> = rx_cdma_bpsk_signal(
        tx.iter().cloned(),
        samp_rate,
        symbol_rate,
        carrier_freq,
        key,
    )
    .collect();

    let sigma = 2f64;
    let noisy_signal: Vec<f64> = awgn(tx.iter().cloned(), sigma).collect();

    let rx_dirty: Vec<Bit> = rx_cdma_bpsk_signal(
        noisy_signal.iter().cloned(),
        samp_rate,
        symbol_rate,
        carrier_freq,
        key,
    )
    .collect();

    let t: Vec<f64> = (0..tx.len())
        .into_iter()
        .map(|idx| {
            let time_step = symbol_rate as f64 / samp_rate as f64;
            idx as f64 * time_step
        })
        .collect();

    let bpsk_tx: Vec<f64> =
        tx_bpsk_signal(data.iter().cloned(), samp_rate, symbol_rate, carrier_freq).collect();
    let t2: Vec<f64> = (0..bpsk_tx.len())
        .into_iter()
        .map(|idx| {
            let time_step = symbol_rate as f64 / samp_rate as f64;
            idx as f64 * time_step
        })
        .collect();
    // assert_eq!(bpsk_tx.len(), tx.len());
    plot!(t2, bpsk_tx, "/tmp/cdma_tx_bpsk.png");
    plot!(t, tx, "/tmp/cdma_tx.png");
    plot!(t, tx, noisy_signal, "/tmp/cdma_tx_awgn.png");

    assert!(save_vector2(&tx, &t, "/tmp/cdma_bpsk.csv").is_ok());
    // plot!(t, rx_clean, rx_dirty, "/tmp/cdma_rx_awgn.png");
    println!("ERROR: {}", error!(rx_clean, rx_dirty));
    assert!(error!(rx_clean, rx_dirty) <= 0.2);
    assert_eq!(rx_clean, rx_dirty);

    let psd: SpectralDensity<f64> = SpectralDensity::builder(&noisy_signal, carrier_freq).build();
    let sd = psd.periodogram();
    plot!(sd.frequency(), (*sd).to_vec(), "/tmp/cdma_specdens.png");
    let psd: SpectralDensity<f64> = SpectralDensity::builder(&tx, carrier_freq).build();
    let sd = psd.periodogram();
    plot!(
        sd.frequency(),
        (*sd).to_vec(),
        "/tmp/cdma_specdens_clean.png"
    );

    let psd: PowerSpectrum<f64> = PowerSpectrum::builder(&noisy_signal).build();
    let sd = psd.periodogram();
    plot!(sd.frequency(), (*sd).to_vec(), "/tmp/cdma_pwrspctrm.png");
    let psd: PowerSpectrum<f64> = PowerSpectrum::builder(&tx).build();
    let sd = psd.periodogram();
    plot!(
        sd.frequency(),
        (*sd).to_vec(),
        "/tmp/cdma_pwrspctrm_clean.png"
    );
}

#[test]
fn python_plotz() -> PyResult<()> {
    let data: Vec<Bit> = vec![true, true, false, false, true];
    let samp_rate = 80_000; // Clock rate for both RX and TX.
    let symbol_rate = 1000; // Rate symbols come out the things.
    let carrier_freq = 2500_f64;

    let h = HadamardMatrix::new(8);
    let key = h.key(2);

    let data_tx: Vec<f64> = data
        .iter()
        .cloned()
        .map(bit_to_nrz)
        .inflate(samp_rate / symbol_rate)
        .collect();

    let bpsk_tx: Vec<f64> =
        tx_bpsk_signal(data.iter().cloned(), samp_rate, symbol_rate, carrier_freq).collect();

    let cdma_tx: Vec<f64> = tx_cdma_bpsk_signal(
        data.iter().cloned(),
        samp_rate,
        symbol_rate,
        carrier_freq,
        &key,
    )
    .collect();

    let t_step: f64 = 1f64 / (samp_rate as f64);

    Python::with_gil(|py| {
        let plt = py.import("matplotlib.pyplot")?;
        let np = py.import("numpy")?;
        let locals = [("np", np), ("plt", plt)].into_py_dict(py);

        locals.set_item("data_tx", data_tx)?;
        locals.set_item("bpsk_tx", bpsk_tx)?;
        locals.set_item("cdma_tx", cdma_tx)?;
        locals.set_item("dt", t_step)?;

        let x = py.eval("lambda s, dt: [dt * i for i in range(len(s))]", None, None)?;
        locals.set_item("x", x)?;

        let (fig, axes): (&PyAny, &PyAny) =
            py.eval("plt.subplots(4)", None, Some(&locals))?.extract()?;
        locals.set_item("fig", fig)?;
        locals.set_item("axes", axes)?;

        py.eval(
            "axes[0].plot(x(data_tx, dt), data_tx, label='DATA: [1 1 0 0 1]')",
            None,
            Some(&locals),
        )?;
        py.eval(
            "axes[1].plot(x(bpsk_tx, dt), bpsk_tx, label='BPSK: (2.5kHz, 1kHz Data Rate)')",
            None,
            Some(&locals),
        )?;
        py.eval(
            "axes[2].plot(x(cdma_tx, dt), cdma_tx, label='CDMA: 8kHz Chip Rate')",
            None,
            Some(&locals),
        )?;

        locals.set_item("samp_rate", samp_rate)?;
        locals.set_item("freq", carrier_freq)?;

        py.eval(
            "plt.psd(bpsk_tx, Fs=samp_rate, Fc=freq)",
            None,
            Some(&locals),
        )?;
        py.eval(
            "plt.psd(cdma_tx, Fs=samp_rate, Fc=freq)",
            None,
            Some(&locals),
        )?;
        py.eval("[x.legend() for x in axes[:-1]]", None, Some(&locals))?;
        // py.eval("plt.legend()", None, Some(&locals))?;
        // py.eval("plt.show()", None, Some(&locals))?;
        py.eval("fig.set_size_inches(16, 9)", None, Some(&locals))?;
        py.eval(
            "fig.savefig('/tmp/cdma_works.png', dpi=300)",
            None,
            Some(&locals),
        )?;

        Ok(())
    })
}

#[test]
#[ignore]
fn mai_plot() {
    let num_users = 63;

    // Simulation parameters.
    let num_bits = 1000; // How many bits to transmit overall.
    let samp_rate = 128_000; // Clock rate for both RX and TX.
    let symbol_rate = 1000; // Rate symbols come out the things.
    let carrier_freq = 2500_f64;
    let num_samples = num_bits * samp_rate / symbol_rate;

    let mut rng = rand::thread_rng();
    // The data each user will transmit.
    let datas: Vec<Vec<Bit>> = (0..num_users)
        .map(|_| (0..num_bits).map(|_| rng.gen::<Bit>()).collect())
        .collect();

    // The keys each user will use.
    let walsh_codes = HadamardMatrix::new(num_users + 1);
    let keys: Vec<Vec<Bit>> = (0..num_users)
        .map(|idx| walsh_codes.key(idx).clone())
        .collect();

    // Calculate BER as a function of users.
    let bers: Vec<f64> = (0..num_users)
        .into_par_iter()
        .map(|user_count| {
            // The channel comprises of `user_count` users' CDMA-BPSK signals added.
            let channel: Vec<f64> = datas
                .iter()
                .take(user_count + 1)
                .zip(keys.iter())
                .map(|(&ref data, &ref key)| {
                    tx_cdma_bpsk_signal(
                        data.iter().cloned(),
                        samp_rate,
                        symbol_rate,
                        carrier_freq,
                        &key,
                    )
                })
                .fold(vec![0f64; num_samples], |mut acc, tx| {
                    acc.iter_mut().zip(tx).for_each(|(s_i, tx_i)| *s_i += tx_i);
                    acc
                });

            // Find the BER for each user, then average.
            let bers: Vec<f64> = (0..user_count)
                .map(|idx| {
                    let rx: Vec<Bit> = rx_cdma_bpsk_signal(
                        channel.iter().cloned(),
                        samp_rate,
                        symbol_rate,
                        carrier_freq,
                        &keys[idx],
                    )
                    .collect();

                    let errors: usize = rx
                        .iter()
                        .zip(&datas[idx])
                        .map(|(rxi, txi)| if rxi == txi { 0 } else { 1 })
                        .sum();
                    errors as f64 / rx.len() as f64
                })
                .collect();
            bers.iter().sum::<f64>() / bers.len() as f64
        })
        .collect();

    let x: Vec<f64> = (0..bers.len()).map(|i| (i + 1) as f64).collect();
    plot!(x, bers, "/tmp/cdma_mai.png");
}
