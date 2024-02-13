use average::{concatenate, Estimate, Kurtosis, Skewness};
use komunikilo::bpsk::{rx_bpsk_signal, tx_bpsk_signal};
use komunikilo::cdma::{rx_cdma_bpsk_signal, tx_cdma_bpsk_signal};
use komunikilo::hadamard::HadamardMatrix;
use komunikilo::{awgn, linspace, Bit};
use num::Float;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use rand::Rng;

concatenate!(NormTest, [Kurtosis, kurtosis], [Skewness, skewness]);

fn normtest(data: &[f64]) -> (f64, f64) {
    let n = data.len() as f64;

    let kurtskew: NormTest = data.iter().cloned().collect();
    let b2 = kurtskew.kurtosis.kurtosis();

    let E = 3f64 * (n - 1f64) / (n + 1f64);
    let varb2 =
        24f64 * n * (n - 2f64) * (n - 3f64) / ((n + 1f64) * (n + 1f64) * (n + 3f64) * (n + 5f64)); // [1]_ Eq. 1
    let x = (b2 - E) / varb2.sqrt(); // [1]_ Eq. 4
                                     // [1]_ Eq. 2:
    let sqrtbeta1 = 6f64 * (n * n - 5f64 * n + 2f64) / ((n + 7f64) * (n + 9f64))
        * ((6.0 * (n + 3f64) * (n + 5f64)) / (n * (n - 2f64) * (n - 3f64))).sqrt();
    // [1]_ Eq. 3:
    let A =
        6f64 + 8f64 / sqrtbeta1 * (2f64 / sqrtbeta1 + (1f64 + 4f64 / (sqrtbeta1.powi(2))).sqrt());
    let term1 = 1f64 - 2f64 / (9f64 * A);
    let denom = 1f64 + x * (2f64 / (A - 4f64)).sqrt();
    let term2 = denom.signum() * (1f64 - 2f64 / A) / denom.abs().powf(1f64 / 3f64);
    // let term2 = np.sign(denom) * np.where(denom == 0.0, np.nan,np.power((1-2.0/A)/np.abs(denom), 1/3.0));
    /*
    if np.any(denom == 0):
        msg = ("Test statistic not defined in some cases due to division by "
               "zero. Return nan in that case...")
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    */
    let Z = (term1 - term2) / (2f64 / (9f64 * A)).sqrt(); // [1]_ Eq. 5

    (0.0, 0f64)
}

#[test]
#[ignore]
fn asdf() {
    let samp_rate = 80_000;
    let symb_rate = 1000;
    let freq = 2000f64;

    let n0 = 2;

    let num_bits = 10_000;
    let mut rng = rand::thread_rng();
    let data: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

    let matrix_size = 16;
    let walsh_codes = HadamardMatrix::new(matrix_size);
    let key = walsh_codes.key(0);

    let tx_cdma: Vec<f64> =
        tx_cdma_bpsk_signal(data.iter().cloned(), samp_rate, symb_rate, freq, key).collect();

    let tx_bpsk: Vec<f64> =
        tx_bpsk_signal(data.iter().cloned(), samp_rate, symb_rate, freq).collect();

    let tx_empty: NormTest = awgn((0..tx_cdma.len()).map(|_| 0f64), (10 * n0) as f64).collect();

    let kurtskew: NormTest = awgn(tx_cdma.iter().cloned(), n0 as f64).collect();

    let kurtskew_noisy: NormTest = awgn(tx_cdma.iter().cloned(), (5 * n0) as f64).collect();

    println!(
        "Empty Chan: || Kurt: {} || Skew: {}",
        tx_empty.kurtosis.kurtosis(),
        tx_empty.skewness.skewness()
    );
    println!(
        "CDMA  AWGN: || Kurt: {} || Skew: {}",
        kurtskew.kurtosis.kurtosis(),
        kurtskew.skewness.skewness()
    );
    println!(
        "CDMA++AWGN: || Kurt: {} || Skew: {}",
        kurtskew_noisy.kurtosis.kurtosis(),
        kurtskew_noisy.skewness.skewness()
    );

    assert!(false);
}
