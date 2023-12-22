use crate::iter::Iter;
use crate::*;
use assert_approx_eq::assert_approx_eq;
use num::Zero;
use realfft::RealFftPlanner;
use std::f64::consts::PI;

const FFT_LEN: usize = 8192;

pub fn bandpass<I>(low: f64, high: f64, stream: I, sample_rate: usize) -> impl Iterator<Item = f64>
where
    I: Iterator<Item = f64>,
{
    let mut planner: RealFftPlanner<f64> = RealFftPlanner::new();
    let fft = planner.plan_fft_forward(FFT_LEN);
    let ifft = planner.plan_fft_inverse(FFT_LEN);

    let frequencies: Vec<f64> = (0..FFT_LEN)
        .map(|i| (i * sample_rate) as f64 / (FFT_LEN as f64))
        .collect();

    stream.chunks(FFT_LEN).flat_map(move |chunk| {
        assert_eq!(chunk.len(), FFT_LEN);
        let mut spectrum = fft.make_output_vec();
        let mut chunk = chunk.clone();
        // Convert with FFT.
        fft.process(&mut chunk, &mut spectrum).unwrap();
        // Perform bandpass.
        spectrum
            .iter_mut()
            .zip(frequencies.iter())
            .for_each(|(s, &f)| {
                if f < low || f > high {
                    *s = Complex::zero();
                }
            });
        // Convert with IFFT.
        let mut real_sig: Vec<f64> = ifft.make_output_vec();
        ifft.process(&mut spectrum, &mut real_sig).unwrap();

        // Normalize FFT.
        real_sig.into_iter().map(|s_i| s_i / (FFT_LEN as f64))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let sample_rate = 10_000;
        let t_step = 1f64 / sample_rate as f64;
        let num_samples = FFT_LEN * 5;

        let time: Vec<f64> = (0..num_samples).map(|i| i as f64 * t_step).collect();

        let f1 = 100f64;
        let f2 = 50f64;
        let f3 = 150f64;

        let s1: Vec<f64> = time.iter().map(|&t| (2f64 * PI * f1 * t).cos()).collect();

        let signal: Vec<f64> = time
            .iter()
            .map(|&t| {
                [f1, f2, f3]
                    .iter()
                    .map(|f_i| (2f64 * PI * f_i * t).cos())
                    .sum::<f64>()
            })
            .collect();

        let eps = 20f64;
        let filtered_signal: Vec<f64> =
            bandpass(f1 - eps, f1 + eps, signal.iter().cloned(), sample_rate).collect();

        // assert_approx_eq!(filtered_signal, s1);
        filtered_signal
            .iter()
            .zip(s1.iter())
            .for_each(|(&fi, &si)| {
                let diff = (fi - si).abs();
                assert!(diff <= 0.4, "{}", diff) // TODO: This should be wayy smaller.
            });
    }
}
