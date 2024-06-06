use crate::iter::Iter;
use crate::*;
use num::Zero;
use realfft::RealFftPlanner;
use sci_rs::signal::filter::{design::*, sosfiltfilt_dyn};

const FFT_LEN: usize = 8192;

pub fn bandpass<I: Iterator<Item = f64>>(
    low: f64,
    high: f64,
    stream: I,
    sample_rate: usize,
    fft_len: usize,
) -> impl Iterator<Item = f64> {
    let mut planner: RealFftPlanner<f64> = RealFftPlanner::new();
    let fft = planner.plan_fft_forward(fft_len);
    let ifft = planner.plan_fft_inverse(fft_len);

    let frequencies: Vec<f64> = (0..fft_len)
        .map(|i| (i * sample_rate) as f64 / (fft_len as f64))
        .collect();

    stream.chunks(fft_len).flat_map(move |chunk| {
        assert_eq!(chunk.len(), fft_len);
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
        real_sig.into_iter().map(move |s_i| s_i / (fft_len as f64))
    })
}

pub fn butterpass<I: Iterator<Item = f64>>(
    low: f64,
    high: f64,
    stream: I,
    sample_rate: usize,
) -> impl Iterator<Item = f64> {
    let DigitalFilter::Sos(bandpass_filter) = butter_dyn(
        64,
        [low, high].to_vec(),
        Some(FilterBandType::Bandpass),
        Some(false),
        Some(FilterOutputType::Sos),
        Some(sample_rate as f64),
    ) else {
        panic!("Failed to design filter");
    };
    stream
        .chunks(FFT_LEN * 10)
        .flat_map(move |chunk| sosfiltfilt_dyn(chunk.into_iter(), &bandpass_filter.sos))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

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
        let filtered_signal: Vec<f64> = bandpass(
            f1 - eps,
            f1 + eps,
            signal.iter().cloned(),
            sample_rate,
            FFT_LEN,
        )
        .collect();

        // assert_approx_eq!(filtered_signal, s1);
        assert_eq!(signal.len(), filtered_signal.len());
        filtered_signal
            .iter()
            .zip(s1.iter())
            .for_each(|(&fi, &si)| {
                let diff = (fi - si).abs();
                assert!(diff <= 0.4, "{}", diff) // TODO: This should be wayy smaller.
            });
    }
}
