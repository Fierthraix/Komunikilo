use std::f64::consts::PI;

use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use realfft::RealFftPlanner;
use rustfft::num_traits::Zero;
use rustfft::{num_complex::Complex};
use sci_rs::signal::filter::{design::*, sosfiltfilt_dyn};

// use rustfft::

#[macro_use]
mod util;

const FFT_LEN: usize = 8192;

#[test]
// #[ignore]
fn fft_plot() {
    let sample_rate = 1000;
    let t_step = 1f64 / sample_rate as f64;
    let seconds = 10;

    let time: Vec<f64> = (0..sample_rate * seconds)
        .map(|i| i as f64 * t_step)
        .collect();

    assert_eq!(time.len(), sample_rate * seconds);

    let mut signal: Vec<f64> = time
        .iter()
        .map(|&t| {
            let mut s_i = 0f64;

            for (f, a) in [(1f64, 3f64), (4f64, 1f64), (7f64, 0.5)] {
                s_i += a * (2f64 * PI * f * t).sin();
            }
            s_i
        })
        .collect();

    plot!(time[..FFT_LEN], signal[..FFT_LEN], "/tmp/fft_sig.png");

    // let mut fftp: FftPlanner<f64> = FftPlanner::new();
    let mut fftp: RealFftPlanner<f64> = RealFftPlanner::new();
    let fftr = fftp.plan_fft_forward(FFT_LEN);

    let mut spectrum: Vec<Complex<f64>> = fftr.make_output_vec();
    fftr.process(&mut signal[..FFT_LEN], &mut spectrum).unwrap();

    let frequencies: Vec<f64> = (0..FFT_LEN)
        .map(|i| (i * sample_rate) as f64 / (FFT_LEN as f64))
        .collect();

    let spectrum_re: Vec<f64> = spectrum
        .iter()
        .map(|&s_i| (s_i.re.powi(2) + s_i.im.powi(2)).sqrt() / (FFT_LEN as f64).sqrt())
        .collect();
    plot!(frequencies[..81], spectrum_re[..81], "/tmp/fft.png");

    let ifftr = fftp.plan_fft_inverse(FFT_LEN);
    let mut ift_res: Vec<f64> = ifftr.make_output_vec();
    ifftr.process(&mut spectrum, &mut ift_res).unwrap();
    ift_res.iter_mut().for_each(|s_i| *s_i /= FFT_LEN as f64);
    plot!(time[..8192], ift_res[..8192], "/tmp/ifft.png");
}

#[test]
fn frequency_hopping_bandpass_filter() {
    let sample_rate = 10_000;
    let t_step = 1f64 / sample_rate as f64;
    let seconds = 1;

    let time: Vec<f64> = (0..sample_rate * seconds)
        .map(|i| i as f64 * t_step)
        .collect();
    assert_eq!(time.len(), sample_rate * seconds);

    // Show that multiplying two sine-waves makes two more @ f1+f2 and f1-f2.
    // Show that using a bandpass filter can select for one of these.

    let f_1: f64 = 50f64;
    let f_2: f64 = 40f64;

    let s1: Vec<f64> = time.iter().map(|&t| (2f64 * PI * f_1 * t).cos()).collect();
    let s2: Vec<f64> = time.iter().map(|&t| (2f64 * PI * f_2 * t).cos()).collect();

    let mut channel: Vec<f64> = s1
        .iter()
        .zip(s2.iter())
        .map(|(&s_1, &s_2)| s_1 * s_2)
        .collect();

    let mut fftp: RealFftPlanner<f64> = RealFftPlanner::new();
    let fftr = fftp.plan_fft_forward(FFT_LEN);

    let mut spectrum: Vec<Complex<f64>> = fftr.make_output_vec();
    fftr.process(&mut channel[..FFT_LEN], &mut spectrum)
        .unwrap();

    let frequencies: Vec<f64> = (0..FFT_LEN)
        .map(|i| (i * sample_rate) as f64 / (FFT_LEN as f64))
        .collect();

    let spectrum_re: Vec<f64> = spectrum
        .iter()
        .map(|&s_i| (s_i.re.powi(2) + s_i.im.powi(2)).sqrt() / (FFT_LEN as f64).sqrt())
        .collect();

    plot!(frequencies[..100], spectrum_re[..100], "/tmp/freq_mult.png");

    let low_pass: f64 = (f_1 + f_2) - 10f64;
    let high_pass: f64 = (f_1 + f_2) + 10f64;
    let order = 80;
    let filter = butter_dyn(
        order,
        [low_pass, high_pass].to_vec(),
        Some(FilterBandType::Bandpass),
        Some(false),
        Some(FilterOutputType::Sos),
        Some(sample_rate as f64),
    );

    let DigitalFilter::Sos(sos) = filter else {
        panic!("Not SOS filter")
    };
    let mut filtered: Vec<f64> = sosfiltfilt_dyn(channel.iter().cloned(), &sos.sos);

    let mut spectrum_2: Vec<Complex<f64>> = fftr.make_output_vec();
    fftr.process(&mut filtered[..FFT_LEN], &mut spectrum_2)
        .unwrap();

    let spectrum_re_2: Vec<f64> = spectrum_2
        .iter()
        .map(|&s_i| (s_i.re.powi(2) + s_i.im.powi(2)).sqrt() / (FFT_LEN as f64).sqrt())
        .collect();

    plot!(
        frequencies[..100],
        spectrum_re_2[..100],
        "/tmp/freq_mult_bpf.png"
    );

    let manual_bpf = |low_cut: f64, high_cut: f64| {
        frequencies
            .iter()
            .zip(spectrum.iter())
            .map(move |(&f, &s)| {
                if low_cut <= f && f <= high_cut {
                    s
                } else {
                    Complex::zero()
                }
            })
    };

    let mut manually_filtered: Vec<Complex<f64>> =
        manual_bpf(f_1 + f_2 - 20f64, f_1 + f_2 + 20f64).collect();

    let manual_re: Vec<f64> = manually_filtered
        .iter()
        .cloned()
        .map(|s_i| (s_i.re.powi(2) + s_i.im.powi(2)).sqrt() / (FFT_LEN as f64).sqrt())
        // .zip(frequencies.iter())
        // .map(|(s, &f)| if 80f64 <= f && f <= 100f64 { s } else { 0f64 })
        .collect();

    // let mut spectrum_3: Vec<Complex<f64>> = fftr.make_output_vec();
    // fftr.process(&mut manually_filtered[..FFT_LEN], &mut spectrum_3)
    //     .unwrap();

    // let spectrum_re_3: Vec<f64> = spectrum_3
    //     .iter()
    //     .map(|&s_i| (s_i.re.powi(2) + s_i.im.powi(2)).sqrt() / (FFT_LEN as f64).sqrt())
    //     .collect();

    plot!(
        frequencies[..100],
        manual_re[..100],
        "/tmp/freq_mult_bpf_manual.png"
    );

    let ifftr = fftp.plan_fft_inverse(FFT_LEN);
    let mut ift_res: Vec<f64> = ifftr.make_output_vec();
    ifftr.process(&mut manually_filtered, &mut ift_res).unwrap();
    ift_res.iter_mut().for_each(|s_i| *s_i /= FFT_LEN as f64);
    plot!(time[..8192], ift_res[..8192], "/tmp/freq_mult_bpf_ifft.png");
}
