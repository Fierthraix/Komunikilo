use crate::{fftshift, hamming_window, hamming_window_complex};
use std::f64::consts::PI;

use ndrustfft::Zero;
use num_complex::Complex;
use numpy::ndarray::{Array1, Array2, ArrayView1};
use rustfft::FftPlanner;

pub fn ssca_base(s: Vec<f64>, n: usize, np: usize) -> Array2<Complex<f64>> {
    assert!(n > np);
    let mut fftp = FftPlanner::new();
    let fft_n = fftp.plan_fft_forward(n);
    let mut scratch_n = vec![Complex::zero(); fft_n.get_inplace_scratch_len()];
    let fft_np = fftp.plan_fft_forward(np);
    let mut scratch_np = vec![Complex::zero(); fft_np.get_inplace_scratch_len()];

    // Hamming Windows.
    let a = Array1::from_vec(hamming_window_complex(np));
    let g_m: Array2<Complex<f64>> = {
        let mut g = Array2::zeros((0, n));
        for _ in 0..np {
            g.push_row(ArrayView1::from(&Vec::from_iter(
                hamming_window(n).into_iter().map(|i| Complex::new(i, 0f64)),
            )))
            .unwrap();
        }
        g.reversed_axes()
    };

    let xg: Array2<Complex<f64>> = {
        let mut xg = Array2::zeros((0, np));
        for i in 0..n {
            // Step 1.
            let mut xa: Array1<Complex<f64>> = a.clone() * ArrayView1::from(&s[i..i + np]);

            // Step 2.
            fft_np.process_with_scratch(xa.as_slice_mut().unwrap(), &mut scratch_np);
            let xat = Array1::from(fftshift(xa.as_slice().unwrap()));

            // Step 3.
            let em = Array1::from_iter((0..np).map(|j| {
                let k: f64 = j as f64 - np as f64 / 2f64;
                (Complex::new(0f64, -2f64 * PI * k * i as f64 / np as f64)).exp()
            }));

            let g = g_m.row(i);

            let xs = Array1::from_iter((0..np).map(|_| Complex::new(s[np / 2 + i], 0f64).conj()));

            xg.push_row(ArrayView1::from(&(xat * em * xs * g))).unwrap();
        }
        xg
    };
    // Step 4.
    let sx: Array2<Complex<f64>> = {
        let mut sx: Array2<Complex<f64>> = Array2::zeros((0, n));
        for xgi in xg.reversed_axes().rows() {
            let mut xgi = Vec::from_iter(xgi.iter().cloned());
            fft_n.process_with_scratch(&mut xgi, &mut scratch_n);
            sx.push_row(ArrayView1::from(&fftshift(&xgi))).unwrap();
        }
        sx.reversed_axes()
    };
    sx
}

pub fn ssca_mapped(s: Vec<f64>, n: usize, np: usize) -> Array2<Complex<f64>> {
    let sx = ssca_base(s, n, np);
    let mut sxf: Array2<Complex<f64>> = Array2::zeros((np + 1, 2 * n + 1));
    // Step 5.
    for q_p in 0..n {
        for k_p in 0..np {
            let f: f64 = k_p as f64 / (2f64 * np as f64) - q_p as f64 / (2f64 * n as f64);
            let a: f64 = k_p as f64 / np as f64 + q_p as f64 / n as f64;
            let k: usize = (np as f64 * (f + 0.5)) as usize;
            let q: usize = (n as f64 * a) as usize;
            sxf[[k, q]] = sx[[q_p, k_p]];
        }
    }
    sxf
}
