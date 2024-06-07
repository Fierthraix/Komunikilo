use crate::{fftshift, hamming_window};
use std::f64::consts::PI;

use ndrustfft::Zero;
use num_complex::Complex;
use numpy::ndarray::{Array, Array1, Array2, ArrayView};
use rustfft::FftPlanner;

pub fn ssca_base(s: Vec<f64>, n: usize, np: usize) -> Array2<Complex<f64>> {
    // Limit to one window
    // let s = s.as_array().slice(s![..n + np]);
    let s = Vec::from(&s[..n + np]);
    let x: Array2<f64> = {
        let mut x = Array2::zeros((0, np));
        for i in 0..n {
            x.push_row(ArrayView::from(&s[i..i + np])).unwrap();
            // println!("{:?}", &s[i..i + np]);
        }
        x
    };

    let a = Array1::from_vec(hamming_window(np));

    // let mut fftp = RealFftPlanner::<f64>::new();
    let mut fftp = FftPlanner::new();
    let fft = fftp.plan_fft_forward(np);
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];

    // xat = np.array([np.fft.fftshift(np.fft.fft(xi * a, Np)) for xi in x])
    let xat: Array2<Complex<f64>> = {
        let mut xat = Array2::zeros((0, np));
        for x_row in x.rows() {
            let row: Vec<Complex<f64>> = {
                // let mut buffer = fftshift((a.clone() * x_row).as_slice().unwrap());
                let mut buffer: Vec<Complex<f64>> = Vec::from_iter(
                    a.iter()
                        .zip(x_row.iter())
                        .map(|(&a_i, &x_i)| Complex::new(a_i * x_i, 0f64)),
                );

                // let mut fft_output: Vec<Complex<f64>> = fft.make_output_vec();
                fft.process_with_scratch(&mut buffer, &mut scratch);
                fftshift(&buffer)
            };
            xat.push_row(ArrayView::from(&row)).unwrap();
        }
        xat
    };

    let em: Array2<Complex<f64>> = {
        let mut em = Array2::zeros((0, np));
        for m in 0..n {
            em.push_row(ArrayView::from(&Array::from_iter((0..np).map(|i| {
                let k: f64 = i as f64 - np as f64 / 2f64;
                (Complex::new(0f64, -2f64 * PI * k * m as f64 / np as f64)).exp()
            }))))
            .unwrap();
        }
        em
    };

    let g: Array2<Complex<f64>> = {
        let mut g = Array2::zeros((0, n));
        for _ in 0..np {
            g.push_row(ArrayView::from(&Vec::from_iter(
                hamming_window(n).into_iter().map(|i| Complex::new(i, 0f64)),
            )))
            .unwrap();
        }
        g.reversed_axes()
    };

    let xs: Array2<Complex<f64>> = {
        let mut xs = Array2::zeros((0, np));
        for i in 0..n {
            xs.push_row(ArrayView::from(&Vec::from_iter(
                (0..np).map(|_| Complex::new(s[np / 2 + i], 0f64).conj()),
            )))
            .unwrap();
        }
        xs
    };

    let xg = xat * em * xs * g;

    let mut fftp = FftPlanner::new();
    let fft = fftp.plan_fft_forward(n);
    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];

    let sx: Array2<Complex<f64>> = {
        let mut sx: Array2<Complex<f64>> = Array2::zeros((0, n));
        for xgi in xg.t().rows() {
            let mut row = Vec::from_iter(xgi.iter().cloned());
            fft.process_with_scratch(&mut row, &mut scratch);
            sx.push_row(ArrayView::from(&fftshift(&row))).unwrap();
        }
        sx.reversed_axes()
    };
    assert_eq!(sx.shape(), [n, np]);
    sx
}
