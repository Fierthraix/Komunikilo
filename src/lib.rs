use num::complex::Complex;
use rand::{rngs::ThreadRng, Rng};
use rand_distr::{Distribution, Normal};

pub mod bpsk;
mod bufmap;
mod costas;
mod filters;
pub mod qpsk;

pub type Bit = bool;

pub fn db(x: f64) -> f64 {
    10f64 * x.log10()
}

pub fn undb(x: f64) -> f64 {
    10f64.powf(x / 10f64)
}

pub fn linspace(start: f64, stop: f64, num: usize) -> impl Iterator<Item = f64> {
    let step = (stop - start) / (num as f64);
    (0..num).map(move |i| start + step * (i as f64))
}

pub fn inflate<I, T>(input: I, rate: usize) -> impl Iterator<Item = T>
where
    I: Iterator<Item = T>,
    T: Clone,
{
    input.flat_map(move |item| std::iter::repeat(item).take(rate))
}

pub fn convolve_linear(signal: Vec<f64>, filter: Vec<f64>) -> Vec<f64> {
    let out_len = signal.len() + filter.len() - 1;
    let mut out = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let mut sum = 0f64;
        let j_min = if i < filter.len() {
            0
        } else {
            i - filter.len()
        };
        for j in j_min..i + 1 {
            if j < signal.len() && (i - j) < filter.len() {
                sum += signal[j] * filter[i - j];
            }
        }
        out.push(sum)
    }

    out
}

pub fn bits_to_nrz<T>(message: T) -> impl Iterator<Item = f64>
where
    T: Iterator<Item = Bit>,
{
    message.map(|bit| if bit { 1_f64 } else { -1_f64 }) // Convert bits to NRZ values.
}

/*
 if the "sign"
*/
pub fn foo_sign<I>(signal: I) -> impl Iterator<Item = Bit>
where
    I: Iterator<Item = Complex<f64>>,
{
    signal.map(|sample| {
        if sample.re == 0f64 {
            sample.im >= 0f64
        } else {
            sample.re >= 0f64
        }
    })
}

pub fn erf(x: f64) -> f64 {
    let t: f64 = 1f64 / (1f64 + 0.5 * x.abs());
    let tau = t
        * (-x.powi(2) - 1.26551223
            + 1.00002368 * t
            + 0.37409196 * t.powi(2)
            + 0.09678418 * t.powi(3)
            - 0.18628806 * t.powi(4)
            + 0.27886807 * t.powi(5)
            - 1.13520398 * t.powi(6)
            + 1.48851587 * t.powi(7)
            - 0.82215223 * t.powi(8)
            + 0.17087277 * t.powi(9))
        .exp();
    if x >= 0f64 {
        1f64 - tau
    } else {
        tau - 1f64
    }
}

pub fn erfc(x: f64) -> f64 {
    1f64 - erf(x)
}

/*
pub fn ber<I, T>(tx: I, rx: I) -> f64
where
    I: Iterator<Item = T>,
    T: Eq,
{
    let num_samples = tx.len();
    let num_errors = tx
        .zip(rx)
        .enumerate()
        .map(|(idx, (tx, rx))| if tx == rx { 1 } else { 0 })
        .sum();
}
*/

pub fn awgn<I>(signal: I, sigma: f64) -> impl Iterator<Item = Complex<f64>>
where
    I: Iterator<Item = Complex<f64>>,
{
    // let mut rng = rand::thread_rng();
    // let dist = Normal::new(0f64, sigma).unwrap();
    signal
        .zip(
            Normal::new(0f64, sigma)
                .unwrap()
                .sample_iter(rand::thread_rng()),
        )
        .map(|(sample, noise)| sample + noise)
}

pub struct Awgn {
    rng: ThreadRng,
    dist: Normal<f64>,
}

impl Awgn {
    pub fn new(sigma: f64) -> Self {
        Self {
            rng: rand::thread_rng(),
            dist: Normal::new(0_f64, sigma).unwrap(),
        }
    }
    //fn noise(&mut self) -> f64 {
    //self.dist.sample_iter
    //}
    pub fn awgn<'a, I>(&'a mut self, signal: I) -> impl Iterator<Item = Complex<f64>> + '_
    where
        I: Iterator<Item = Complex<f64>> + 'a,
    {
        signal
            .zip(self.dist.sample_iter(&mut self.rng))
            .map(|(sample, noise)| sample + noise)
    }
}

#[cfg(test)]
mod tests {
    use crate::convolve_linear;

    #[test]
    fn it_works() {
        let fs = 44100;
        let baud = 900;
        let Nbits = 4000;
        let _f0 = 1800;
        let Ns = fs / baud;
        let _N = Nbits * Ns;

        let signal: Vec<f64> = (0..50).map(|x| x.into()).collect();
        // let signal: Vec<f64> = (0..10).map(|x| x.into()).collect();
        // let signal: Vec<f64> = (0..50).map(|x| x as f64).collect();

        let filter = vec![1., 1., 1., 1.];

        let convolution = convolve_linear(signal, filter);

        let expected = vec![
            0., 1., 3., 6., 10., 14., 18., 22., 26., 30., 34., 38., 42., 46., 50., 54., 58., 62.,
            66., 70., 74., 78., 82., 86., 90., 94., 98., 102., 106., 110., 114., 118., 122., 126.,
            130., 134., 138., 142., 146., 150., 154., 158., 162., 166., 170., 174., 178., 182.,
            186., 190., 144., 97., 49.,
        ];

        // let expected = vec![0., 1., 3., 6., 10., 14., 18., 22., 26., 30., 24., 17., 9.];

        assert_eq!(expected.len(), convolution.len());
        assert_eq!(expected, convolution);
    }
}
