use std::collections::VecDeque;
use std::iter::{Iterator, Sum};
use std::ops::Mul;

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

pub struct Convolver2<T, I>
where
    I: Iterator<Item = (T, T)>,
    T: Mul<T, Output = T> + Sum<T> + Copy,
{
    source: I,
    filter: Vec<T>,
    buffer1: VecDeque<T>,
    buffer2: VecDeque<T>,
}

impl<T, I> Convolver2<T, I>
where
    I: Iterator<Item = (T, T)>,
    T: Mul<T, Output = T> + Sum<T> + Copy,
{
    pub fn new(source: I, filter: Vec<T>) -> Convolver2<T, I> {
        let filter_len = filter.len();
        Convolver2 {
            source,
            filter,
            buffer1: VecDeque::with_capacity(filter_len + 1),
            buffer2: VecDeque::with_capacity(filter_len + 1),
        }
    }

    // pub fn push(&mut self) {}

    pub fn convolve(&self) -> (T, T) {
        (
            self.buffer1
                .iter()
                .zip(self.filter.iter().rev())
                .map(|(&buf, &filt)| buf * filt)
                .sum(),
            self.buffer2
                .iter()
                .zip(self.filter.iter().rev())
                .map(|(&buf, &filt)| buf * filt)
                .sum(),
        )
    }
}

pub fn convolve2<T, I>(signal: I, filter: Vec<T>) -> impl Iterator<Item = (T, T)>
where
    I: Iterator<Item = (T, T)>,
    T: Mul<T, Output = T> + Sum<T> + Copy,
{
    Convolver2::new(signal, filter)
}

/*
pub fn convolve2<T>(signal: I, signal2: T, filter: Vec<T>) -> impl Iterator<Item = (T, T)>
where
    I: Iterator<Item = (T, T)>,
    T: Mul<T, Output = T> + Sum<T> + Copy,
{
    let c1 = Convolver2::new(filter);
    let c2 = Convolver2::new(filter);
}
*/

impl<T, I> Iterator for Convolver2<T, I>
where
    I: Iterator<Item = (T, T)>,
    T: Mul<T, Output = T> + Sum<T> + Copy,
{
    type Item = (T, T);

    fn next(&mut self) -> Option<(T, T)> {
        match self.source.next() {
            Some(nums) => {
                // Here is where the convolution happens.

                let (num1, num2) = nums;

                self.buffer1.push_front(num1);
                self.buffer2.push_front(num2);

                if self.buffer1.len() > self.filter.len() {
                    // Deal with front, before buffer is filled to capacity.
                    self.buffer1.pop_back();
                }
                if self.buffer2.len() > self.filter.len() {
                    // Deal with front, before buffer is filled to capacity.
                    self.buffer2.pop_back();
                }

                Some(self.convolve())
            }
            None => {
                // if self.buffer.is_empty() {
                if self.buffer1.len() <= 1 || self.buffer2.len() <= 1 {
                    None
                } else {
                    self.buffer1.pop_back();
                    self.buffer2.pop_back();
                    Some(self.convolve())
                }
            }
        }
    }
}

pub fn convolve<T, I>(signal: I, filter: Vec<T>) -> impl Iterator<Item = T>
where
    I: Iterator<Item = T>,
    T: Mul<T, Output = T> + Sum<T> + Copy,
{
    Convolver::new(signal, filter)
}

pub struct Convolver<T, I>
where
    I: Iterator<Item = T>,
    T: Mul<T, Output = T> + Sum<T> + Copy,
{
    source: I,
    filter: Vec<T>,
    buffer: VecDeque<T>,
}

impl<T, I> Convolver<T, I>
where
    I: Iterator<Item = T>,
    T: Mul<T, Output = T> + Sum<T> + Copy,
{
    pub fn new(source: I, filter: Vec<T>) -> Convolver<T, I> {
        let filter_len = filter.len();
        Self {
            source,
            filter,
            buffer: VecDeque::with_capacity(filter_len + 1),
        }
    }

    fn convolve(&self) -> T {
        self.buffer
            .iter()
            .zip(self.filter.iter().rev())
            .map(|(&buf, &filt)| buf * filt)
            .sum()
    }
}

impl<T, I> Iterator for Convolver<T, I>
where
    I: Iterator<Item = T>,
    T: Mul<T, Output = T> + Sum<T> + Copy,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        match self.source.next() {
            Some(num) => {
                // Here is where the convolution happens.

                self.buffer.push_front(num);

                if self.buffer.len() > self.filter.len() {
                    // Deal with front, before buffer is filled to capacity.
                    self.buffer.pop_back();
                }

                Some(self.convolve())
            }
            None => {
                // if self.buffer.is_empty() {
                if self.buffer.len() <= 1 {
                    None
                } else {
                    self.buffer.pop_back();
                    Some(self.convolve())
                }
            }
        }
    }
}

// pub fn convolve_stream<I>(signal: I, filter: Vec<f64>) -> impl Iterator<Item = f64>
// where
//     I: Iterator<Item = f64>,
// {
//     let mut buf = VecDeque::with_capacity(filter.len());
// }

#[cfg(test)]
mod test {

    use super::*;

    const EXPECTED: [f64; 53] = [
        0., 1., 3., 6., 10., 14., 18., 22., 26., 30., 34., 38., 42., 46., 50., 54., 58., 62., 66.,
        70., 74., 78., 82., 86., 90., 94., 98., 102., 106., 110., 114., 118., 122., 126., 130.,
        134., 138., 142., 146., 150., 154., 158., 162., 166., 170., 174., 178., 182., 186., 190.,
        144., 97., 49.,
    ];
    #[test]
    fn linear_convolve() {
        let signal: Vec<f64> = (0..50).map(|x| x.into()).collect();
        let filter = vec![1., 1., 1., 1.];

        let convolution = convolve_linear(signal, filter);

        let expected = Vec::from(EXPECTED);
        assert_eq!(expected.len(), convolution.len());
        assert_eq!(expected, convolution);
    }

    #[test]
    fn stream_convolve() {
        let signal: Vec<f64> = (0..50).map(|x| x.into()).collect();
        let filter = vec![1., 1., 1., 1.];

        let convolution: Vec<f64> = convolve(signal.into_iter(), filter).into_iter().collect();

        let expected = Vec::from(EXPECTED);
        assert_eq!(expected.len(), convolution.len());
        assert_eq!(expected, convolution);
    }
}
