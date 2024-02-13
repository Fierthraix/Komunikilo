use std::collections::VecDeque;
use std::iter::{Iterator, Sum};
use std::ops::Mul;

pub trait ConvolveIt: Iterator {}

impl<I: Iterator> ConvolveIt for I {}

pub struct Convolver<T, I: Iterator<Item = T>>
where
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

    fn _convolve(&self) -> T {
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

                Some(self._convolve())
            }
            None => {
                // if self.buffer.is_empty() {
                if self.buffer.len() <= 1 {
                    None
                } else {
                    self.buffer.pop_back();
                    Some(self._convolve())
                }
            }
        }
    }
}

pub struct Nonvolver<T, I, const N: usize>
where
    I: Iterator<Item = [T; N]>,
    T: Mul<T, Output = T> + std::ops::AddAssign + Copy + Default,
{
    source: I,
    filter: Vec<T>,
    buffer: VecDeque<[T; N]>,
}

impl<T, I, const N: usize> Nonvolver<T, I, N>
where
    I: Iterator<Item = [T; N]>,
    T: Mul<T, Output = T> + std::ops::AddAssign + Copy + Default,
{
    pub fn new(source: I, filter: Vec<T>) -> Nonvolver<T, I, N> {
        let filter_len = filter.len();
        Nonvolver {
            source,
            filter,
            buffer: VecDeque::with_capacity(filter_len + 1),
        }
    }

    pub fn _nonvolve(&self) -> [T; N] {
        let mut ret = [T::default(); N];
        self.buffer
            .iter()
            .zip(self.filter.iter().rev())
            .for_each(|(&buf, &filt)| {
                buf.iter()
                    .enumerate()
                    .for_each(|(jdx, &buf)| ret[jdx] += buf * filt)
            });
        ret
    }
}

impl<T, I, const N: usize> Iterator for Nonvolver<T, I, N>
where
    I: Iterator<Item = [T; N]>,
    T: Mul<T, Output = T> + std::ops::AddAssign + Copy + Default,
{
    type Item = [T; N];

    fn next(&mut self) -> Option<[T; N]> {
        match self.source.next() {
            Some(nums) => {
                // Here is where the convolution happens.
                self.buffer.push_front(nums);

                if self.buffer.len() > self.filter.len() {
                    // Deal with front, before buffer is filled to capacity.
                    self.buffer.pop_back();
                }
                Some(self._nonvolve())
            }
            None => {
                if self.buffer.len() <= 1 {
                    None
                } else {
                    self.buffer.pop_back();
                    Some(self._nonvolve())
                }
            }
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::iter::Iter;

    fn convolve_linear(signal: Vec<f64>, filter: Vec<f64>) -> Vec<f64> {
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

        let convolution: Vec<f64> = signal.into_iter().convolve(filter).collect();

        let expected = Vec::from(EXPECTED);
        assert_eq!(expected.len(), convolution.len());
        assert_eq!(expected, convolution);
    }

    #[test]
    fn nonvolver() {
        const N: usize = 4;
        let signal: Vec<[f64; N]> = (0..50)
            .map(|x| {
                let y = x.into();
                [y; N]
            })
            .collect();
        let filter = vec![1., 1., 1., 1.];

        let convolution: Vec<[f64; 4]> = signal.into_iter().nonvolve(filter).collect();

        let expected: Vec<[f64; N]> = EXPECTED.iter().map(|&x| [x; N]).collect();
        assert_eq!(expected, convolution);
    }
}
