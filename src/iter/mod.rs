use std::iter::Sum;
use std::ops::{AddAssign, Mul};

mod chunks;
mod convolution;
mod inflate;
mod integrate;
mod take_every;
use crate::iter::chunks::Chunks;
use crate::iter::convolution::{Convolver, Nonvolver};
use crate::iter::inflate::Inflate;
use crate::iter::integrate::{Integrate, Nintegrate};
use crate::iter::take_every::Take;

pub trait Iter: Iterator {
    ///
    /// ```rust
    /// # use komunikilo::iter::Iter;
    /// ```
    fn chunks<T>(self, num_chunks: usize) -> Chunks<T, Self>
    where
        Self: Iterator<Item = T> + Sized,
        T: Copy,
    {
        Chunks::new(self, num_chunks)
    }

    ///
    /// ```rust
    /// # use komunikilo::iter::Iter;
    /// ```
    fn convolve<T>(self, filter: Vec<T>) -> Convolver<T, Self>
    where
        Self: Iterator<Item = T> + Sized,
        T: Mul<T, Output = T> + Sum<T> + Copy,
    {
        Convolver::new(self, filter)
    }

    ///
    /// ```rust
    /// # use komunikilo::iter::Iter;
    /// ```
    fn nonvolve<T, const N: usize>(self, filter: Vec<T>) -> Nonvolver<T, Self, N>
    where
        Self: Iterator<Item = [T; N]> + Sized,
        T: Mul<T, Output = T> + AddAssign + Copy + Default,
    {
        Nonvolver::new(self, filter)
    }

    /// Makes `copies` copies of an input iterator.
    /// ```rust
    /// # use komunikilo::iter::Iter;
    /// let copies = 5;
    /// let first: Vec<usize> = (0..5).flat_map(|x| std::iter::repeat(x).take(copies)).collect();
    /// let second: Vec<usize> = (0..5).into_iter().inflate(copies).collect();
    /// assert_eq!(first, second);
    /// ```
    fn inflate<T>(self, copies: usize) -> Inflate<T, Self>
    where
        Self: Iterator<Item = T> + Sized,
        T: Copy,
    {
        Inflate::new(self, copies)
    }

    /// Like `.sum()`, except it returns all intermediate values.
    /// ```rust
    /// # use komunikilo::iter::Iter;
    /// let num = 1426;
    /// let first: Vec<usize> = (1..num + 1).into_iter().collect();
    /// let second: Vec<usize> = [1].into_iter().cycle().take(num).integrate().collect();
    /// assert_eq!(first, second);
    /// ```
    fn integrate<T>(self) -> Integrate<T, Self>
    where
        Self: Iterator<Item = T> + Sized,
        T: std::default::Default,
    {
        Integrate::new(self)
    }

    /// Integrate a simultaneous stream of numbers.
    /// ```rust
    /// # use komunikilo::iter::Iter;
    /// const N: usize = 4;
    /// let first: Vec<[usize; N]> = (1..100 + 1).into_iter().map(|x| [x; N]).collect();
    /// let second: Vec<[usize; N]> = [1]
    ///     .into_iter()
    ///     .cycle()
    ///     .take(100)
    ///     .map(|x| [x; N])
    ///     .nintegrate()
    ///     .collect();
    /// assert_eq!(first, second);
    /// ```
    fn nintegrate<T, const N: usize>(self) -> Nintegrate<T, Self, N>
    where
        Self: Iterator<Item = [T; N]> + Sized,
        T: std::default::Default + Copy,
    {
        Nintegrate::new(self)
    }

    /// Equivalent to:
    /// ```rust
    /// # use komunikilo::iter::Iter;
    /// let frequency = 5;
    /// let first: Vec<usize> = (0..100)
    ///     .enumerate()
    ///     .filter_map(|(idx, val)| {
    ///         if idx % frequency == 0 {
    ///             Some(val)
    ///         } else {
    ///             None
    ///         }
    ///     })
    ///     .collect();
    /// let second: Vec<usize> = (0..100)
    ///     .take_every(frequency)
    ///     .collect();
    /// assert_eq!(first, second);
    /// ```
    fn take_every<T>(self, frequency: usize) -> Take<T, Self>
    where
        Self: Iterator<Item = T> + Sized,
    {
        Take::new(self, frequency)
    }
}

impl<I: Iterator> Iter for I {}
