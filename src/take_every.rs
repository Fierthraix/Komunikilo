pub trait TakeIt: Iterator {
    /// Equivalent to:
    /// ```rust
    /// use crate::comms::take_every::TakeIt;
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

impl<I: Iterator> TakeIt for I {}

pub struct Take<T, I>
where
    I: Iterator<Item = T>,
{
    source: I,
    frequency: usize,
    curr: usize,
}

impl<T, I> Take<T, I>
where
    I: Iterator<Item = T>,
{
    pub fn new(source: I, frequency: usize) -> Take<T, I> {
        Self {
            source,
            frequency,
            curr: 0,
        }
    }
}

impl<T, I> Iterator for Take<T, I>
where
    I: Iterator<Item = T>,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        loop {
            if self.curr % self.frequency == 0 {
                self.curr += 1;
                return self.source.next();
            } else {
                self.source.next();
                self.curr += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX: usize = 250;

    #[test]
    fn it_works() {
        let expected: Vec<usize> = (0..MAX / 5).into_iter().map(|i| i * 5).collect();

        let result: Vec<usize> = (0..MAX).into_iter().take_every(5).collect();

        assert_eq!(expected, result);
    }
}
