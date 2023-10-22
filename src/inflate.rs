pub trait InflateIt: Iterator {
    fn inflate<T>(self, copies: usize) -> Inflate<T, Self>
    where
        Self: Iterator<Item = T> + Sized,
        T: Copy,
    {
        Inflate::new(self, copies)
    }
}

impl<I: Iterator> InflateIt for I {}

pub struct Inflate<T, I>
where
    I: Iterator<Item = T>,
    T: Copy,
{
    source: I,
    copies: usize,
    curr_copy: usize,
    curr: Option<T>,
}

impl<T, I> Inflate<T, I>
where
    I: Iterator<Item = T>,
    T: Copy,
{
    fn new(source: I, copies: usize) -> Inflate<T, I> {
        Self {
            source,
            copies,
            curr_copy: 0,
            curr: None,
        }
    }
}

impl<T, I> Iterator for Inflate<T, I>
where
    I: Iterator<Item = T>,
    T: Copy,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if self.curr.is_none() || self.curr_copy == self.copies {
            self.curr = self.source.next();
            self.curr_copy = 1;
        } else {
            self.curr_copy += 1;
        }
        self.curr
    }
}

#[cfg(test)]
mod test {

    use super::*;

    fn inflater<I, T>(input: I, rate: usize) -> impl Iterator<Item = T>
    where
        I: Iterator<Item = T>,
        T: Clone,
    {
        input.flat_map(move |item| std::iter::repeat(item).take(rate))
    }

    #[test]
    fn inflated() {
        let result: Vec<usize> = (0..5).into_iter().inflate(14).collect();
        let expected: Vec<usize> = inflater((0..5).into_iter(), 14).collect();

        assert_eq!(result, expected);
    }
}
