pub trait IntegrateIt: Iterator {
    fn integrate<T>(self) -> Integrate<T, Self>
    where
        Self: Iterator<Item = T> + Sized,
        T: std::default::Default,
    {
        Integrate::new(self)
    }
}

impl<I: Iterator> IntegrateIt for I {}

pub struct Integrate<T, I>
where
    I: Iterator<Item = T>,
{
    source: I,
    sum: T,
}

impl<T, I> Integrate<T, I>
where
    I: Iterator<Item = T>,
    T: std::default::Default,
{
    pub fn new(source: I) -> Integrate<T, I> {
        Self {
            source,
            sum: T::default(),
        }
    }
}

impl<T, I> Iterator for Integrate<T, I>
where
    I: Iterator<Item = T>,
    T: std::default::Default + std::ops::AddAssign + Copy,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.sum += self.source.next()?;
        Some(self.sum)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn it_works() {
        let num = 1426;

        let expected: Vec<usize> = (1..num + 1).into_iter().collect();
        let result: Vec<usize> = [1].into_iter().cycle().take(num).integrate().collect();
        assert_eq!(expected, result);
    }
}
