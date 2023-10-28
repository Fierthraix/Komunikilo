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

pub trait NintegrateIt: Iterator {
    fn nintegrate<T, const N: usize>(self) -> Nintegrate<T, Self, N>
    where
        Self: Iterator<Item = [T; N]> + Sized,
        T: std::default::Default + Copy,
    {
        Nintegrate::new(self)
    }
}

impl<I: Iterator> NintegrateIt for I {}

pub struct Nintegrate<T, I, const N: usize>
where
    I: Iterator<Item = [T; N]>,
{
    source: I,
    sum: [T; N],
}

impl<T, I, const N: usize> Nintegrate<T, I, N>
where
    I: Iterator<Item = [T; N]>,
    T: std::default::Default + Copy,
{
    pub fn new(source: I) -> Nintegrate<T, I, N> {
        Self {
            source,
            sum: [T::default(); N],
        }
    }
}

impl<T, I, const N: usize> Iterator for Nintegrate<T, I, N>
where
    I: Iterator<Item = [T; N]>,
    T: std::default::Default + std::ops::AddAssign + Copy,
{
    type Item = [T; N];

    fn next(&mut self) -> Option<[T; N]> {
        self.source
            .next()?
            .into_iter()
            .zip(self.sum.iter_mut())
            .for_each(|(row, sum)| *sum += row);
        Some(self.sum)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn integrate() {
        let num = 1426;

        let expected: Vec<usize> = (1..num + 1).into_iter().collect();
        let result: Vec<usize> = [1].into_iter().cycle().take(num).integrate().collect();
        assert_eq!(expected, result);
    }

    const N: usize = 4;
    #[test]
    fn nintegrate() {
        let num = 1426;

        let expected: Vec<[usize; N]> = (1..num + 1).into_iter().map(|x| [x; N]).collect();
        let result: Vec<[usize; N]> = [1]
            .into_iter()
            .cycle()
            .take(num)
            .map(|x| [x; N])
            .nintegrate()
            .collect();
        assert_eq!(expected, result);
    }
}
