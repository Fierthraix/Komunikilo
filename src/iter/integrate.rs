pub struct Integrate<T, I: Iterator<Item = T>> {
    source: I,
    sum: T,
}

impl<T: std::default::Default, I: Iterator<Item = T>> Integrate<T, I> {
    pub fn new(source: I) -> Integrate<T, I> {
        Self {
            source,
            sum: T::default(),
        }
    }
}

impl<T, I: Iterator<Item = T>> Iterator for Integrate<T, I>
where
    T: std::default::Default + std::ops::AddAssign + Copy,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.sum += self.source.next()?;
        Some(self.sum)
    }
}

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
