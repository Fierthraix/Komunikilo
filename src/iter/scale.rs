pub struct Scale<T: std::ops::Mul<f64, Output = T>, I: Iterator<Item = T>> {
    source: I,
    scalar: f64,
}

impl<T: std::ops::Mul<f64, Output = T>, I: Iterator<Item = T>> Scale<T, I> {
    pub fn new(source: I, scalar: f64) -> Scale<T, I> {
        Self { source, scalar }
    }
}

impl<T: std::ops::Mul<f64, Output = T>, I: Iterator<Item = T>> Iterator for Scale<T, I> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        Some(self.source.next()? * self.scalar)
    }
}
