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
