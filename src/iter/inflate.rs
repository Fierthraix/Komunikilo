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
    pub fn new(source: I, copies: usize) -> Inflate<T, I> {
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
