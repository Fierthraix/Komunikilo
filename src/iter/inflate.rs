#[derive(Clone)]
pub struct Inflate<T: Copy, I: Iterator<Item = T>> {
    source: I,
    copies: usize,
    curr_copy: usize,
    curr: Option<T>,
}

impl<T: Copy, I: Iterator<Item = T>> Inflate<T, I> {
    pub fn new(source: I, copies: usize) -> Inflate<T, I> {
        Self {
            source,
            copies,
            curr_copy: 0,
            curr: None,
        }
    }
}

impl<T: Copy, I: Iterator<Item = T>> Iterator for Inflate<T, I> {
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
