pub trait ChunkExt: Iterator {
    fn chunks<T>(self, num_chunks: usize) -> Chunks<T, Self>
    where
        Self: Iterator<Item = T> + Sized,
        T: Copy,
    {
        Chunks::new(self, num_chunks)
    }
}

impl<I: Iterator> ChunkExt for I {}

pub struct Chunks<T, I>
where
    I: Iterator<Item = T>,
    T: Copy,
{
    source: I,
    num_chunks: usize,
}

impl<T, I> Chunks<T, I>
where
    I: Iterator<Item = T>,
    T: Copy,
{
    pub fn new(source: I, num_chunks: usize) -> Chunks<T, I> {
        Self { source, num_chunks }
    }
}

impl<T, I> Iterator for Chunks<T, I>
where
    I: Iterator<Item = T>,
    T: Copy,
{
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Vec<T>> {
        let mut buf = Vec::with_capacity(self.num_chunks);
        // Get up to num_chunks items
        while buf.len() < self.num_chunks {
            match self.source.next() {
                Some(t) => {
                    buf.push(t);
                }
                None => {
                    break;
                }
            }
        }
        if buf.is_empty() {
            None
        } else {
            Some(buf)
        }
    }
}
