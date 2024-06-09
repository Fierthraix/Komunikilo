#[inline]
pub fn logistic_map(mu: f64, x0: f64) -> f64 {
    mu * x0 * (1f64 - x0)
}

pub struct LogisticMap {
    mu: f64,
    xn: f64,
}

impl LogisticMap {
    pub fn new(mu: f64, x0: f64) -> Self {
        assert!((0f64..=1f64).contains(&x0));
        Self { mu, xn: x0 }
    }
}

impl Iterator for LogisticMap {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        self.xn = logistic_map(self.mu, self.xn);
        Some(self.xn)
    }
}
