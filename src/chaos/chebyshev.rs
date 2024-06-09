#[inline]
/// Can be either 1-2(x_n^2) or 2(x_n^2)-1.
pub fn chebyshev(xn: f64) -> f64 {
    let mu = 3f64;
    // 1f64 - 2f64 * xn.powi(2)
    // 2f64 * xn.powi(2) - 1f64
    mu * xn.powi(3) - (1f64 - mu) * xn
}

pub struct Chebyshev {
    xn: f64,
}

impl Chebyshev {
    pub fn new(x0: f64) -> Self {
        assert!((-1f64..=1f64).contains(&x0));
        Self { xn: x0 }
    }
}

impl Iterator for Chebyshev {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        self.xn = chebyshev(self.xn);
        Some(self.xn)
    }
}
