mod chebyshev;
mod logistic_map;
pub use chebyshev::*;
pub use logistic_map::*;

pub fn tent_map(mu: f64, x0: f64) -> f64 {
    2f64 * mu * (1f64 - x0.abs()) - 1f64
}

pub fn bernoulli_map(mu: f64, x0: f64) -> f64 {
    if x0 < 0f64 {
        mu * x0 + 1f64
    } else {
        mu * x0 - 1f64
    }
}
