use komunikilo::linspace;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;

#[macro_use]
mod util;

#[test]
fn asdf() {
    let x: Vec<f64> = linspace(-6f64, 6f64, 250).collect();
    let y1: Vec<f64> = x.iter().cloned().map(|x| x.cos()).collect();
    let y2: Vec<f64> = x.iter().cloned().map(|x| x.sin()).collect();

    plot!(x, y1, y2, "/tmp/qwer.png");
    // assert!(false);
}
