use num::complex::Complex;
pub struct CostasLoop {
    alpha: f64,
    beta: f64,
    vco_phase: f64,
    vco_freq: f64,
}

impl CostasLoop {
    pub fn new() -> Self {
        Self {
            alpha: 0.132,
            beta: 0.00932,
            vco_phase: 0f64,
            vco_freq: 0f64,
        }
    }

    pub fn update(&mut self, sample: Complex<f64>) -> Complex<f64> {
        let vco_out = self.vco_phase * Complex::<f64>::new(0f64, -1f64);
        let out = sample * vco_out;

        let error = out.re * out.im;

        self.vco_freq += self.beta * error;
        self.vco_phase += self.alpha * error + self.vco_freq;

        out
    }
}
