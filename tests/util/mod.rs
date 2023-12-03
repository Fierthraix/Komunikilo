macro_rules! plot {
    ($x:expr, $y:expr, $name:expr) => {
        let mut plot = Plot::new();
        let mut curve = Curve::new();
        curve.draw(&$x, &$y);
        plot.add(&curve);
        plot.save($name).unwrap();
    };
    ($x:expr, $y1:expr, $y2:expr, $name:expr) => {
        plot!($x, $y1, $y2, false, $name)
    };
    ($x:expr, $y1:expr, $y2:expr, $log:expr, $name:expr) => {
        let mut plot = Plot::new();
        let mut curve1 = Curve::new();
        let mut curve2 = Curve::new();
        curve1.draw(&$x, &$y1);
        curve2.draw(&$x, &$y2);
        plot.add(&curve1);
        plot.add(&curve2);
        plot.set_log_y($log);
        plot.save($name).unwrap();
    };
}

macro_rules! ber_plot {
    ($x:expr, $y1:expr, $y2:expr, $name:expr) => {
        plot!($x, $y1, $y2, true, $name)
    };
    ($x:expr, $y:expr, $name:expr) => {
        let mut plot = Plot::new();
        let mut curve = Curve::new();
        curve.draw(&$x, &$y);
        plot.add(&curve);
        plot.set_log_y(true);
        plot.save($name).unwrap();
    };
}

macro_rules! error {
    ($thing1:expr, $thing2:expr) => {
        $thing1
            .iter()
            .cloned()
            .zip($thing2.iter().cloned())
            .map(|(t1, t2)| if t1 == t2 { 0f64 } else { 1f64 })
            .sum::<f64>()
            / $thing1.len() as f64
    };
}

pub fn not_inf(num: f64) -> f64 {
    if num == std::f64::INFINITY {
        std::f64::MAX
    } else if num == std::f64::NEG_INFINITY {
        -std::f64::MAX
    } else {
        num
    }
}

pub fn save_vector(v: &[f64], filename: &str) -> Result<(), std::io::Error> {
    let mut w = csv::Writer::from_writer(std::fs::File::create(filename)?);

    w.write_record(&["v (V)"])?;

    for i in v {
        w.write_record(&[i.to_string()])?;
    }

    Ok(())
}

pub fn save_vector2(v: &[f64], t: &[f64], filename: &str) -> Result<(), std::io::Error> {
    let mut w = csv::Writer::from_writer(std::fs::File::create(filename)?);

    w.write_record(&["t (s)", "v (V)"])?;

    for (v_i, t_i) in v.iter().zip(t.iter()) {
        w.write_record(&[t_i.to_string(), v_i.to_string()])?;
    }

    Ok(())
}
