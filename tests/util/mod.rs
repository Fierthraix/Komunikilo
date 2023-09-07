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
}
