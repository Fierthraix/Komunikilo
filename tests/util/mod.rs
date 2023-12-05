use pyo3::prelude::*;
use pyo3::types::IntoPyDict;

macro_rules! init_matplotlib {
    ($py: expr) => {{
        let matplotlib = $py.import("matplotlib").unwrap();
        let plt = $py.import("matplotlib.pyplot").unwrap();
        let locals = [("matplotlib", matplotlib), ("plt", plt)].into_py_dict($py);
        $py.eval("matplotlib.use('agg')", None, Some(&locals))
            .unwrap();
        locals
    }};
}

macro_rules! plot {
    ($x:expr, $y:expr, $name:expr) => {
        Python::with_gil(|py| {
            let locals = init_matplotlib!(py);

            locals.set_item("x", &$x).unwrap();
            locals.set_item("y", &$y).unwrap();
            let (fig, axes): (&PyAny, &PyAny) = py
                .eval("plt.subplots(1)", None, Some(&locals))
                .unwrap()
                .extract()
                .unwrap();
            locals.set_item("fig", fig).unwrap();
            locals.set_item("axes", axes).unwrap();
            py.eval("fig.set_size_inches(16, 9)", None, Some(&locals))
                .unwrap();
            py.eval("axes.plot(x, y)", None, Some(&locals)).unwrap();
            py.eval(&format!("fig.savefig('{}')", $name), None, Some(&locals))
                .unwrap();
            py.eval("plt.close('all')", None, Some(&locals)).unwrap();
        })
    };
    ($x:expr, $y1:expr, $y2:expr, $name:expr) => {
        plot!($x, $y1, $y2, false, $name)
    };
    ($x:expr, $y1:expr, $y2:expr, $log:expr, $name:expr) => {
        Python::with_gil(|py| {
            let locals = init_matplotlib!(py);

            locals.set_item("x", &$x).unwrap();
            locals.set_item("y1", &$y1).unwrap();
            locals.set_item("y2", &$y2).unwrap();
            let (fig, axes): (&PyAny, &PyAny) = py
                .eval("plt.subplots(1)", None, Some(&locals))
                .unwrap()
                .extract()
                .unwrap();
            locals.set_item("fig", fig).unwrap();
            locals.set_item("axes", axes).unwrap();
            py.eval("fig.set_size_inches(16, 9)", None, Some(&locals))
                .unwrap();
            if $log {
                py.eval("axes.set_yscale('log')", None, Some(&locals))
                    .unwrap();
            }
            py.eval("axes.plot(x, y1)", None, Some(&locals)).unwrap();
            py.eval("axes.plot(x, y2)", None, Some(&locals)).unwrap();
            py.eval(&format!("fig.savefig('{}')", $name), None, Some(&locals))
                .unwrap();
            py.eval("plt.close('all')", None, Some(&locals)).unwrap();
        })
    };
}

macro_rules! ber_plot {
    ($x:expr, $y1:expr, $y2:expr, $name:expr) => {
        plot!($x, $y1, $y2, true, $name)
    };
    ($x:expr, $y:expr, $name:expr) => {
        Python::with_gil(|py| {
            let locals = init_matplotlib!(py);

            locals.set_item("x", &$x).unwrap();
            locals.set_item("y", &$y).unwrap();
            let (fig, axes): (&PyAny, &PyAny) = py
                .eval("plt.subplots(1)", None, Some(&locals))
                .unwrap()
                .extract()
                .unwrap();
            locals.set_item("fig", fig).unwrap();
            locals.set_item("axes", axes).unwrap();
            py.eval("fig.set_size_inches(16, 9)", None, Some(&locals))
                .unwrap();
            if $log {
                py.eval("axes.set_yscale('log')", None, Some(&locals))
                    .unwrap();
            }
            py.eval("axes.plot(x, y)", None, Some(&locals)).unwrap();
            py.eval(&format!("fig.savefig('{}')", $name), None, Some(&locals))
                .unwrap();
            py.eval("plt.close('all')", None, Some(&locals)).unwrap();
        })
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
