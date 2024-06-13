#![allow(dead_code, unused_macros)]
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;

macro_rules! init_matplotlib {
    ($py: expr) => {{
        let matplotlib = $py.import_bound("matplotlib").unwrap();
        let plt = $py.import_bound("matplotlib.pyplot").unwrap();
        let locals = [("matplotlib", matplotlib), ("plt", plt)].into_py_dict_bound($py);
        $py.eval_bound("matplotlib.use('agg')", None, Some(&locals))
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
                .eval_bound("plt.subplots(1)", None, Some(&locals))
                .unwrap()
                .extract()
                .unwrap();
            locals.set_item("fig", fig).unwrap();
            locals.set_item("axes", axes).unwrap();
            for line in [
                "fig.set_size_inches(16, 9)",
                "axes.plot(x, y)",
                &format!("fig.savefig('{}')", $name),
                "plt.close('all')",
            ] {
                py.eval_bound(line, None, Some(&locals)).unwrap();
            }
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
                .eval_bound("plt.subplots(1)", None, Some(&locals))
                .unwrap()
                .extract()
                .unwrap();
            locals.set_item("fig", fig).unwrap();
            locals.set_item("axes", axes).unwrap();
            py.eval_bound("fig.set_size_inches(16, 9)", None, Some(&locals))
                .unwrap();
            if $log {
                py.eval_bound("axes.set_yscale('log')", None, Some(&locals))
                    .unwrap();
            }
            for line in [
                "axes.plot(x, y1)",
                "axes.plot(x, y2)",
                &format!("fig.savefig('{}')", $name),
                "plt.close('all')",
            ] {
                py.eval_bound(line, None, Some(&locals)).unwrap();
            }
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
                .eval_bound("plt.subplots(1)", None, Some(&locals))
                .unwrap()
                .extract()
                .unwrap();
            locals.set_item("fig", fig).unwrap();
            locals.set_item("axes", axes).unwrap();
            py.eval_bound("fig.set_size_inches(16, 9)", None, Some(&locals))
                .unwrap();
            if $log {
                py.eval_bound("axes.set_yscale('log')", None, Some(&locals))
                    .unwrap();
            }
            for line in [
                "axes.plot(x, y)",
                &format!("fig.savefig('{}')", $name),
                "plt.close('all')",
            ] {
                py.eval_bound(line, None, Some(&locals)).unwrap();
            }
        })
    };
}

macro_rules! dot_plot {
    ($i:expr, $q:expr, $name:expr) => {
        Python::with_gil(|py| {
            let locals = init_matplotlib!(py);

            locals.set_item("i", &$i).unwrap();
            locals.set_item("q", &$q).unwrap();
            let (fig, axes): (&PyAny, &PyAny) = py
                .eval_bound("plt.subplots(1)", None, Some(&locals))
                .unwrap()
                .extract()
                .unwrap();
            locals.set_item("fig", fig).unwrap();
            locals.set_item("axes", axes).unwrap();
            py.eval_bound("fig.set_size_inches(16, 9)", None, Some(&locals))
                .unwrap();
            for line in [
                "axes.plot(i, q, marker='.', linestyle='None')",
                &format!("fig.savefig('{}')", $name),
                "plt.close('all')",
            ] {
                py.eval_bound(line, None, Some(&locals)).unwrap();
            }
        })
    };
}

macro_rules! iq_plot {
    ($iq_data:expr, $name:expr) => {
        let i: Vec<_> = $iq_data.iter().cloned().map(|s_i| s_i.re).collect();
        let q: Vec<_> = $iq_data.iter().cloned().map(|s_i| s_i.im).collect();
        dot_plot!(i, q, $name)
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

pub fn fit_erfc(x: &[f64], y: &[f64]) -> (f64, f64, f64, f64) {
    Python::with_gil(|py| {
        let scipy = PyModule::import_bound(py, "scipy")?;
        let locals = [("scipy", scipy)].into_py_dict_bound(py);

        locals.set_item("x", x)?;
        locals.set_item("y", y)?;

        let erfc: Py<PyAny> = PyModule::from_code_bound(
            py,
            "def erfc(x, a, b, z, f):
                from scipy.special import erfc
                return a * erfc((x - z) * f) + b",
            "",
            "",
        )?
        .getattr("erfc")?
        .into();
        locals.set_item("erfc", erfc)?;

        let popt: Vec<f64> = py
            .eval_bound(
                "scipy.optimize.curve_fit(erfc, x, y, maxfev=8_000_000)[0]",
                None,
                Some(&locals),
            )?
            .extract()?;

        let mut x = popt.into_iter();
        let a = (
            x.next().unwrap(),
            x.next().unwrap(),
            x.next().unwrap(),
            x.next().unwrap(),
        );

        Ok::<(f64, f64, f64, f64), PyErr>(a)
    })
    .unwrap()
}

pub fn save_vector(v: &[f64], filename: &str) -> Result<(), std::io::Error> {
    let mut w = csv::Writer::from_writer(std::fs::File::create(filename)?);

    w.write_record(["v (V)"])?;

    for i in v {
        w.write_record(&[i.to_string()])?;
    }

    Ok(())
}

pub fn save_vector2(v: &[f64], t: &[f64], filename: &str) -> Result<(), std::io::Error> {
    let mut w = csv::Writer::from_writer(std::fs::File::create(filename)?);

    w.write_record(["t (s)", "v (V)"])?;

    for (v_i, t_i) in v.iter().zip(t.iter()) {
        w.write_record(&[t_i.to_string(), v_i.to_string()])?;
    }

    Ok(())
}
