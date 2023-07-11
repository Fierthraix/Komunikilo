use std::f64::consts::PI;

pub fn rrcosfilter(taps: usize, beta: f64, symbol_length: f64, samp_rate: f64) -> Vec<f64> {
    let Ts = symbol_length;
    (0..taps)
        .map(|i| {
            let t = ((i as f64) - (taps as f64) / 2f64) / samp_rate;
            if t == 0f64 {
                1f64 / Ts + beta * (4f64 / PI - 1f64)
            } else if beta != 0f64 && (t == Ts / (2f64 * beta) || Ts == -Ts / (2f64 * beta)) {
                beta / (Ts * 2f64.sqrt())
                    * ((1f64 + 2f64 / PI) * (PI / (4f64 * beta)).sin()
                        + (1f64 - 2f64 / PI) * (PI / (4f64 * beta)).cos())
            } else {
                ((1f64 / Ts) * (PI * t / Ts * (1f64 - beta)).sin()
                    + 4f64 * beta * t / Ts * (PI * t / Ts * (1f64 + beta)).cos())
                    / (PI * t / Ts * (1f64 - (4f64 * beta * t / Ts).powi(2)))
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrcosfilter_test() {
        let taps = rrcosfilter(4, 0.35, 1f64, 1f64);
        assert_eq!(
            taps,
            vec![
                0.05711930601768127,
                -0.08469026591932287,
                1.095633840657307,
                -0.08469026591932287
            ]
        );

        let taps2 = rrcosfilter(101, 0.35, 1f64, 1f64);
        assert_eq!(
            taps2,
            vec![
                -7.669023977254505e-5,
                7.843510716088054e-5,
                9.004815536690988e-6,
                -9.370120347122792e-5,
                7.892757302719291e-5,
                2.732121030775636e-5,
                -0.00011210264803042665,
                7.65551042992531e-5,
                5.0139497596940556e-5,
                -0.00013182616383857463,
                7.03638624512653e-5,
                7.841248515763483e-5,
                -0.00015274770659869406,
                5.904892648036072e-5,
                0.00011341899338553956,
                -0.00017465048256314435,
                4.07650901203614e-5,
                0.00015693258392904716,
                -0.00019715545502326691,
                1.2804802417924928e-5,
                0.000211502181203086,
                -0.00021958682521943583,
                -2.89742782985729e-5,
                0.0002809394151791147,
                -0.00024070034375325632,
                -9.122097197152258e-5,
                0.0003712102360883397,
                -0.0002581040050205789,
                -0.0001852276513967734,
                0.0004921732310906981,
                -0.00026693387872092427,
                -0.00033153903101948207,
                0.0006612474250488546,
                -0.00025654087097179873,
                -0.0005709344662064253,
                0.000911972759085134,
                -0.00020113672154206487,
                -0.0009942888977514754,
                0.0013168453097252683,
                -2.85435618959621e-5,
                -0.0018401830576497225,
                0.0020607101909758687,
                0.0005141654910550923,
                -0.003917195182466115,
                0.0037533146532071344,
                0.0027501561350721324,
                -0.011626617311094805,
                0.009572376754617008,
                0.025614994067311637,
                -0.1351641165601599,
                0.6077736180593346,
                0.6077736180593346,
                -0.1351641165601599,
                0.025614994067311637,
                0.009572376754617008,
                -0.011626617311094805,
                0.0027501561350721324,
                0.0037533146532071344,
                -0.003917195182466115,
                0.0005141654910550923,
                0.0020607101909758687,
                -0.0018401830576497225,
                -2.85435618959621e-5,
                0.0013168453097252683,
                -0.0009942888977514754,
                -0.00020113672154206487,
                0.000911972759085134,
                -0.0005709344662064253,
                -0.00025654087097179873,
                0.0006612474250488546,
                -0.00033153903101948207,
                -0.00026693387872092427,
                0.0004921732310906981,
                -0.0001852276513967734,
                -0.0002581040050205789,
                0.0003712102360883397,
                -9.122097197152258e-5,
                -0.00024070034375325632,
                0.0002809394151791147,
                -2.89742782985729e-5,
                -0.00021958682521943583,
                0.000211502181203086,
                1.2804802417924928e-5,
                -0.00019715545502326691,
                0.00015693258392904716,
                4.07650901203614e-5,
                -0.00017465048256314435,
                0.00011341899338553956,
                5.904892648036072e-5,
                -0.00015274770659869406,
                7.841248515763483e-5,
                7.03638624512653e-5,
                -0.00013182616383857463,
                5.0139497596940556e-5,
                7.65551042992531e-5,
                -0.00011210264803042665,
                2.732121030775636e-5,
                7.892757302719291e-5,
                -9.370120347122792e-5,
                9.004815536690988e-6,
                7.843510716088054e-5
            ]
        );
    }

    #[test]
    fn it_works() {
        let fs = 44100;
        let baud = 900;
        let Nbits = 4000;
        let f0 = 1800;
        let Ns = fs / baud;
        let N = Nbits * Ns;
    }
}
