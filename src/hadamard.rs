use crate::Bit;

#[derive(Debug, PartialEq, Eq)]
pub struct HadamardMatrix {
    matrix: Vec<Vec<Bit>>,
}

impl HadamardMatrix {
    pub fn new(n: usize) -> Self {
        let mut matrix: Vec<Vec<Bit>> = vec![vec![true; n]; n];

        let mut i1 = 1;
        while i1 < n {
            for i2 in 0..i1 {
                for i3 in 0..i1 {
                    matrix[i1 + i2][i3] = matrix[i2][i3];
                    matrix[i2][i1 + i3] = matrix[i2][i3];
                    matrix[i1 + i2][i1 + i3] = !matrix[i2][i3];
                }
            }
            i1 += i1;
        }

        Self { matrix }
    }
}

impl std::fmt::Display for HadamardMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut out: String = String::new();
        for row in &self.matrix {
            for &bit in row {
                out.push_str(if bit { "T " } else { "F " })
            }
            out.push('\n')
        }
        write!(f, "{}", out)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    fn hadamard_from_str(string: &str) -> HadamardMatrix {
        let mut matrix: Vec<Vec<Bit>> = Vec::new();

        for line in string.split('\n') {
            let row: Vec<Bit> = line
                .chars()
                .filter_map(|chr| match chr {
                    'T' | 't' => Some(true),
                    'F' | 'f' => Some(false),
                    _ => None,
                })
                .collect();
            if !row.is_empty() {
                matrix.push(row)
            }
        }

        // Sanity Check.
        let num_rows = matrix.len();
        for row in &matrix {
            assert_eq!(num_rows, row.len());
        }

        HadamardMatrix { matrix }
    }

    #[test]
    fn it_works() {
        let expected2 = hadamard_from_str(
            "T T
             T F",
        );

        let expected4 = hadamard_from_str(
            "T T T T
             T F T F
             T T F F
             T F F T",
        );

        let expected8 = hadamard_from_str(
            "T T T T T T T T
             T F T F T F T F
             T T F F T T F F
             T F F T T F F T
             T T T T F F F F
             T F T F F T F T
             T T F F F F T T
             T F F T F T T F",
        );

        let expected16 = hadamard_from_str(
            "T T T T T T T T T T T T T T T T
             T F T F T F T F T F T F T F T F
             T T F F T T F F T T F F T T F F
             T F F T T F F T T F F T T F F T
             T T T T F F F F T T T T F F F F
             T F T F F T F T T F T F F T F T
             T T F F F F T T T T F F F F T T
             T F F T F T T F T F F T F T T F
             T T T T T T T T F F F F F F F F
             T F T F T F T F F T F T F T F T
             T T F F T T F F F F T T F F T T
             T F F T T F F T F T T F F T T F
             T T T T F F F F F F F F T T T T
             T F T F F T F T F T F T T F T F
             T T F F F F T T F F T T T T F F
             T F F T F T T F F T T F T F F T",
        );

        let computed2 = HadamardMatrix::new(2);
        let computed4 = HadamardMatrix::new(4);
        let computed8 = HadamardMatrix::new(8);
        let computed16 = HadamardMatrix::new(16);

        assert_eq!(computed2, expected2);
        assert_eq!(computed4, expected4);
        assert_eq!(computed8, expected8);
        assert_eq!(computed16, expected16);
    }
}
