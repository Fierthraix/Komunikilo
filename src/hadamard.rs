use crate::Bit;

#[derive(Debug, PartialEq, Eq)]
pub struct HadamardMatrix {
    matrix: Vec<Vec<Bit>>,
}

impl HadamardMatrix {
    pub fn new(n: usize) -> Self {
        assert!((n & (n - 1)) == 0);
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
    pub fn key(&self, n: usize) -> &Vec<Bit> {
        // TODO: FIXME: Make it more obvious there are 2^n-1 codes, instad of 2^n.
        self.matrix.get((n % self.matrix.len()) + 1).unwrap()
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
    use itertools::Itertools;
    use rstest::rstest;

    #[rstest]
    // #[case(2)]
    // #[case(4)]
    #[case(8)]
    #[case(16)]
    #[case(32)]
    #[case(64)]
    #[case(128)]
    #[case(256)]
    // #[case(512)]
    // #[case(1024)]
    // #[case(2048)]
    fn hadamard(#[case] matrix_size: usize) {
        let h = HadamardMatrix::new(matrix_size);

        for keys in h.matrix.iter().combinations(2) {
            let key_1 = keys[0];
            let key_2 = keys[1];

            // Check for orthogonality of the bitsequences.
            let checksum: usize = key_1
                .iter()
                .zip(key_2.iter())
                .map(|(&b1, &b2)| if b1 && b2 { 1 } else { 0 })
                .sum::<usize>()
                % 2;
            assert_eq!(checksum, 0);
        }
    }
}
