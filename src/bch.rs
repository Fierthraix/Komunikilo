use crate::iter::Iter;
use crate::{bools_to_u8s, u8_to_bools, Bit};
use reed_solomon::{Decoder, Encoder};

const DAT_VALUE: usize = 7;
const ECC_VALUE: usize = 5;
const FRAME_VALUE: usize = DAT_VALUE + ECC_VALUE;

/// Should do a (7, 4, 1) BCH Coding.
pub fn bch_stream_encode<I: Iterator<Item = Bit>>(data: I) -> impl Iterator<Item = Bit> {
    bools_to_u8s(data)
        .chunks(DAT_VALUE)
        .flat_map(|chunk| {
            let enc = Encoder::new(ECC_VALUE);
            let encoded: Vec<u8> = Vec::from(&enc.encode(&chunk)[..]);
            encoded.into_iter()
        })
        .flat_map(u8_to_bools)
}

pub fn bch_stream_decode<I: Iterator<Item = Bit>>(data: I) -> impl Iterator<Item = Bit> {
    bools_to_u8s(data)
        .chunks(FRAME_VALUE)
        .flat_map(|chunk| {
            let dec = Decoder::new(ECC_VALUE);
            // let decoded: Vec<u8> = dec.correct(&chunk, None);
            // let decoded = dec.correct(&chunk, None);

            if let Ok(decoded) = dec.correct(&chunk, None) {
                // let decoded = Vec::from(&decoded[..]);
                let decoded = Vec::from(decoded.data());
                decoded.into_iter()
            } else {
                // Just return the corrupted data section...
                let data = Vec::from(&chunk[..chunk.len() - ECC_VALUE]);
                data.into_iter()
            }
        })
        .flat_map(u8_to_bools)
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::Rng;

    #[test]
    fn will_the_real_reed_solomon() {
        let data: Vec<u8> =  Vec::from(b"Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris n");

        // Length of error correction code
        let ecc_len = 32;

        // Create encoder and decoder with
        let enc = Encoder::new(ecc_len);
        let dec = Decoder::new(ecc_len);

        // Encode data
        let encoded = enc.encode(&data[..]);

        // Simulate some transmission errors
        let mut corrupted = *encoded;
        for i in 0..4 {
            corrupted[i] = 0x0;
        }
        corrupted[111] = 0x0;
        corrupted[161] = 0x0;
        corrupted[201] = 0x0;

        // Try to recover data
        let known_erasures = [0];
        let recovered = dec.correct(&corrupted, Some(&known_erasures)).unwrap();

        let orig_str = std::str::from_utf8(&data).unwrap();
        let recv_str = std::str::from_utf8(recovered.data()).unwrap();

        assert_eq!(orig_str, recv_str);
    }

    #[test]
    fn stream_bch_works() {
        let mut rng = rand::thread_rng();
        let num_bits = 100; //1_000_000; // Ensure there will be padding.

        let data: Vec<Bit> = (0..num_bits).map(|_| rng.gen::<Bit>()).collect();

        let encoded: Vec<Bit> = bch_stream_encode(data.iter().cloned()).collect();

        let mut decoded: Vec<Bit> = bch_stream_decode(encoded.clone().into_iter()).collect();

        decoded.drain(data.len()..);
        assert_eq!(data.len(), decoded.len());
        assert_eq!(data, decoded);
    }
}
