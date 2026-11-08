#![allow(clippy::missing_safety_doc)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
const IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

#[cfg(target_arch = "x86_64")]
#[rustfmt::skip]
const K: [u64; 80] = [
    0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc,
    0x3956c25bf348b538, 0x59f111f1b605d019, 0x923f82a4af194f9b, 0xab1c5ed5da6d8118,
    0xd807aa98a3030242, 0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235, 0xc19bf174cf692694,
    0xe49b69c19ef14ad2, 0xefbe4786384f25e3, 0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65,
    0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,
    0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2, 0xd5a79147930aa725, 0x06ca6351e003826f, 0x142929670a0e6e70,
    0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
    0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b,
    0xa2bfe8a14cf10364, 0xa81a664bbc423001, 0xc24b8b70d0f89791, 0xc76c51a30654be30,
    0xd192e819d6ef5218, 0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb, 0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
    0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b,
    0xca273eceea26619c, 0xd186b8c721c0c207, 0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178,
    0x06f067aa72176fba, 0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,
    0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817,
];

/// SHA-512 of eight 32-byte messages. `words` holds the messages as big-endian
/// u64 words, and the digest comes back the same way: `state[i]` is digest
/// bytes `i * 8..i * 8 + 8` of each lane, read big-endian.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
#[allow(unused_assignments)] // the last rounds' schedule writes go unread
pub unsafe fn sha512_8_words(words: &[__m512i; 4]) -> [__m512i; 8] {
    #[inline(always)]
    unsafe fn rr<const N: i32>(x: __m512i) -> __m512i {
        _mm512_ror_epi64::<N>(x)
    }
    #[inline(always)]
    unsafe fn xor3(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
        _mm512_ternarylogic_epi64::<0x96>(a, b, c)
    }
    #[inline(always)]
    unsafe fn add(a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi64(a, b)
    }

    // A local array and literal indices let the compiler see the padding
    // (w[4], w[5..15] = 0, w[15]) as constants and drop that work.
    let zero = _mm512_setzero_si512();
    let mut w = [
        words[0],
        words[1],
        words[2],
        words[3],
        _mm512_set1_epi64(0x8000000000000000u64 as i64),
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        _mm512_set1_epi64(256),
    ];

    let mut a = _mm512_set1_epi64(IV[0] as i64);
    let mut b = _mm512_set1_epi64(IV[1] as i64);
    let mut c = _mm512_set1_epi64(IV[2] as i64);
    let mut d = _mm512_set1_epi64(IV[3] as i64);
    let mut e = _mm512_set1_epi64(IV[4] as i64);
    let mut f = _mm512_set1_epi64(IV[5] as i64);
    let mut g = _mm512_set1_epi64(IV[6] as i64);
    let mut h = _mm512_set1_epi64(IV[7] as i64);

    macro_rules! round {
        ($a:ident, $b:ident, $c:ident, $d:ident,
         $e:ident, $f:ident, $g:ident, $h:ident, $t:expr) => {
            let t: usize = $t;
            let wt = if t >= 16 {
                let w15 = w[(t - 15) & 15];
                let w2 = w[(t - 2) & 15];
                let s0 = xor3(
                    rr::<1>(w15),
                    rr::<8>(w15),
                    _mm512_srli_epi64(w15, 7),
                );
                let s1 = xor3(
                    rr::<19>(w2),
                    rr::<61>(w2),
                    _mm512_srli_epi64(w2, 6),
                );
                let wt =
                    add(add(w[t & 15], s0), add(w[(t - 7) & 15], s1));
                w[t & 15] = wt;
                wt
            } else {
                w[t]
            };
            let big_s1 =
                xor3(rr::<14>($e), rr::<18>($e), rr::<41>($e));
            let ch = _mm512_ternarylogic_epi64::<0xCA>($e, $f, $g);
            let kt = _mm512_set1_epi64(K[t] as i64);
            let t1 = add(add(add($h, big_s1), add(ch, kt)), wt);
            let big_s0 =
                xor3(rr::<28>($a), rr::<34>($a), rr::<39>($a));
            let maj = _mm512_ternarylogic_epi64::<0xE8>($a, $b, $c);
            $d = add($d, t1);
            $h = add(t1, add(big_s0, maj));
        };
    }
    // Rotating the names replaces the eight register moves a round would
    // otherwise end with.
    macro_rules! rounds8 {
        ($t:expr) => {
            round!(a, b, c, d, e, f, g, h, $t);
            round!(h, a, b, c, d, e, f, g, $t + 1);
            round!(g, h, a, b, c, d, e, f, $t + 2);
            round!(f, g, h, a, b, c, d, e, $t + 3);
            round!(e, f, g, h, a, b, c, d, $t + 4);
            round!(d, e, f, g, h, a, b, c, $t + 5);
            round!(c, d, e, f, g, h, a, b, $t + 6);
            round!(b, c, d, e, f, g, h, a, $t + 7);
        };
    }
    rounds8!(0);
    rounds8!(8);
    rounds8!(16);
    rounds8!(24);
    rounds8!(32);
    rounds8!(40);
    rounds8!(48);
    rounds8!(56);
    rounds8!(64);
    rounds8!(72);

    [
        add(a, _mm512_set1_epi64(IV[0] as i64)),
        add(b, _mm512_set1_epi64(IV[1] as i64)),
        add(c, _mm512_set1_epi64(IV[2] as i64)),
        add(d, _mm512_set1_epi64(IV[3] as i64)),
        add(e, _mm512_set1_epi64(IV[4] as i64)),
        add(f, _mm512_set1_epi64(IV[5] as i64)),
        add(g, _mm512_set1_epi64(IV[6] as i64)),
        add(h, _mm512_set1_epi64(IV[7] as i64)),
    ]
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use sha2::{Digest, Sha512};

    fn rng(state: &mut u64) -> [u8; 32] {
        let mut out = [0u8; 32];
        for chunk in out.chunks_mut(8) {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            chunk.copy_from_slice(&state.to_le_bytes());
        }
        out
    }

    /// Byte-level wrapper, as the chained grind would see it.
    #[target_feature(enable = "avx512f")]
    unsafe fn sha512_8(inputs: &[[u8; 32]; 8]) -> [[u8; 64]; 8] {
        let mut words = [[0u64; 8]; 4];
        for lane in 0..8 {
            for w in 0..4 {
                words[w][lane] = u64::from_be_bytes(
                    inputs[lane][w * 8..w * 8 + 8]
                        .try_into()
                        .unwrap(),
                );
            }
        }
        let state = sha512_8_words(&[
            _mm512_loadu_si512(words[0].as_ptr() as *const __m512i),
            _mm512_loadu_si512(words[1].as_ptr() as *const __m512i),
            _mm512_loadu_si512(words[2].as_ptr() as *const __m512i),
            _mm512_loadu_si512(words[3].as_ptr() as *const __m512i),
        ]);
        let mut out = [[0u8; 64]; 8];
        for i in 0..8 {
            let mut lanes = [0u64; 8];
            _mm512_storeu_si512(
                lanes.as_mut_ptr() as *mut __m512i,
                state[i],
            );
            for l in 0..8 {
                out[l][i * 8..i * 8 + 8]
                    .copy_from_slice(&lanes[l].to_be_bytes());
            }
        }
        out
    }

    #[test]
    fn sha512_8_matches_sha2() {
        if !crate::fast::simd::available() {
            eprintln!("skip: no avx512");
            return;
        }
        let mut st = 0xfeed_face_cafe_babeu64;
        for _ in 0..1000 {
            let mut ins = [[0u8; 32]; 8];
            for x in ins.iter_mut() {
                *x = rng(&mut st);
            }
            let got = unsafe { sha512_8(&ins) };
            for lane in 0..8 {
                let want: [u8; 64] = Sha512::digest(ins[lane]).into();
                assert_eq!(got[lane], want, "lane {lane}");
            }
        }
    }
}
