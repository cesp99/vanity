//! SHA-256 for the `grind` (CreateAccountWithSeed) CPU path.
//!
//! The address is SHA256(base[32] || seed[16] || owner[32]): 80 bytes, two
//! blocks. Block 1 is base, seed and owner[..16]; block 2 is owner[16..] and
//! the padding, so its whole message schedule is fixed for a run.
//!
//! A worker fixes the first 12 seed characters and walks every value of the
//! last 4, which make up W11 and nothing else. Rounds 0-10, round 11 apart
//! from W11, and the parts of block 1's schedule that do not depend on W11
//! are computed once per prefix. Each hash then costs rounds 12-63 of block
//! 1, the W11-dependent schedule words, and the 64 rounds of block 2.
//!
//! Seeds are public, so they come from a counter rather than a CSPRNG. They
//! stay 16 characters of the same alphanumeric alphabet as before.

use crate::fast::MatchTargets;
use crate::TOTAL_ATTEMPTS;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU64, Ordering};

pub const SEED_LEN: usize = 16;
const PREFIX_LEN: usize = 12;

const SEED_ALPHABET: &[u8; 62] =
    b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// Two-character seed pieces, as big-endian ASCII pairs.
const PAIRS: usize = 62 * 62;
/// Seeds per prefix: every value of the last four characters.
const TAILS: usize = PAIRS * PAIRS;
/// Prefixes the counter can reach: characters 2..12 in base 62.
const COUNTED: u64 = 62u64.pow(10);
/// Hashes between checks of the stop flag and the unfilled kinds.
const CHECK_EVERY: usize = 4096;

static PAIR: [u16; PAIRS] = pair_table();

const fn pair_table() -> [u16; PAIRS] {
    let mut t = [0u16; PAIRS];
    let mut i = 0;
    while i < PAIRS {
        t[i] = ((SEED_ALPHABET[i / 62] as u16) << 8)
            | SEED_ALPHABET[i % 62] as u16;
        i += 1;
    }
    t
}

#[cfg(target_arch = "x86_64")]
const IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f,
    0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

#[cfg(target_arch = "x86_64")]
#[rustfmt::skip]
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn ssig0(x: u32) -> u32 {
    x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn ssig1(x: u32) -> u32 {
    x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
}

#[cfg(target_arch = "x86_64")]
fn round(s: &mut [u32; 8], kw: u32) {
    let [a, b, c, d, e, f, g, h] = *s;
    let big_s1 =
        e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
    let t1 = h
        .wrapping_add(big_s1)
        .wrapping_add((e & f) ^ (!e & g))
        .wrapping_add(kw);
    let big_s0 =
        a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
    let t2 = big_s0.wrapping_add((a & b) ^ (a & c) ^ (b & c));
    *s = [t1.wrapping_add(t2), a, b, c, d.wrapping_add(t1), e, f, g];
}

#[cfg(target_arch = "x86_64")]
fn be_words<const N: usize>(bytes: &[u8]) -> [u32; N] {
    core::array::from_fn(|i| {
        u32::from_be_bytes(
            bytes[i * 4..i * 4 + 4]
                .try_into()
                .unwrap(),
        )
    })
}

/// Hands out 12-character seed prefixes. Each one is returned once per
/// space, so threads sharing a space never hash the same seed. The random
/// start keeps separate runs from walking the same seeds.
struct SeedSpace {
    head: [u8; 2],
    start: u64,
    next: AtomicU64,
}

impl SeedSpace {
    fn new() -> Self {
        let r: [u64; 2] = rand::random();
        Self {
            head: [
                SEED_ALPHABET[(r[0] % 62) as usize],
                SEED_ALPHABET[(r[0] / 62 % 62) as usize],
            ],
            start: r[1] % COUNTED,
            next: AtomicU64::new(0),
        }
    }

    fn next_prefix(&self) -> [u8; PREFIX_LEN] {
        let k = self
            .next
            .fetch_add(1, Ordering::Relaxed);
        let mut v = (self.start + k % COUNTED) % COUNTED;
        let mut out = [0u8; PREFIX_LEN];
        out[..2].copy_from_slice(&self.head);
        for c in out[2..].iter_mut().rev() {
            *c = SEED_ALPHABET[(v % 62) as usize];
            v /= 62;
        }
        out
    }
}

/// Everything about a `grind` run that the hash loop needs.
pub struct SeedJob {
    base: [u8; 32],
    owner: [u8; 32],
    /// K + W for block 2, which only holds owner[16..] and padding.
    #[cfg(target_arch = "x86_64")]
    kw2: [u32; 64],
    space: SeedSpace,
}

impl SeedJob {
    pub fn new(base: [u8; 32], owner: [u8; 32]) -> Self {
        Self {
            base,
            owner,
            #[cfg(target_arch = "x86_64")]
            kw2: block2_kw(&owner),
            space: SeedSpace::new(),
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn block2_kw(owner: &[u8; 32]) -> [u32; 64] {
    let mut w = [0u32; 64];
    w[..4].copy_from_slice(&be_words::<4>(&owner[16..]));
    w[4] = 0x8000_0000;
    w[15] = 80 * 8;
    for t in 16..64 {
        w[t] = ssig1(w[t - 2])
            .wrapping_add(w[t - 7])
            .wrapping_add(ssig0(w[t - 15]))
            .wrapping_add(w[t - 16]);
    }
    core::array::from_fn(|t| K[t].wrapping_add(w[t]))
}

/// Block 1 with seed characters 0..12 fixed, waiting for W11.
#[cfg(target_arch = "x86_64")]
struct Midstate {
    /// State after round 11 computed with W11 = 0. W11 only enters through
    /// that round's t1, so the real `a` and `e` are these plus W11.
    s: [u32; 8],
    /// K[t] + W[t] for rounds whose word does not depend on W11.
    kw: [u32; 64],
    /// The part of W[t] that does not depend on W11, for t in 18..40. From
    /// W40 on, every term of the recurrence depends on W11.
    part: [u32; 40],
}

#[cfg(target_arch = "x86_64")]
impl Midstate {
    fn new(
        base: &[u8; 32],
        owner: &[u8; 32],
        prefix: &[u8; PREFIX_LEN],
    ) -> Self {
        let mut block = [0u8; 64];
        block[..32].copy_from_slice(base);
        block[32..44].copy_from_slice(prefix);
        block[48..].copy_from_slice(&owner[..16]);
        let w: [u32; 16] = be_words(&block);

        let mut s = IV;
        for t in 0..12 {
            round(&mut s, K[t].wrapping_add(w[t]));
        }

        // Words that never see W11 (w[11] is 0 here, and unused).
        let w16 = ssig1(w[14])
            .wrapping_add(w[9])
            .wrapping_add(ssig0(w[1]))
            .wrapping_add(w[0]);
        let w17 = ssig1(w[15])
            .wrapping_add(w[10])
            .wrapping_add(ssig0(w[2]))
            .wrapping_add(w[1]);
        let w19 = ssig1(w17)
            .wrapping_add(w[12])
            .wrapping_add(ssig0(w[4]))
            .wrapping_add(w[3]);
        let w21 = ssig1(w19)
            .wrapping_add(w[14])
            .wrapping_add(ssig0(w[6]))
            .wrapping_add(w[5]);
        let w23 = ssig1(w21)
            .wrapping_add(w16)
            .wrapping_add(ssig0(w[8]))
            .wrapping_add(w[7]);

        let mut kw = [0u32; 64];
        for t in 12..16 {
            kw[t] = K[t].wrapping_add(w[t]);
        }
        kw[16] = K[16].wrapping_add(w16);
        kw[17] = K[17].wrapping_add(w17);
        kw[19] = K[19].wrapping_add(w19);
        kw[21] = K[21].wrapping_add(w21);
        kw[23] = K[23].wrapping_add(w23);

        // What each word adds on top of the W11-dependent terms that
        // `hash16` computes; the comments show the full recurrence.
        let mut part = [0u32; 40];
        // s1(W16) + W11 + s0(W3) + W2
        part[18] = ssig1(w16)
            .wrapping_add(ssig0(w[3]))
            .wrapping_add(w[2]);
        // s1(W18) + W13 + s0(W5) + W4
        part[20] = w[13]
            .wrapping_add(ssig0(w[5]))
            .wrapping_add(w[4]);
        // s1(W20) + W15 + s0(W7) + W6
        part[22] = w[15]
            .wrapping_add(ssig0(w[7]))
            .wrapping_add(w[6]);
        // s1(W22) + W17 + s0(W9) + W8
        part[24] = w17
            .wrapping_add(ssig0(w[9]))
            .wrapping_add(w[8]);
        // s1(W23) + W18 + s0(W10) + W9
        part[25] = ssig1(w23)
            .wrapping_add(ssig0(w[10]))
            .wrapping_add(w[9]);
        // s1(W24) + W19 + s0(W11) + W10
        part[26] = w19.wrapping_add(w[10]);
        // s1(W25) + W20 + s0(W12) + W11
        part[27] = ssig0(w[12]);
        // s1(W26) + W21 + s0(W13) + W12
        part[28] = w21
            .wrapping_add(ssig0(w[13]))
            .wrapping_add(w[12]);
        // s1(W27) + W22 + s0(W14) + W13
        part[29] = ssig0(w[14]).wrapping_add(w[13]);
        // s1(W28) + W23 + s0(W15) + W14
        part[30] = w23
            .wrapping_add(ssig0(w[15]))
            .wrapping_add(w[14]);
        // s1(W29) + W24 + s0(W16) + W15
        part[31] = ssig0(w16).wrapping_add(w[15]);
        // s1(W30) + W25 + s0(W17) + W16
        part[32] = ssig0(w17).wrapping_add(w16);
        // s1(W31) + W26 + s0(W18) + W17
        part[33] = w17;
        // s1(W32) + W27 + s0(W19) + W18
        part[34] = ssig0(w19);
        // s1(W33) + W28 + s0(W20) + W19
        part[35] = w19;
        // s1(W34) + W29 + s0(W21) + W20
        part[36] = ssig0(w21);
        // s1(W35) + W30 + s0(W22) + W21
        part[37] = w21;
        // s1(W36) + W31 + s0(W23) + W22
        part[38] = ssig0(w23);
        // s1(W37) + W32 + s0(W24) + W23
        part[39] = w23;

        Self { s, kw, part }
    }
}

// ─── AVX-512 (16 lanes) ──────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
pub fn avx512_available() -> bool {
    is_x86_feature_detected!("avx512f")
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn add(a: __m512i, b: __m512i) -> __m512i {
    _mm512_add_epi32(a, b)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn splat(x: u32) -> __m512i {
    _mm512_set1_epi32(x as i32)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn xor3(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    _mm512_ternarylogic_epi32::<0x96>(a, b, c)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn s0(x: __m512i) -> __m512i {
    xor3(
        _mm512_ror_epi32::<7>(x),
        _mm512_ror_epi32::<18>(x),
        _mm512_srli_epi32::<3>(x),
    )
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn s1(x: __m512i) -> __m512i {
    xor3(
        _mm512_ror_epi32::<17>(x),
        _mm512_ror_epi32::<19>(x),
        _mm512_srli_epi32::<10>(x),
    )
}

/// Block 1's W18..W64 for one group: entry `t - 18` is W[t]. W19, W21 and
/// W23 do not depend on W11; they are in `m.kw` instead and left zero here.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn schedule1(m: &Midstate, w11: __m512i) -> [__m512i; 46] {
    // Literal indices only, so that the array lives in registers.
    let mut w = [_mm512_setzero_si512(); 46];
    macro_rules! wt {
        ($t:expr) => {
            w[$t - 18]
        };
    }
    let p = |t: usize| splat(m.part[t]);
    wt!(18) = add(p(18), w11);
    wt!(20) = add(s1(wt!(18)), p(20));
    wt!(22) = add(s1(wt!(20)), p(22));
    wt!(24) = add(s1(wt!(22)), p(24));
    wt!(25) = add(p(25), wt!(18));
    wt!(26) = add(add(s1(wt!(24)), s0(w11)), p(26));
    wt!(27) = add(add(s1(wt!(25)), wt!(20)), add(p(27), w11));
    wt!(28) = add(s1(wt!(26)), p(28));
    wt!(29) = add(add(s1(wt!(27)), wt!(22)), p(29));
    wt!(30) = add(s1(wt!(28)), p(30));
    wt!(31) = add(add(s1(wt!(29)), wt!(24)), p(31));
    wt!(32) = add(add(s1(wt!(30)), wt!(25)), p(32));
    wt!(33) = add(add(s1(wt!(31)), wt!(26)), add(s0(wt!(18)), p(33)));
    wt!(34) = add(add(s1(wt!(32)), wt!(27)), add(p(34), wt!(18)));
    wt!(35) = add(add(s1(wt!(33)), wt!(28)), add(s0(wt!(20)), p(35)));
    wt!(36) = add(add(s1(wt!(34)), wt!(29)), add(p(36), wt!(20)));
    wt!(37) = add(add(s1(wt!(35)), wt!(30)), add(s0(wt!(22)), p(37)));
    wt!(38) = add(add(s1(wt!(36)), wt!(31)), add(p(38), wt!(22)));
    wt!(39) = add(add(s1(wt!(37)), wt!(32)), add(s0(wt!(24)), p(39)));
    macro_rules! sched {
        ($($t:literal)*) => {$(
            wt!($t) = add(
                add(s1(wt!($t - 2)), wt!($t - 7)),
                add(s0(wt!($t - 15)), wt!($t - 16)),
            );
        )*};
    }
    sched!(40 41 42 43 44 45 46 47 48 49 50 51);
    sched!(52 53 54 55 56 57 58 59 60 61 62 63);
    w
}

/// SHA-256 of `S` groups of 16 seeds. Within a group the seeds differ only in
/// W11; each group has its own prefix. The result is big-endian digest words:
/// `out[s][i]` holds bytes `4 * i..4 * i + 4` of every lane of group `s`.
/// Independent groups give the core more to overlap.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn hash16<const S: usize>(
    ms: [&Midstate; S],
    kw2: &[u32; 64],
    w11: [__m512i; S],
) -> [[__m512i; 8]; S] {
    macro_rules! round {
        ($a:ident, $b:ident, $c:ident, $d:ident,
         $e:ident, $f:ident, $g:ident, $h:ident, $kw:ident, $t:expr) => {
            for i in 0..S {
                let big_s1 = xor3(
                    _mm512_ror_epi32::<6>($e[i]),
                    _mm512_ror_epi32::<11>($e[i]),
                    _mm512_ror_epi32::<25>($e[i]),
                );
                let ch = _mm512_ternarylogic_epi32::<0xCA>(
                    $e[i], $f[i], $g[i],
                );
                let t1 = add(add($h[i], big_s1), add(ch, $kw!($t, i)));
                let big_s0 = xor3(
                    _mm512_ror_epi32::<2>($a[i]),
                    _mm512_ror_epi32::<13>($a[i]),
                    _mm512_ror_epi32::<22>($a[i]),
                );
                let maj = _mm512_ternarylogic_epi32::<0xE8>(
                    $a[i], $b[i], $c[i],
                );
                $d[i] = add($d[i], t1);
                $h[i] = add(t1, add(big_s0, maj));
            }
        };
    }
    // Rotating the names replaces the register moves a round would
    // otherwise end with. `$kw!(t, i)` must give round t's K + W for
    // group i.
    macro_rules! rounds8 {
        ($kw:ident, $t:expr, $a:ident, $b:ident, $c:ident, $d:ident,
         $e:ident, $f:ident, $g:ident, $h:ident) => {
            round!($a, $b, $c, $d, $e, $f, $g, $h, $kw, $t);
            round!($h, $a, $b, $c, $d, $e, $f, $g, $kw, $t + 1);
            round!($g, $h, $a, $b, $c, $d, $e, $f, $kw, $t + 2);
            round!($f, $g, $h, $a, $b, $c, $d, $e, $kw, $t + 3);
            round!($e, $f, $g, $h, $a, $b, $c, $d, $kw, $t + 4);
            round!($d, $e, $f, $g, $h, $a, $b, $c, $kw, $t + 5);
            round!($c, $d, $e, $f, $g, $h, $a, $b, $kw, $t + 6);
            round!($b, $c, $d, $e, $f, $g, $h, $a, $kw, $t + 7);
        };
    }

    // Block 1 schedule. Words without W11 are in `m.kw` already.
    let ws: [[__m512i; 46]; S] =
        core::array::from_fn(|i| schedule1(ms[i], w11[i]));

    macro_rules! kw1 {
        ($t:expr, $i:expr) => {
            match $t {
                12..=17 | 19 | 21 | 23 => splat(ms[$i].kw[$t]),
                _ => add(ws[$i][$t - 18], splat(K[$t])),
            }
        };
    }

    // State after round 11; `a` and `e` still lack W11.
    let state = |j: usize| -> [__m512i; S] {
        core::array::from_fn(|i| splat(ms[i].s[j]))
    };
    let (mut a, mut b, mut c, mut d) =
        (state(0), state(1), state(2), state(3));
    let (mut e, mut f, mut g, mut h) =
        (state(4), state(5), state(6), state(7));
    for i in 0..S {
        a[i] = add(a[i], w11[i]);
        e[i] = add(e[i], w11[i]);
    }

    round!(a, b, c, d, e, f, g, h, kw1, 12);
    round!(h, a, b, c, d, e, f, g, kw1, 13);
    round!(g, h, a, b, c, d, e, f, kw1, 14);
    round!(f, g, h, a, b, c, d, e, kw1, 15);
    // After four rounds the roles have moved by four names.
    let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h) =
        (e, f, g, h, a, b, c, d);
    rounds8!(kw1, 16, a, b, c, d, e, f, g, h);
    rounds8!(kw1, 24, a, b, c, d, e, f, g, h);
    rounds8!(kw1, 32, a, b, c, d, e, f, g, h);
    rounds8!(kw1, 40, a, b, c, d, e, f, g, h);
    rounds8!(kw1, 48, a, b, c, d, e, f, g, h);
    rounds8!(kw1, 56, a, b, c, d, e, f, g, h);

    let mut mid = [[_mm512_setzero_si512(); 8]; S];
    for i in 0..S {
        mid[i] = [
            add(a[i], splat(IV[0])),
            add(b[i], splat(IV[1])),
            add(c[i], splat(IV[2])),
            add(d[i], splat(IV[3])),
            add(e[i], splat(IV[4])),
            add(f[i], splat(IV[5])),
            add(g[i], splat(IV[6])),
            add(h[i], splat(IV[7])),
        ];
    }

    // Block 2: the schedule is fixed, so every round's K + W is too.
    let col = |j: usize| -> [__m512i; S] {
        core::array::from_fn(|i| mid[i][j])
    };
    let (mut a, mut b, mut c, mut d) = (col(0), col(1), col(2), col(3));
    let (mut e, mut f, mut g, mut h) = (col(4), col(5), col(6), col(7));
    macro_rules! kw2 {
        ($t:expr, $i:expr) => {
            splat(kw2[$t])
        };
    }
    rounds8!(kw2, 0, a, b, c, d, e, f, g, h);
    rounds8!(kw2, 8, a, b, c, d, e, f, g, h);
    rounds8!(kw2, 16, a, b, c, d, e, f, g, h);
    rounds8!(kw2, 24, a, b, c, d, e, f, g, h);
    rounds8!(kw2, 32, a, b, c, d, e, f, g, h);
    rounds8!(kw2, 40, a, b, c, d, e, f, g, h);
    rounds8!(kw2, 48, a, b, c, d, e, f, g, h);
    rounds8!(kw2, 56, a, b, c, d, e, f, g, h);

    core::array::from_fn(|i| {
        [
            add(a[i], mid[i][0]),
            add(b[i], mid[i][1]),
            add(c[i], mid[i][2]),
            add(d[i], mid[i][3]),
            add(e[i], mid[i][4]),
            add(f[i], mid[i][5]),
            add(g[i], mid[i][6]),
            add(h[i], mid[i][7]),
        ]
    })
}

/// W11 for tails `n..n + 16`, where tail `n` is characters 12..16 of the
/// seed: pair `n / PAIRS` then pair `n % PAIRS`.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn tail_words(hi: usize, lo: usize) -> __m512i {
    if lo + 16 <= PAIRS {
        let los = _mm512_cvtepu16_epi32(_mm256_loadu_si256(
            PAIR.as_ptr().add(lo) as *const __m256i,
        ));
        let his = _mm512_set1_epi32(((PAIR[hi] as u32) << 16) as i32);
        _mm512_or_si512(his, los)
    } else {
        let words: [u32; 16] =
            core::array::from_fn(|j| tail_word(hi * PAIRS + lo + j));
        _mm512_loadu_si512(words.as_ptr() as *const __m512i)
    }
}

#[inline]
fn tail_word(n: usize) -> u32 {
    ((PAIR[n / PAIRS] as u32) << 16) | PAIR[n % PAIRS] as u32
}

fn seed_of(prefix: &[u8; PREFIX_LEN], tail: usize) -> [u8; SEED_LEN] {
    let mut seed = [0u8; SEED_LEN];
    seed[..PREFIX_LEN].copy_from_slice(prefix);
    seed[PREFIX_LEN..].copy_from_slice(&tail_word(tail).to_be_bytes());
    seed
}

/// The 32-bit form of [`MatchTargets::prefix_filter`]: `(lo, span)` over the
/// digest's first four bytes. Coarser, so still conservative.
#[cfg(any(test, target_arch = "x86_64"))]
fn filter32(
    target: &MatchTargets,
    active: u64,
) -> Option<Vec<(u32, u32)>> {
    let iv = target.prefix_filter(active)?;
    Some(
        iv.iter()
            .map(|&(lo, span)| {
                let lo32 = (lo >> 32) as u32;
                (lo32, ((lo + span) >> 32) as u32 - lo32)
            })
            .collect(),
    )
}

/// `on_hit(seed, address, mask)`, see [`grind_seeds`].
type OnHit<'a> = dyn FnMut(&[u8; SEED_LEN], &[u8; 32], u64) + 'a;

/// Groups of 16 lanes hashed per call, each on its own prefix.
#[cfg(target_arch = "x86_64")]
const GROUPS: usize = 2;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn grind_avx512(
    job: &SeedJob,
    target: &MatchTargets,
    max_count: u32,
    done: &dyn Fn() -> bool,
    on_hit: &mut OnHit,
) {
    let mut active = 0u64;
    let mut filter = None;
    let mut local = 0u64;
    loop {
        let prefixes: [_; GROUPS] =
            core::array::from_fn(|_| job.space.next_prefix());
        let mids: [Midstate; GROUPS] = core::array::from_fn(|i| {
            Midstate::new(&job.base, &job.owner, &prefixes[i])
        });
        let ms: [&Midstate; GROUPS] =
            core::array::from_fn(|i| &mids[i]);
        let (mut hi, mut lo) = (0, 0);
        for n in (0..TAILS).step_by(16) {
            if n % CHECK_EVERY == 0 {
                TOTAL_ATTEMPTS.fetch_add(local, Ordering::Relaxed);
                local = 0;
                if done() {
                    return;
                }
                // Kinds only fill up, so a stale mask is a superset and the
                // credit in `on_hit` has the final say.
                let unfilled = crate::unfilled_mask(max_count);
                if unfilled != active {
                    active = unfilled;
                    filter = filter32(target, active);
                }
            }
            let tails = tail_words(hi, lo);
            let digests = hash16(ms, &job.kw2, [tails; GROUPS]);
            for (group, digest) in digests.iter().enumerate() {
                let candidates = match &filter {
                    Some(iv) => {
                        let mut k = 0u16;
                        for &(start, span) in iv {
                            let off = _mm512_sub_epi32(
                                digest[0],
                                _mm512_set1_epi32(start as i32),
                            );
                            k |= _mm512_cmple_epu32_mask(
                                off,
                                _mm512_set1_epi32(span as i32),
                            );
                        }
                        k
                    }
                    None => u16::MAX,
                };
                if candidates == 0 {
                    continue;
                }
                let mut words = [[0u32; 16]; 8];
                for i in 0..8 {
                    _mm512_storeu_si512(
                        words[i].as_mut_ptr() as *mut __m512i,
                        digest[i],
                    );
                }
                let mut lanes = candidates;
                while lanes != 0 {
                    let lane = lanes.trailing_zeros() as usize;
                    lanes &= lanes - 1;
                    let mut bytes = [0u8; 32];
                    for i in 0..8 {
                        bytes[i * 4..i * 4 + 4].copy_from_slice(
                            &words[i][lane].to_be_bytes(),
                        );
                    }
                    let mask = target.match_mask_active(&bytes, active);
                    if mask != 0 {
                        let seed = seed_of(&prefixes[group], n + lane);
                        on_hit(&seed, &bytes, mask);
                        // Stop as soon as a hit completes the quotas, like
                        // the per-key loop this replaced.
                        if done() {
                            TOTAL_ATTEMPTS.fetch_add(
                                local + (16 * GROUPS) as u64,
                                Ordering::Relaxed,
                            );
                            return;
                        }
                    }
                }
            }
            local += (16 * GROUPS) as u64;
            lo += 16;
            if lo >= PAIRS {
                lo -= PAIRS;
                hi += 1;
            }
        }
    }
}

// ─── portable ────────────────────────────────────────────────────────────────

fn grind_portable(
    job: &SeedJob,
    target: &MatchTargets,
    max_count: u32,
    done: &dyn Fn() -> bool,
    on_hit: &mut OnHit,
) {
    let base_sha = Sha256::new().chain_update(job.base);
    let mut active = 0u64;
    let mut local = 0u64;
    loop {
        let prefix = job.space.next_prefix();
        for n in 0..TAILS {
            if n % CHECK_EVERY == 0 {
                TOTAL_ATTEMPTS.fetch_add(local, Ordering::Relaxed);
                local = 0;
                if done() {
                    return;
                }
                active = crate::unfilled_mask(max_count);
            }
            let seed = seed_of(&prefix, n);
            let bytes: [u8; 32] = base_sha
                .clone()
                .chain_update(seed)
                .chain_update(job.owner)
                .finalize()
                .into();
            local += 1;
            let mask = target.match_mask_active(&bytes, active);
            if mask != 0 {
                on_hit(&seed, &bytes, mask);
                if done() {
                    TOTAL_ATTEMPTS.fetch_add(local, Ordering::Relaxed);
                    return;
                }
            }
        }
    }
}

pub fn backend_name() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    if avx512_available() {
        return "avx512 (16-lane sha256)";
    }
    "sha2"
}

/// Grind seeds on this thread until `done()`. `on_hit(seed, address, mask)`
/// gets every seed whose address matched the unfilled kinds in `mask`. The
/// address comes from the fast path, so the caller must recompute it before
/// trusting it. Attempts are added to `TOTAL_ATTEMPTS`.
pub fn grind_seeds(
    job: &SeedJob,
    target: &MatchTargets,
    max_count: u32,
    done: impl Fn() -> bool,
    mut on_hit: impl FnMut(&[u8; SEED_LEN], &[u8; 32], u64),
) {
    #[cfg(target_arch = "x86_64")]
    if avx512_available() {
        unsafe {
            grind_avx512(job, target, max_count, &done, &mut on_hit)
        };
        return;
    }
    grind_portable(job, target, max_count, &done, &mut on_hit);
}

#[cfg(test)]
mod tests {
    use super::*;
    use solana_pubkey::Pubkey;

    fn rng(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn bytes32(state: &mut u64) -> [u8; 32] {
        let mut out = [0u8; 32];
        for chunk in out.chunks_mut(8) {
            chunk.copy_from_slice(&rng(state).to_le_bytes());
        }
        out
    }

    fn reference(
        base: &[u8; 32],
        seed: &[u8],
        owner: &[u8; 32],
    ) -> [u8; 32] {
        Sha256::new()
            .chain_update(base)
            .chain_update(seed)
            .chain_update(owner)
            .finalize()
            .into()
    }

    fn random_prefix(state: &mut u64) -> [u8; PREFIX_LEN] {
        core::array::from_fn(|_| {
            SEED_ALPHABET[(rng(state) % 62) as usize]
        })
    }

    #[test]
    fn pair_table_is_alphanumeric_and_ordered() {
        for i in 1..PAIRS {
            assert!(PAIR[i - 1] < PAIR[i]);
        }
        for n in [0, 1, 61, 62, PAIRS - 1, PAIRS, TAILS - 1] {
            let w = tail_word(n).to_be_bytes();
            assert!(w
                .iter()
                .all(|c| c.is_ascii_alphanumeric()));
        }
        assert_eq!(&tail_word(0).to_be_bytes(), b"0000");
        assert_eq!(&tail_word(TAILS - 1).to_be_bytes(), b"zzzz");
        assert_eq!(&tail_word(PAIRS + 1).to_be_bytes(), b"0101");
    }

    #[test]
    fn prefixes_are_distinct_alphanumeric() {
        let space = SeedSpace::new();
        let mut seen = std::collections::HashSet::new();
        for _ in 0..10_000 {
            let p = space.next_prefix();
            assert!(p
                .iter()
                .all(|c| SEED_ALPHABET.contains(c)));
            assert_eq!(&p[..2], &space.head);
            assert!(seen.insert(p));
        }
        // The counter wraps inside the counted characters only.
        let space = SeedSpace {
            head: *b"ab",
            start: COUNTED - 1,
            next: AtomicU64::new(0),
        };
        assert_eq!(&space.next_prefix(), b"abzzzzzzzzzz");
        assert_eq!(&space.next_prefix(), b"ab0000000000");
    }

    #[test]
    fn seeds_are_sixteen_alphanumeric_chars() {
        let mut st = 0x1234_5678_9abc_def0u64;
        let prefix = random_prefix(&mut st);
        for _ in 0..1000 {
            let n = (rng(&mut st) as usize) % TAILS;
            let seed = seed_of(&prefix, n);
            assert_eq!(seed.len(), SEED_LEN);
            assert!(seed
                .iter()
                .all(|c| SEED_ALPHABET.contains(c)));
            let s = core::str::from_utf8(&seed).unwrap();
            assert_eq!(s.len(), 16);
        }
    }

    #[test]
    fn filter32_contains_filter64() {
        let t =
            MatchTargets::new(&[("Cesp99", ""), ("zz", "a")], false);
        let wide = t.prefix_filter(u64::MAX).unwrap();
        let narrow = filter32(&t, u64::MAX).unwrap();
        for &(lo, span) in &wide {
            for x in [lo, lo + span / 2, lo + span] {
                let x32 = (x >> 32) as u32;
                assert!(narrow
                    .iter()
                    .any(|&(s, sp)| x32.wrapping_sub(s) <= sp));
            }
        }
        let suffix = MatchTargets::new(&[("", "zz")], false);
        assert!(filter32(&suffix, u64::MAX).is_none());
    }

    #[test]
    fn create_with_seed_is_this_sha256() {
        let mut st = 0xdead_beef_0bad_f00du64;
        for _ in 0..200 {
            let base = bytes32(&mut st);
            let owner = bytes32(&mut st);
            let seed = seed_of(
                &random_prefix(&mut st),
                (rng(&mut st) as usize) % TAILS,
            );
            let want = Pubkey::create_with_seed(
                &Pubkey::new_from_array(base),
                core::str::from_utf8(&seed).unwrap(),
                &Pubkey::new_from_array(owner),
            )
            .unwrap();
            assert_eq!(
                reference(&base, &seed, &owner),
                want.to_bytes()
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn hash16_matches_sha2_and_create_with_seed() {
        if !avx512_available() {
            eprintln!("skip: no avx512f");
            return;
        }
        let mut st = 0xfeed_face_cafe_babeu64;
        for iter in 0..300 {
            let base = bytes32(&mut st);
            let owner = bytes32(&mut st);
            let job = SeedJob::new(base, owner);
            let prefix = random_prefix(&mut st);
            let m = Midstate::new(&base, &owner, &prefix);
            // Include the ends of the tail range and chunks that cross a
            // `hi` boundary (the scalar branch of `tail_words`).
            let n = match iter {
                0 => 0,
                1 => TAILS - 16,
                2 => PAIRS - 4,
                _ => (rng(&mut st) as usize % (TAILS / 16)) * 16,
            };
            let (hi, lo) = (n / PAIRS, n % PAIRS);
            let mut words = [[0u32; 16]; 8];
            unsafe {
                let [out] =
                    hash16([&m], &job.kw2, [tail_words(hi, lo)]);
                for i in 0..8 {
                    _mm512_storeu_si512(
                        words[i].as_mut_ptr() as *mut __m512i,
                        out[i],
                    );
                }
            }
            for lane in 0..16 {
                let seed = seed_of(&prefix, n + lane);
                let mut got = [0u8; 32];
                for i in 0..8 {
                    got[i * 4..i * 4 + 4]
                        .copy_from_slice(&words[i][lane].to_be_bytes());
                }
                assert_eq!(
                    got,
                    reference(&base, &seed, &owner),
                    "iter {iter} lane {lane}"
                );
                if lane % 5 == 0 {
                    let want = Pubkey::create_with_seed(
                        &Pubkey::new_from_array(base),
                        core::str::from_utf8(&seed).unwrap(),
                        &Pubkey::new_from_array(owner),
                    )
                    .unwrap();
                    assert_eq!(got, want.to_bytes());
                }
            }
        }
    }

    /// Run a backend until it reports `want` hits, checking each one.
    fn collect_hits(
        grind: fn(
            &SeedJob,
            &MatchTargets,
            u32,
            &dyn Fn() -> bool,
            &mut OnHit,
        ),
        patterns: &[(&str, &str)],
        want: usize,
    ) -> Vec<([u8; SEED_LEN], [u8; 32], u64)> {
        let mut st = 0x0123_4567_89ab_cdefu64;
        let base = bytes32(&mut st);
        let owner = bytes32(&mut st);
        let job = SeedJob::new(base, owner);
        let target = MatchTargets::new(patterns, false);
        let hits = std::cell::RefCell::new(Vec::new());
        let done = || hits.borrow().len() >= want;
        let mut on_hit =
            |seed: &[u8; SEED_LEN], addr: &[u8; 32], mask| {
                hits.borrow_mut()
                    .push((*seed, *addr, mask));
            };
        grind(&job, &target, u32::MAX, &done, &mut on_hit);
        let hits = hits.into_inner();
        // Which kinds are active comes from the shared quota counters, which
        // other tests reset, so only check that every claimed kind is real.
        for (seed, addr, mask) in &hits {
            assert_eq!(*addr, reference(&base, seed, &owner));
            assert_ne!(*mask, 0);
            let s = fd_bs58::encode_32(*addr);
            for (i, (p, sfx)) in patterns.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    assert!(
                        s.starts_with(p) && s.ends_with(sfx),
                        "{s} {patterns:?}"
                    );
                }
            }
        }
        hits
    }

    // Kind 0 is always active (nothing credits it here), so it decides
    // which matcher path a run exercises.
    #[test]
    fn portable_finds_real_hits() {
        collect_hits(grind_portable, &[("A", ""), ("", "z")], 20);
        collect_hits(grind_portable, &[("", "z"), ("A", "")], 20);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_finds_real_hits() {
        if !avx512_available() {
            eprintln!("skip: no avx512f");
            return;
        }
        let run = |job: &SeedJob,
                   t: &MatchTargets,
                   max: u32,
                   done: &dyn Fn() -> bool,
                   hit: &mut OnHit| unsafe {
            grind_avx512(job, t, max, done, hit)
        };
        // Prefix-only patterns go through the vector filter; a suffix makes
        // every lane take the exact check.
        collect_hits(run, &[("A", ""), ("zz", "")], 20);
        collect_hits(run, &[("", "z"), ("A", "")], 20);
    }
}
