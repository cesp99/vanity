#![allow(clippy::missing_safety_doc)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
use core::mem::MaybeUninit;

/// Eight field elements in radix 2^51, one per lane.
///
/// `madd52` silently truncates its operands to 52 bits, so every limb that
/// reaches [`Fe8::mul`] must be below 2^52. `mul`, `add` and `sub` all return
/// limbs below 2^51 + 2^15, which is also small enough to be the subtrahend
/// of `sub`.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub struct Fe8 {
    pub l: [__m512i; 5],
}

#[cfg(target_arch = "x86_64")]
const M51: u64 = (1u64 << 51) - 1;

pub fn available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512ifma")
            && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("avx512bw")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
impl Fe8 {
    #[cfg(test)]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn from_bytes8(inputs: &[[u8; 32]; 8]) -> Fe8 {
        let mut limbs = [[0u64; 8]; 5];
        for lane in 0..8 {
            let l = bytes_to_limbs51(&inputs[lane]);
            for m in 0..5 {
                limbs[m][lane] = l[m];
            }
        }
        Fe8 {
            l: [
                _mm512_loadu_si512(limbs[0].as_ptr() as *const __m512i),
                _mm512_loadu_si512(limbs[1].as_ptr() as *const __m512i),
                _mm512_loadu_si512(limbs[2].as_ptr() as *const __m512i),
                _mm512_loadu_si512(limbs[3].as_ptr() as *const __m512i),
                _mm512_loadu_si512(limbs[4].as_ptr() as *const __m512i),
            ],
        }
    }

    /// Canonical little-endian encoding of every lane.
    #[target_feature(enable = "avx512f,avx512dq")]
    pub unsafe fn to_bytes8(&self) -> [[u8; 32]; 8] {
        let l = canonical_limbs(self);
        let words = [
            _mm512_or_si512(l[0], _mm512_slli_epi64::<51>(l[1])),
            _mm512_or_si512(
                _mm512_srli_epi64::<13>(l[1]),
                _mm512_slli_epi64::<38>(l[2]),
            ),
            _mm512_or_si512(
                _mm512_srli_epi64::<26>(l[2]),
                _mm512_slli_epi64::<25>(l[3]),
            ),
            _mm512_or_si512(
                _mm512_srli_epi64::<39>(l[3]),
                _mm512_slli_epi64::<12>(l[4]),
            ),
        ];
        let mut out = [[0u8; 32]; 8];
        for i in 0..4 {
            let mut lanes = MaybeUninit::<[u64; 8]>::uninit();
            _mm512_storeu_si512(
                lanes.as_mut_ptr() as *mut __m512i,
                words[i],
            );
            let lanes = lanes.assume_init_ref();
            for lane in 0..8 {
                out[lane][i * 8..i * 8 + 8]
                    .copy_from_slice(&lanes[lane].to_le_bytes());
            }
        }
        out
    }
}

// ─── arithmetic ──────────────────────────────────────────────────────────────

/// One parallel carry pass. Any u64 limbs come out below 2^51 + 2^13, and
/// limb 0 below 2^51 + 19 * 2^13.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn carry(c: [__m512i; 5]) -> [__m512i; 5] {
    let mask = _mm512_set1_epi64(M51 as i64);
    let c19 = _mm512_set1_epi64(19);
    let c0 = _mm512_srli_epi64::<51>(c[0]);
    let c1 = _mm512_srli_epi64::<51>(c[1]);
    let c2 = _mm512_srli_epi64::<51>(c[2]);
    let c3 = _mm512_srli_epi64::<51>(c[3]);
    let c4 = _mm512_srli_epi64::<51>(c[4]);
    // c4 < 2^13, so the 32x32 multiply cannot lose anything.
    [
        _mm512_add_epi64(
            _mm512_and_si512(c[0], mask),
            _mm512_mul_epu32(c4, c19),
        ),
        _mm512_add_epi64(_mm512_and_si512(c[1], mask), c0),
        _mm512_add_epi64(_mm512_and_si512(c[2], mask), c1),
        _mm512_add_epi64(_mm512_and_si512(c[3], mask), c2),
        _mm512_add_epi64(_mm512_and_si512(c[4], mask), c3),
    ]
}

/// q = 1 when the lane's value is >= p, else 0. Input must be below 2p with
/// limbs that do not overflow, which `carry` and `mul` outputs satisfy.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn reduction_q(l: &[__m512i; 5]) -> __m512i {
    let mut q = _mm512_srli_epi64::<51>(_mm512_add_epi64(
        l[0],
        _mm512_set1_epi64(19),
    ));
    q = _mm512_srli_epi64::<51>(_mm512_add_epi64(l[1], q));
    q = _mm512_srli_epi64::<51>(_mm512_add_epi64(l[2], q));
    q = _mm512_srli_epi64::<51>(_mm512_add_epi64(l[3], q));
    _mm512_srli_epi64::<51>(_mm512_add_epi64(l[4], q))
}

/// Fully reduced limbs: the unique representative in [0, p), each below 2^51.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline]
unsafe fn canonical_limbs(a: &Fe8) -> [__m512i; 5] {
    let mask = _mm512_set1_epi64(M51 as i64);
    let mut l = carry(a.l);
    let q = reduction_q(&l);
    // Adding 19q and dropping bit 255 subtracts p when q = 1.
    l[0] = _mm512_add_epi64(
        l[0],
        _mm512_mul_epu32(q, _mm512_set1_epi64(19)),
    );
    l[1] = _mm512_add_epi64(l[1], _mm512_srli_epi64::<51>(l[0]));
    l[0] = _mm512_and_si512(l[0], mask);
    l[2] = _mm512_add_epi64(l[2], _mm512_srli_epi64::<51>(l[1]));
    l[1] = _mm512_and_si512(l[1], mask);
    l[3] = _mm512_add_epi64(l[3], _mm512_srli_epi64::<51>(l[2]));
    l[2] = _mm512_and_si512(l[2], mask);
    l[4] = _mm512_add_epi64(l[4], _mm512_srli_epi64::<51>(l[3]));
    l[3] = _mm512_and_si512(l[3], mask);
    l[4] = _mm512_and_si512(l[4], mask);
    l
}

#[cfg(target_arch = "x86_64")]
impl Fe8 {
    #[target_feature(enable = "avx512f")]
    #[inline]
    pub unsafe fn add(&self, o: &Fe8) -> Fe8 {
        Fe8 {
            l: carry(self.add_uncarried(o).l),
        }
    }

    /// Limbs may exceed 2^52, so this is only valid as the first operand of
    /// `add` or `sub`, never as an input to `mul`.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn add_uncarried(&self, o: &Fe8) -> Fe8 {
        Fe8 {
            l: [
                _mm512_add_epi64(self.l[0], o.l[0]),
                _mm512_add_epi64(self.l[1], o.l[1]),
                _mm512_add_epi64(self.l[2], o.l[2]),
                _mm512_add_epi64(self.l[3], o.l[3]),
                _mm512_add_epi64(self.l[4], o.l[4]),
            ],
        }
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    pub unsafe fn sub(&self, o: &Fe8) -> Fe8 {
        // Add 2*p (limbwise) before subtracting so no lane underflows.
        let two_p = two_p();
        let c = [
            _mm512_sub_epi64(
                _mm512_add_epi64(self.l[0], two_p[0]),
                o.l[0],
            ),
            _mm512_sub_epi64(
                _mm512_add_epi64(self.l[1], two_p[1]),
                o.l[1],
            ),
            _mm512_sub_epi64(
                _mm512_add_epi64(self.l[2], two_p[2]),
                o.l[2],
            ),
            _mm512_sub_epi64(
                _mm512_add_epi64(self.l[3], two_p[3]),
                o.l[3],
            ),
            _mm512_sub_epi64(
                _mm512_add_epi64(self.l[4], two_p[4]),
                o.l[4],
            ),
        ];
        Fe8 { l: carry(c) }
    }

    #[target_feature(enable = "avx512f,avx512ifma,avx512dq")]
    #[inline]
    pub unsafe fn mul(&self, o: &Fe8) -> Fe8 {
        let a = &self.l;
        let b = &o.l;
        // A product's high half has weight 2^52 = 2 * 2^51, so the high
        // halves are summed and doubled before the low halves join them.
        let mut z = [_mm512_setzero_si512(); 10];
        for i in 0..5 {
            for j in 0..5 {
                z[i + j + 1] =
                    _mm512_madd52hi_epu64(z[i + j + 1], a[i], b[j]);
            }
        }
        for k in 1..10 {
            z[k] = _mm512_add_epi64(z[k], z[k]);
        }
        for i in 0..5 {
            for j in 0..5 {
                z[i + j] = _mm512_madd52lo_epu64(z[i + j], a[i], b[j]);
            }
        }
        // Each z[k] < 15 * 2^52, so z[k] + 19 * z[k + 5] < 2^61.
        let c19 = _mm512_set1_epi64(19);
        Fe8 {
            l: carry([
                _mm512_add_epi64(z[0], _mm512_mullo_epi64(z[5], c19)),
                _mm512_add_epi64(z[1], _mm512_mullo_epi64(z[6], c19)),
                _mm512_add_epi64(z[2], _mm512_mullo_epi64(z[7], c19)),
                _mm512_add_epi64(z[3], _mm512_mullo_epi64(z[8], c19)),
                _mm512_add_epi64(z[4], _mm512_mullo_epi64(z[9], c19)),
            ]),
        }
    }

    #[target_feature(enable = "avx512f,avx512ifma,avx512dq")]
    #[inline]
    pub unsafe fn square(&self) -> Fe8 {
        self.mul(self)
    }
}

/// 2p, limbwise. Every limb is below 2^52, so `two_p - x` needs no carry
/// before it can be multiplied.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn two_p() -> [__m512i; 5] {
    [
        _mm512_set1_epi64(((1u64 << 52) - 38) as i64),
        _mm512_set1_epi64(((1u64 << 52) - 2) as i64),
        _mm512_set1_epi64(((1u64 << 52) - 2) as i64),
        _mm512_set1_epi64(((1u64 << 52) - 2) as i64),
        _mm512_set1_epi64(((1u64 << 52) - 2) as i64),
    ]
}

// ─── field constants / invert (8-lane) ───────────────────────────────────────

#[cfg(target_arch = "x86_64")]
impl Fe8 {
    #[cfg(test)]
    #[target_feature(enable = "avx512f")]
    #[inline]
    pub unsafe fn splat_limbs(l: &[u64; 5]) -> Fe8 {
        Fe8 {
            l: [
                _mm512_set1_epi64(l[0] as i64),
                _mm512_set1_epi64(l[1] as i64),
                _mm512_set1_epi64(l[2] as i64),
                _mm512_set1_epi64(l[3] as i64),
                _mm512_set1_epi64(l[4] as i64),
            ],
        }
    }

    #[target_feature(enable = "avx512f,avx512ifma,avx512dq")]
    #[inline]
    unsafe fn pow2k(&self, k: u32) -> Fe8 {
        let mut r = *self;
        for _ in 0..k {
            r = r.square();
        }
        r
    }

    #[target_feature(enable = "avx512f,avx512ifma,avx512dq")]
    pub unsafe fn invert(&self) -> Fe8 {
        let t0 = self.square();
        let t1 = t0.square().square();
        let t2 = self.mul(&t1);
        let t3 = t0.mul(&t2);
        let t4 = t3.square();
        let t5 = t2.mul(&t4);
        let t6 = t5.pow2k(5);
        let t7 = t6.mul(&t5);
        let t8 = t7.pow2k(10);
        let t9 = t8.mul(&t7);
        let t10 = t9.pow2k(20);
        let t11 = t10.mul(&t9);
        let t12 = t11.pow2k(10);
        let t13 = t12.mul(&t7);
        let t14 = t13.pow2k(50);
        let t15 = t14.mul(&t13);
        let t16 = t15.pow2k(100);
        let t17 = t16.mul(&t15);
        let t18 = t17.pow2k(50);
        let t19 = t18.mul(&t13);
        t19.pow2k(5).mul(&t3)
    }
}

#[cfg(target_arch = "x86_64")]
pub fn bytes_to_limbs51(b: &[u8; 32]) -> [u64; 5] {
    let load8 = |i: usize| -> u64 {
        u64::from_le_bytes([
            b[i],
            b[i + 1],
            b[i + 2],
            b[i + 3],
            b[i + 4],
            b[i + 5],
            b[i + 6],
            b[i + 7],
        ])
    };
    [
        load8(0) & M51,
        (load8(6) >> 3) & M51,
        (load8(12) >> 6) & M51,
        (load8(19) >> 1) & M51,
        (load8(24) >> 12) & M51,
    ]
}

// ─── 8-lane Edwards points ───────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub struct Point8 {
    pub x: Fe8,
    pub y: Fe8,
    pub z: Fe8,
    pub t: Fe8,
}

/// Affine Niels form, z = 1 for every lane.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub struct Niels8 {
    pub y_plus_x: Fe8,
    pub y_minus_x: Fe8,
    pub t2d: Fe8,
}

#[cfg(target_arch = "x86_64")]
impl Point8 {
    /// (x, y) from an affine entry. E = 2x and H = 2y, so scaling by Z = 4
    /// makes T = xy * Z = E * H exact. One multiplication, where adding the
    /// entry to the identity would spend seven.
    #[target_feature(enable = "avx512f,avx512ifma,avx512dq")]
    #[inline]
    unsafe fn from_affine(n: &Niels8) -> Point8 {
        let e = n.y_plus_x.sub(&n.y_minus_x);
        let h = n.y_plus_x.add(&n.y_minus_x);
        let zero = _mm512_setzero_si512();
        Point8 {
            x: e.add(&e),
            y: h.add(&h),
            z: Fe8 {
                l: [_mm512_set1_epi64(4), zero, zero, zero, zero],
            },
            t: e.mul(&h),
        }
    }

    /// Mixed addition, seven multiplications. Callers that drop the result's
    /// `t` also drop its multiplication once this is inlined.
    #[target_feature(enable = "avx512f,avx512ifma,avx512dq")]
    #[inline]
    unsafe fn add_affine(&self, n: &Niels8) -> Point8 {
        let y_plus_x = self.y.add(&self.x);
        let y_minus_x = self.y.sub(&self.x);
        let pp = y_plus_x.mul(&n.y_plus_x);
        let mm = y_minus_x.mul(&n.y_minus_x);
        let tt2d = self.t.mul(&n.t2d);
        let zz2 = self.z.add_uncarried(&self.z);
        let cx = pp.sub(&mm);
        let cy = pp.add(&mm);
        let cz = zz2.add(&tt2d);
        let ct = zz2.sub(&tt2d);
        // completed -> extended
        Point8 {
            x: cx.mul(&ct),
            y: cy.mul(&cz),
            z: cz.mul(&ct),
            t: cx.mul(&cy),
        }
    }
}

// A clamped scalar is 2^254 + 8t with t < 2^251, so only t is windowed: the
// low 3 bits save nothing to look up, and 2^254 * B is folded into the top
// window's entries. 251 is prime, so the top window always has fewer than W
// bits and its digit (at most 2^bits, it keeps the final carry) fits in the
// table without a sign.
//
// Throughput keeps rising with W even as nearly every lookup misses L3, because
// an entry is prefetched one addition ahead of its use: fewer additions beat
// the misses. W = 20 costs about 800 MB; W = 21 costs twice that for 3% more.
#[cfg(target_arch = "x86_64")]
const SIMD_W: usize = 20; // window radix = 2^SIMD_W
#[cfg(target_arch = "x86_64")]
const SIMD_FREE_BITS: usize = 251;
#[cfg(target_arch = "x86_64")]
const SIMD_WINDOWS: usize = SIMD_FREE_BITS.div_ceil(SIMD_W);
#[cfg(target_arch = "x86_64")]
const SIMD_POS: usize = 1 << (SIMD_W - 1); // largest digit magnitude
#[cfg(target_arch = "x86_64")]
const SIMD_ENTRIES: usize = SIMD_POS + 1; // digit magnitudes 0..=POS
#[cfg(target_arch = "x86_64")]
const SIMD_TOP_BITS: usize =
    SIMD_FREE_BITS - SIMD_W * (SIMD_WINDOWS - 1);
#[cfg(target_arch = "x86_64")]
const SIMD_TOP_ENTRIES: usize = (1 << SIMD_TOP_BITS) + 1;
#[cfg(target_arch = "x86_64")]
const SIMD_ENTRY_SHIFT: u32 = 7; // 128-byte entries, two cache lines
#[cfg(target_arch = "x86_64")]
const SIMD_WINDOW_BYTES: usize = SIMD_ENTRIES << SIMD_ENTRY_SHIFT;
#[cfg(target_arch = "x86_64")]
const _: () = assert!(SIMD_TOP_BITS < SIMD_W);

/// y+x, y-x, 2dxy as canonical limbs, plus one unused word.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
#[repr(C, align(128))]
struct SimdEntry([u64; 16]);

/// Multiples 0..=POS of 8 * 2^(W*i) * B for each window i, normalized to
/// affine. The top window's are offset by 2^254 * B.
#[cfg(target_arch = "x86_64")]
pub struct SimdTable(Box<[SimdEntry]>);

#[cfg(target_arch = "x86_64")]
pub fn build_simd_table() -> SimdTable {
    use crate::fast::field::{batch_invert, Fe};
    use crate::fast::group::{edwards_d2, Point};
    use crate::fast::BASE;
    use rayon::prelude::*;

    let d2 = edwards_d2();
    let mut top_bit = BASE; // 2^254 * B
    for _ in 0..254 {
        top_bit = top_bit.double();
    }
    let mut steps = Vec::with_capacity(SIMD_WINDOWS);
    let mut cur = BASE.double().double().double(); // 8 * 2^(W*i) * B
    for _ in 0..SIMD_WINDOWS {
        steps.push(cur);
        for _ in 0..SIMD_W {
            cur = cur.double();
        }
    }

    // The top window is last, so its shorter run leaves the others' offsets
    // alone and arrives as the short final chunk below.
    let mut table = vec![
        SimdEntry([0u64; 16]);
        (SIMD_WINDOWS - 1) * SIMD_ENTRIES
            + SIMD_TOP_ENTRIES
    ]
    .into_boxed_slice();
    table
        .par_chunks_mut(SIMD_ENTRIES)
        .enumerate()
        .for_each(|(window, entries)| {
            let step = steps[window].as_niels(&d2);
            let mut mult = if window == SIMD_WINDOWS - 1 {
                top_bit
            } else {
                Point::IDENTITY
            };
            // Short inversion batches keep the transient points far smaller
            // than the table, at the price of a few more inversions.
            for chunk in entries.chunks_mut(4096) {
                let mut points = Vec::with_capacity(chunk.len());
                for _ in 0..chunk.len() {
                    points.push(mult);
                    mult = mult.add_niels(&step);
                }
                let mut zinv: Vec<Fe> =
                    points.iter().map(|p| p.z).collect();
                batch_invert(&mut zinv);
                for (i, entry) in chunk.iter_mut().enumerate() {
                    let x = points[i].x.mul(&zinv[i]);
                    let y = points[i].y.mul(&zinv[i]);
                    let fields =
                        [y.add(&x), y.sub(&x), x.mul(&y).mul(&d2)];
                    for (f, field) in fields.iter().enumerate() {
                        let limbs = bytes_to_limbs51(&field.to_bytes());
                        entry.0[f * 5..f * 5 + 5]
                            .copy_from_slice(&limbs);
                    }
                }
            }
        });
    SimdTable(table)
}

/// Where one window's table entry lives for each lane.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct WindowDigit {
    entries: [*const __m512i; 8],
    negative: __mmask8,
}

/// Signed-digit recoding of `scalars` (little-endian u64 words). Only bits
/// 3..254 are read, so the scalars need not be clamped first.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline]
unsafe fn recode(
    scalars: &[__m512i; 4],
    table: &SimdTable,
    out: &mut [MaybeUninit<WindowDigit>; SIMD_WINDOWS],
) {
    let words = [
        scalars[0],
        scalars[1],
        scalars[2],
        _mm512_and_si512(
            scalars[3],
            _mm512_set1_epi64(((1u64 << 62) - 1) as i64),
        ),
    ];
    let digit_mask = _mm512_set1_epi64(((1u64 << SIMD_W) - 1) as i64);
    let pos = _mm512_set1_epi64(SIMD_POS as i64);
    let base = table.0.as_ptr() as usize;
    let mut digit_carry = _mm512_setzero_si512();
    for i in 0..SIMD_WINDOWS {
        let bit = 3 + SIMD_W * i;
        let word = bit / 64;
        let off = bit % 64;
        let mut d = _mm512_srl_epi64(
            words[word],
            _mm_cvtsi32_si128(off as i32),
        );
        if off + SIMD_W > 64 && word < 3 {
            d = _mm512_or_si512(
                d,
                _mm512_sll_epi64(
                    words[word + 1],
                    _mm_cvtsi32_si128((64 - off) as i32),
                ),
            );
        }
        d = _mm512_add_epi64(
            _mm512_and_si512(d, digit_mask),
            digit_carry,
        );
        if i + 1 < SIMD_WINDOWS {
            // Fold digits >= 2^(W-1) down, carrying up.
            digit_carry = _mm512_srli_epi64::<{ SIMD_W as u32 }>(
                _mm512_add_epi64(d, pos),
            );
            d = _mm512_sub_epi64(
                d,
                _mm512_slli_epi64::<{ SIMD_W as u32 }>(digit_carry),
            );
        }
        let window_base =
            _mm512_set1_epi64((base + i * SIMD_WINDOW_BYTES) as i64);
        let entries = _mm512_add_epi64(
            window_base,
            _mm512_slli_epi64::<SIMD_ENTRY_SHIFT>(_mm512_abs_epi64(d)),
        );
        let slot = out[i].as_mut_ptr();
        _mm512_storeu_si512(
            core::ptr::addr_of_mut!((*slot).entries) as *mut __m512i,
            entries,
        );
        core::ptr::addr_of_mut!((*slot).negative)
            .write(_mm512_movepi64_mask(d));
    }
}

/// Transpose eight rows of eight words.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn transpose8(r: [__m512i; 8]) -> [__m512i; 8] {
    let t0 = _mm512_unpacklo_epi64(r[0], r[1]);
    let t1 = _mm512_unpackhi_epi64(r[0], r[1]);
    let t2 = _mm512_unpacklo_epi64(r[2], r[3]);
    let t3 = _mm512_unpackhi_epi64(r[2], r[3]);
    let t4 = _mm512_unpacklo_epi64(r[4], r[5]);
    let t5 = _mm512_unpackhi_epi64(r[4], r[5]);
    let t6 = _mm512_unpacklo_epi64(r[6], r[7]);
    let t7 = _mm512_unpackhi_epi64(r[6], r[7]);
    let u0 = _mm512_shuffle_i64x2::<0x44>(t0, t2);
    let u1 = _mm512_shuffle_i64x2::<0x44>(t1, t3);
    let u2 = _mm512_shuffle_i64x2::<0xEE>(t0, t2);
    let u3 = _mm512_shuffle_i64x2::<0xEE>(t1, t3);
    let u4 = _mm512_shuffle_i64x2::<0x44>(t4, t6);
    let u5 = _mm512_shuffle_i64x2::<0x44>(t5, t7);
    let u6 = _mm512_shuffle_i64x2::<0xEE>(t4, t6);
    let u7 = _mm512_shuffle_i64x2::<0xEE>(t5, t7);
    [
        _mm512_shuffle_i64x2::<0x88>(u0, u4),
        _mm512_shuffle_i64x2::<0x88>(u1, u5),
        _mm512_shuffle_i64x2::<0xDD>(u0, u4),
        _mm512_shuffle_i64x2::<0xDD>(u1, u5),
        _mm512_shuffle_i64x2::<0x88>(u2, u6),
        _mm512_shuffle_i64x2::<0x88>(u3, u7),
        _mm512_shuffle_i64x2::<0xDD>(u2, u6),
        _mm512_shuffle_i64x2::<0xDD>(u3, u7),
    ]
}

#[cfg(target_arch = "x86_64")]
impl Niels8 {
    /// Whole-entry loads and a transpose. A negative digit is the same entry
    /// with y+x and y-x swapped and t2d negated.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn load(digit: &WindowDigit) -> Niels8 {
        let e = &digit.entries;
        let a = transpose8([
            _mm512_load_si512(e[0]),
            _mm512_load_si512(e[1]),
            _mm512_load_si512(e[2]),
            _mm512_load_si512(e[3]),
            _mm512_load_si512(e[4]),
            _mm512_load_si512(e[5]),
            _mm512_load_si512(e[6]),
            _mm512_load_si512(e[7]),
        ]);
        let b = transpose8([
            _mm512_load_si512(e[0].add(1)),
            _mm512_load_si512(e[1].add(1)),
            _mm512_load_si512(e[2].add(1)),
            _mm512_load_si512(e[3].add(1)),
            _mm512_load_si512(e[4].add(1)),
            _mm512_load_si512(e[5].add(1)),
            _mm512_load_si512(e[6].add(1)),
            _mm512_load_si512(e[7].add(1)),
        ]);
        let neg = digit.negative;
        let two_p = two_p();
        let swap = |p: __m512i, m: __m512i| {
            (
                _mm512_mask_blend_epi64(neg, p, m),
                _mm512_mask_blend_epi64(neg, m, p),
            )
        };
        let (p0, m0) = swap(a[0], a[5]);
        let (p1, m1) = swap(a[1], a[6]);
        let (p2, m2) = swap(a[2], a[7]);
        let (p3, m3) = swap(a[3], b[0]);
        let (p4, m4) = swap(a[4], b[1]);
        Niels8 {
            y_plus_x: Fe8 {
                l: [p0, p1, p2, p3, p4],
            },
            y_minus_x: Fe8 {
                l: [m0, m1, m2, m3, m4],
            },
            t2d: Fe8 {
                l: [
                    _mm512_mask_sub_epi64(b[2], neg, two_p[0], b[2]),
                    _mm512_mask_sub_epi64(b[3], neg, two_p[1], b[3]),
                    _mm512_mask_sub_epi64(b[4], neg, two_p[2], b[4]),
                    _mm512_mask_sub_epi64(b[5], neg, two_p[3], b[5]),
                    _mm512_mask_sub_epi64(b[6], neg, two_p[4], b[6]),
                ],
            },
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn prefetch_entries(digit: &WindowDigit) {
    for lane in 0..8 {
        let ptr = digit.entries[lane] as *const i8;
        _mm_prefetch::<_MM_HINT_T0>(ptr);
        _mm_prefetch::<_MM_HINT_T0>(ptr.add(64));
    }
}

/// `scalars` are the clamped ed25519 scalars as little-endian u64 words. The
/// clamp's fixed bits are assumed, not read (see [`recode`]). The result's
/// `t` is not computed.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f,avx512ifma,avx512dq")]
pub unsafe fn scalarmult8(
    scalars: &[__m512i; 4],
    table: &SimdTable,
) -> (Fe8, Fe8, Fe8) {
    let mut digits =
        [const { MaybeUninit::<WindowDigit>::uninit() }; SIMD_WINDOWS];
    recode(scalars, table, &mut digits);
    let digit = |i: usize| digits[i].assume_init_ref();

    prefetch_entries(digit(1));
    let mut acc = Point8::from_affine(&Niels8::load(digit(0)));
    for i in 1..SIMD_WINDOWS - 1 {
        prefetch_entries(digit(i + 1));
        acc = acc.add_affine(&Niels8::load(digit(i)));
    }
    let last = acc.add_affine(&Niels8::load(digit(SIMD_WINDOWS - 1)));
    (last.x, last.y, last.z)
}

/// Big-endian u64 lanes of `a`'s canonical encoding's first eight bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw")]
#[inline]
unsafe fn canonical_first8(a: &Fe8) -> __m512i {
    let q = reduction_q(&a.l);
    let l0 = _mm512_add_epi64(
        a.l[0],
        _mm512_mul_epu32(q, _mm512_set1_epi64(19)),
    );
    let l1 = _mm512_add_epi64(a.l[1], _mm512_srli_epi64::<51>(l0));
    let mask = _mm512_set1_epi64(M51 as i64);
    bswap64(_mm512_or_si512(
        _mm512_and_si512(l0, mask),
        _mm512_slli_epi64::<51>(l1),
    ))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
pub unsafe fn bswap64(x: __m512i) -> __m512i {
    _mm512_shuffle_epi8(
        x,
        _mm512_set4_epi64(
            0x08090a0b0c0d0e0f,
            0x0001020304050607,
            0x08090a0b0c0d0e0f,
            0x0001020304050607,
        ),
    )
}

/// Lanes whose first eight encoded bytes fall in any `(lo, span)` interval.
/// `y` must be a `mul` output. Only the sign bit in the last byte depends on
/// x, so this needs no x at all.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw")]
#[inline]
unsafe fn filter_mask(y: &Fe8, intervals: &[(u64, u64)]) -> u8 {
    let first8 = canonical_first8(y);
    let mut mask = 0u8;
    for &(lo, span) in intervals {
        // Unsigned wraparound: below `lo` becomes larger than any span.
        let off =
            _mm512_sub_epi64(first8, _mm512_set1_epi64(lo as i64));
        mask |= _mm512_cmple_epu64_mask(
            off,
            _mm512_set1_epi64(span as i64),
        );
    }
    mask
}

/// The compressed ed25519 encoding: y, with x's parity in the top bit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline]
unsafe fn compress_affine(x: &Fe8, y: &Fe8) -> [[u8; 32]; 8] {
    let xbytes = x.to_bytes8();
    let mut out = y.to_bytes8();
    for lane in 0..8 {
        out[lane][31] |= (xbytes[lane][0] & 1) << 7;
    }
    out
}

#[cfg(all(test, target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512ifma,avx512dq")]
pub unsafe fn compress8(x: &Fe8, y: &Fe8, z: &Fe8) -> [[u8; 32]; 8] {
    let recip = z.invert();
    compress_affine(&x.mul(&recip), &y.mul(&recip))
}

#[cfg(target_arch = "x86_64")]
pub const SIMD_GROUPS: usize = 64;
#[cfg(target_arch = "x86_64")]
pub const SIMD_BATCH: usize = SIMD_GROUPS * 8;

/// Seeds as the big-endian message words SHA-512 consumes:
/// `[group][word][lane]`. The digest's upper half is the next seed in exactly
/// this form, so chaining needs no byte shuffling at all.
#[cfg(target_arch = "x86_64")]
type SeedWords = [[[u64; 8]; 4]; SIMD_GROUPS];

/// One thread's working memory for grinding `SIMD_BATCH` keys at a time.
#[cfg(target_arch = "x86_64")]
pub struct KeyBatch {
    chains: [SeedWords; 2],
    used: usize,
    xs: [Fe8; SIMD_GROUPS],
    ys: [Fe8; SIMD_GROUPS],
    zs: [Fe8; SIMD_GROUPS],
    prefix: [Fe8; SIMD_GROUPS],
    /// Lanes that passed the filter in each group.
    pub hits: [u8; SIMD_GROUPS],
    /// Valid only for lanes set in `hits`.
    pub pubkeys: [[[u8; 32]; 8]; SIMD_GROUPS],
}

#[cfg(target_arch = "x86_64")]
impl KeyBatch {
    /// All seeds zero until set through [`KeyBatch::set_next_seed`].
    pub fn new() -> Box<KeyBatch> {
        // All-zero is a valid value of every field.
        Box::new(unsafe { MaybeUninit::zeroed().assume_init() })
    }

    /// The seed behind `pubkeys[group][lane]` from the latest `run`.
    pub fn used_seed(&self, group: usize, lane: usize) -> [u8; 32] {
        let words = &self.chains[self.used][group];
        let mut seed = [0u8; 32];
        for w in 0..4 {
            seed[w * 8..w * 8 + 8]
                .copy_from_slice(&words[w][lane].to_be_bytes());
        }
        seed
    }

    /// Replace what the next `run` hashes for this lane.
    pub fn set_next_seed(
        &mut self,
        group: usize,
        lane: usize,
        seed: &[u8; 32],
    ) {
        let words = &mut self.chains[self.used ^ 1][group];
        for w in 0..4 {
            words[w][lane] = u64::from_be_bytes(
                seed[w * 8..w * 8 + 8]
                    .try_into()
                    .unwrap(),
            );
        }
    }

    /// Hash the pending seeds into scalars and the next chained seeds, then
    /// derive every public key's y. Lanes passing `intervals`, or all lanes
    /// when there are none to filter on, also get `pubkeys` filled in.
    #[target_feature(enable = "avx512f,avx512ifma,avx512dq,avx512bw")]
    pub unsafe fn run(
        &mut self,
        table: &SimdTable,
        intervals: Option<&[(u64, u64)]>,
    ) {
        use crate::fast::sha512_simd::sha512_8_words;

        self.used ^= 1;
        let used = self.used;
        for g in 0..SIMD_GROUPS {
            let words = &self.chains[used][g];
            let digest = sha512_8_words(&[
                _mm512_loadu_si512(words[0].as_ptr() as *const __m512i),
                _mm512_loadu_si512(words[1].as_ptr() as *const __m512i),
                _mm512_loadu_si512(words[2].as_ptr() as *const __m512i),
                _mm512_loadu_si512(words[3].as_ptr() as *const __m512i),
            ]);
            let next = &mut self.chains[used ^ 1][g];
            for w in 0..4 {
                _mm512_storeu_si512(
                    next[w].as_mut_ptr() as *mut __m512i,
                    digest[4 + w],
                );
            }
            let (x, y, z) = scalarmult8(
                &[
                    bswap64(digest[0]),
                    bswap64(digest[1]),
                    bswap64(digest[2]),
                    bswap64(digest[3]),
                ],
                table,
            );
            self.xs[g] = x;
            self.ys[g] = y;
            self.zs[g] = z;
            self.prefix[g] = if g == 0 {
                z
            } else {
                self.prefix[g - 1].mul(&z)
            };
        }
        // One inversion for the whole batch (Montgomery's trick).
        let mut inv = self.prefix[SIMD_GROUPS - 1].invert();
        for g in (0..SIMD_GROUPS).rev() {
            let zinv = if g == 0 {
                inv
            } else {
                inv.mul(&self.prefix[g - 1])
            };
            let y = self.ys[g].mul(&zinv);
            let hits = match intervals {
                Some(i) => filter_mask(&y, i),
                None => 0xff,
            };
            self.hits[g] = hits;
            if hits != 0 {
                let x = self.xs[g].mul(&zinv);
                self.pubkeys[g] = compress_affine(&x, &y);
            }
            if g != 0 {
                inv = inv.mul(&self.zs[g]);
            }
        }
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::fast::field::Fe;

    // Simple deterministic PRNG for test inputs.
    fn rng_bytes(state: &mut u64) -> [u8; 32] {
        let mut out = [0u8; 32];
        for chunk in out.chunks_mut(8) {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            chunk.copy_from_slice(&state.to_le_bytes());
        }
        out
    }

    fn fill8(state: &mut u64) -> [[u8; 32]; 8] {
        let mut a = [[0u8; 32]; 8];
        for x in a.iter_mut() {
            *x = rng_bytes(state);
        }
        a
    }

    #[test]
    fn roundtrip_bytes() {
        if !available() {
            eprintln!("skip: no IFMA");
            return;
        }
        let mut st = 0x1234_5678_9abc_def1u64;
        for _ in 0..500 {
            let ins = fill8(&mut st);
            // Reduce inputs to canonical form so comparison is well-defined.
            let expect: Vec<[u8; 32]> = ins
                .iter()
                .map(|b| Fe::from_bytes(b).to_bytes())
                .collect();
            let got = unsafe { Fe8::from_bytes8(&ins).to_bytes8() };
            for lane in 0..8 {
                assert_eq!(
                    got[lane],
                    expect[lane].as_slice(),
                    "roundtrip lane {lane}"
                );
            }
        }
    }

    fn check_binop(
        scalar: impl Fn(&Fe, &Fe) -> Fe,
        simd: unsafe fn(&Fe8, &Fe8) -> Fe8,
        name: &str,
    ) {
        if !available() {
            eprintln!("skip {name}: no IFMA");
            return;
        }
        let mut st = 0xdead_beef_0000_0001u64;
        for _ in 0..2000 {
            let a = fill8(&mut st);
            let b = fill8(&mut st);
            let mut expect = [[0u8; 32]; 8];
            for lane in 0..8 {
                let fa = Fe::from_bytes(&a[lane]);
                let fb = Fe::from_bytes(&b[lane]);
                expect[lane] = scalar(&fa, &fb).to_bytes();
            }
            let got = unsafe {
                let fa = Fe8::from_bytes8(&a);
                let fb = Fe8::from_bytes8(&b);
                simd(&fa, &fb).to_bytes8()
            };
            for lane in 0..8 {
                assert_eq!(
                    got[lane], expect[lane],
                    "{name} lane {lane}"
                );
            }
        }
    }

    #[test]
    fn mul_matches_scalar() {
        check_binop(|a, b| a.mul(b), |a, b| unsafe { a.mul(b) }, "mul");
    }

    #[test]
    fn add_matches_scalar() {
        check_binop(|a, b| a.add(b), |a, b| unsafe { a.add(b) }, "add");
    }

    // Chained multiply: feeds a mul's output back into another mul, which is
    // where an out-of-range limb (>= 2^52) would surface as truncation.
    #[test]
    fn mul_chained_matches_scalar() {
        use crate::fast::field::Fe;
        if !available() {
            return;
        }
        let mut st = 0x1357_9bdf_2468_ace0u64;
        for _ in 0..1000 {
            let a = fill8(&mut st);
            let b = fill8(&mut st);
            let c = fill8(&mut st);
            let got = unsafe {
                Fe8::from_bytes8(&a)
                    .mul(&Fe8::from_bytes8(&b))
                    .mul(&Fe8::from_bytes8(&c))
                    .to_bytes8()
            };
            for lane in 0..8 {
                let expect = Fe::from_bytes(&a[lane])
                    .mul(&Fe::from_bytes(&b[lane]))
                    .mul(&Fe::from_bytes(&c[lane]))
                    .to_bytes();
                assert_eq!(got[lane], expect, "lane {lane}");
            }
        }
    }

    #[test]
    fn sub_matches_scalar() {
        check_binop(|a, b| a.sub(b), |a, b| unsafe { a.sub(b) }, "sub");
    }

    #[test]
    fn square_matches_scalar() {
        if !available() {
            return;
        }
        let mut st = 0x0f0f_0f0f_1234_5678u64;
        for _ in 0..2000 {
            let a = fill8(&mut st);
            let mut expect = [[0u8; 32]; 8];
            for lane in 0..8 {
                expect[lane] = Fe::from_bytes(&a[lane])
                    .square()
                    .to_bytes();
            }
            let got = unsafe {
                Fe8::from_bytes8(&a)
                    .square()
                    .to_bytes8()
            };
            for lane in 0..8 {
                assert_eq!(
                    got[lane], expect[lane],
                    "square lane {lane}"
                );
            }
        }
    }

    #[test]
    fn invert_matches_scalar() {
        if !available() {
            return;
        }
        let mut st = 0xabcd_1234_5678_9999u64;
        for _ in 0..500 {
            let a = fill8(&mut st);
            let mut expect = [[0u8; 32]; 8];
            for lane in 0..8 {
                expect[lane] = Fe::from_bytes(&a[lane])
                    .invert()
                    .to_bytes();
            }
            let got = unsafe {
                Fe8::from_bytes8(&a)
                    .invert()
                    .to_bytes8()
            };
            for lane in 0..8 {
                assert_eq!(
                    got[lane], expect[lane],
                    "invert lane {lane}"
                );
            }
        }
    }

    fn affine8(p: &crate::fast::group::Point) -> Niels8 {
        use crate::fast::group::edwards_d2;
        let n = p.as_niels(&edwards_d2());
        let to8 = |f: &Fe| unsafe {
            Fe8::splat_limbs(&bytes_to_limbs51(&f.to_bytes()))
        };
        Niels8 {
            y_plus_x: to8(&n.y_plus_x),
            y_minus_x: to8(&n.y_minus_x),
            t2d: to8(&n.t2d),
        }
    }

    fn compress_point8(p: &Point8) -> [[u8; 32]; 8] {
        unsafe { compress8(&p.x, &p.y, &p.z) }
    }

    fn basepoint_bytes() -> [u8; 32] {
        let mut expect = [0x66u8; 32];
        expect[0] = 0x58;
        expect
    }

    #[test]
    fn add_affine_identity_plus_base() {
        use crate::fast::group::Point;
        if !available() {
            return;
        }
        let base = affine8(&crate::fast::BASE);
        let ident = affine8(&Point::IDENTITY);
        let got =
            unsafe { Point8::from_affine(&ident).add_affine(&base) };
        assert_eq!(compress_point8(&got)[0], basepoint_bytes());
        // ...and the T it leaves behind must work for a further add.
        let twice = unsafe { got.add_affine(&base) };
        assert_eq!(
            compress_point8(&twice)[0],
            crate::fast::scalarmult_compress_scalar(&{
                let mut k = [0u8; 32];
                k[0] = 2;
                k
            }),
        );
    }

    #[test]
    fn from_affine_of_basepoint() {
        if !available() {
            return;
        }
        let base = affine8(&crate::fast::BASE);
        let p = unsafe { Point8::from_affine(&base) };
        assert_eq!(compress_point8(&p)[0], basepoint_bytes());
        // Its T was derived, so check it by adding through it too.
        let next = unsafe { p.add_affine(&base) };
        let mut k = [0u8; 32];
        k[0] = 2;
        assert_eq!(
            compress_point8(&next)[0],
            crate::fast::scalarmult_compress_scalar(&k),
        );
    }

    #[test]
    fn compress8_of_basepoint() {
        if !available() {
            return;
        }
        // Broadcast the scalar BASE point into all 8 lanes, compress.
        let b = crate::fast::BASE;
        let to8 = |f: &Fe| unsafe {
            Fe8::splat_limbs(&bytes_to_limbs51(&f.to_bytes()))
        };
        let got =
            unsafe { compress8(&to8(&b.x), &to8(&b.y), &to8(&b.z)) };
        assert_eq!(got[0], basepoint_bytes(), "compress8 basepoint");
    }

    #[test]
    fn compress8_of_identity() {
        if !available() {
            return;
        }
        let to8 = |f: &Fe| unsafe {
            Fe8::splat_limbs(&bytes_to_limbs51(&f.to_bytes()))
        };
        let got = unsafe {
            compress8(&to8(&Fe::ZERO), &to8(&Fe::ONE), &to8(&Fe::ONE))
        };
        let mut expect = [0u8; 32];
        expect[0] = 1;
        assert_eq!(got[0], expect, "compress8 identity");
    }

    #[target_feature(enable = "avx512f,avx512ifma,avx512dq")]
    unsafe fn mult8(
        scalars: &[[u8; 32]; 8],
        table: &SimdTable,
    ) -> [[u8; 32]; 8] {
        let mut words = [[0u64; 8]; 4];
        for lane in 0..8 {
            for w in 0..4 {
                words[w][lane] = u64::from_le_bytes(
                    scalars[lane][w * 8..w * 8 + 8]
                        .try_into()
                        .unwrap(),
                );
            }
        }
        let (x, y, z) = scalarmult8(
            &[
                _mm512_loadu_si512(words[0].as_ptr() as *const __m512i),
                _mm512_loadu_si512(words[1].as_ptr() as *const __m512i),
                _mm512_loadu_si512(words[2].as_ptr() as *const __m512i),
                _mm512_loadu_si512(words[3].as_ptr() as *const __m512i),
            ],
            table,
        );
        compress8(&x, &y, &z)
    }

    fn clamp(s: &mut [u8; 32]) {
        s[0] &= 248;
        s[31] &= 63;
        s[31] |= 64;
    }

    /// Run `scalars` eight at a time, in rotating lane positions so that
    /// every lane gets every scalar, against the scalar comb.
    fn check_scalars(scalars: &[[u8; 32]], what: &str) {
        let table = crate::fast::simd_table();
        for rot in 0..8 {
            for (c, chunk) in scalars.chunks(8).enumerate() {
                let mut sc = [[0u8; 32]; 8];
                for lane in 0..8 {
                    sc[(lane + rot) % 8] = chunk[lane % chunk.len()];
                }
                let got = unsafe { mult8(&sc, table) };
                for lane in 0..8 {
                    let expect =
                        crate::fast::scalarmult_compress_scalar(
                            &sc[lane],
                        );
                    assert_eq!(
                        got[lane], expect,
                        "{what}: chunk {c} rot {rot} lane {lane}"
                    );
                }
            }
        }
    }

    #[test]
    fn scalarmult8_matches_scalar_comb() {
        if !available() {
            return;
        }
        // Smallest-first probe: small free bits, every lane.
        let mut small = Vec::new();
        for k in [0u8, 1, 2, 3, 5, 16, 17, 31] {
            let mut s = [0u8; 32];
            s[0] = k << 3;
            clamp(&mut s);
            small.push(s);
        }
        check_scalars(&small, "small");
        let mut st = 0x9999_7777_3333_1111u64;
        let mut random = Vec::new();
        for _ in 0..200 {
            let mut s = rng_bytes(&mut st);
            clamp(&mut s);
            random.push(s);
        }
        check_scalars(&random, "random");
    }

    /// Write `value` into window `w` of a clamped scalar's free bits.
    fn set_window(s: &mut [u8; 32], w: usize, value: u64) {
        for b in 0..SIMD_W {
            let bit = 3 + SIMD_W * w + b;
            if bit >= 254 {
                break;
            }
            let (byte, off) = (bit / 8, bit % 8);
            s[byte] &= !(1 << off);
            s[byte] |= (((value >> b) & 1) as u8) << off;
        }
    }

    // Every digit extreme in every window, against both neighbour backdrops,
    // because a neighbouring all-ones window is what feeds in a carry.
    #[test]
    fn scalarmult8_digit_extremes_every_window() {
        if !available() {
            return;
        }
        let w = SIMD_W as u32;
        let pos = SIMD_POS as u64;
        let max = (1u64 << w) - 1;
        let values = [0, 1, pos - 1, pos, pos + 1, max - 1, max];
        let mut all_zero = [0u8; 32];
        clamp(&mut all_zero);
        let mut all_one = [0xffu8; 32];
        clamp(&mut all_one);
        let mut scalars = vec![all_zero, all_one];
        for window in 0..SIMD_WINDOWS {
            for &v in &values {
                for backdrop in [all_zero, all_one] {
                    let mut s = backdrop;
                    set_window(&mut s, window, v);
                    scalars.push(s);
                }
            }
        }
        // Same digit everywhere: the longest possible carry ripples.
        for &v in &values {
            let mut s = all_zero;
            for window in 0..SIMD_WINDOWS {
                set_window(&mut s, window, v);
            }
            scalars.push(s);
        }
        check_scalars(&scalars, "extreme");
    }

    #[test]
    fn recode_reconstructs_the_scalar() {
        use num_bigint::BigInt;
        if !available() {
            return;
        }
        let table = crate::fast::simd_table();
        let base = table.0.as_ptr() as usize;
        let mut st = 0x1111_2222_3333_4444u64;
        let mut extreme = [0xffu8; 32];
        clamp(&mut extreme);
        for round in 0..2000 {
            let mut sc = [[0u8; 32]; 8];
            for s in sc.iter_mut() {
                *s = if round == 0 {
                    extreme
                } else {
                    rng_bytes(&mut st)
                };
                clamp(s);
            }
            let mut digits = [const { MaybeUninit::<WindowDigit>::uninit() };
                SIMD_WINDOWS];
            unsafe {
                let mut words = [[0u64; 8]; 4];
                for lane in 0..8 {
                    for w in 0..4 {
                        words[w][lane] = u64::from_le_bytes(
                            sc[lane][w * 8..w * 8 + 8]
                                .try_into()
                                .unwrap(),
                        );
                    }
                }
                let v = [
                    _mm512_loadu_si512(
                        words[0].as_ptr() as *const __m512i
                    ),
                    _mm512_loadu_si512(
                        words[1].as_ptr() as *const __m512i
                    ),
                    _mm512_loadu_si512(
                        words[2].as_ptr() as *const __m512i
                    ),
                    _mm512_loadu_si512(
                        words[3].as_ptr() as *const __m512i
                    ),
                ];
                recode(&v, table, &mut digits);
            }
            for lane in 0..8 {
                let mut sum = BigInt::from(1u8) << 254usize;
                for (i, d) in digits.iter().enumerate() {
                    let d = unsafe { d.assume_init_ref() };
                    let window = base + i * SIMD_WINDOW_BYTES;
                    let magnitude = (d.entries[lane] as usize - window)
                        >> SIMD_ENTRY_SHIFT;
                    assert!(magnitude <= SIMD_POS, "window {i}");
                    if i == SIMD_WINDOWS - 1 {
                        assert!(
                            magnitude < SIMD_TOP_ENTRIES,
                            "top window"
                        );
                    }
                    let negative = d.negative & (1 << lane) != 0;
                    assert!(
                        !(negative && i == SIMD_WINDOWS - 1),
                        "the top window has no negative entries"
                    );
                    let digit = if negative {
                        -BigInt::from(magnitude)
                    } else {
                        BigInt::from(magnitude)
                    };
                    sum += digit << (3 + SIMD_W * i);
                }
                assert_eq!(
                    sum,
                    BigInt::from_bytes_le(
                        num_bigint::Sign::Plus,
                        &sc[lane]
                    ),
                    "round {round} lane {lane}"
                );
            }
        }
    }

    fn scalar_filter(
        bytes: &[u8; 32],
        intervals: &[(u64, u64)],
    ) -> bool {
        let first8 = u64::from_be_bytes(bytes[..8].try_into().unwrap());
        intervals
            .iter()
            .any(|&(lo, span)| first8.wrapping_sub(lo) <= span)
    }

    #[target_feature(enable = "avx512f,avx512ifma,avx512dq,avx512bw")]
    unsafe fn filter8(ys: &[[u8; 32]; 8], iv: &[(u64, u64)]) -> u8 {
        // Multiplying by one yields limbs shaped like any `mul` output.
        let one = Fe8::splat_limbs(&[1, 0, 0, 0, 0]);
        filter_mask(&Fe8::from_bytes8(ys).mul(&one), iv)
    }

    fn check_filter8(values: &[[u8; 32]], iv: &[(u64, u64)]) {
        for chunk in values.chunks(8) {
            let mut ys = [[0u8; 32]; 8];
            for lane in 0..8 {
                ys[lane] = chunk[lane % chunk.len()];
            }
            let got = unsafe { filter8(&ys, iv) };
            for lane in 0..8 {
                // Compare with the canonical encoding of the same element.
                let canon = Fe::from_bytes(&ys[lane]).to_bytes();
                assert_eq!(
                    got & (1 << lane) != 0,
                    scalar_filter(&canon, iv),
                    "{:02x?}",
                    ys[lane]
                );
            }
        }
    }

    #[test]
    fn filter_mask_matches_scalar_at_boundaries() {
        if !available() {
            return;
        }
        let target = crate::fast::MatchTargets::new(
            &[("Cesp99", ""), ("ab", "c"), ("zzz", "")],
            false,
        );
        let iv = target.prefix_filter(u64::MAX).unwrap();
        assert!(iv.len() >= 2);
        let mut st = 0x0123_4567_89ab_cdefu64;
        let mut values = Vec::new();
        for &(lo, span) in &iv {
            let hi = lo + span;
            for first8 in [
                lo.wrapping_sub(1),
                lo,
                lo + 1,
                hi - 1,
                hi,
                hi.wrapping_add(1),
            ] {
                for filler in
                    [[0u8; 32], [0xffu8; 32], rng_bytes(&mut st)]
                {
                    let mut v = filler;
                    v[..8].copy_from_slice(&first8.to_be_bytes());
                    v[31] &= 0x7f;
                    values.push(v);
                }
            }
        }
        for _ in 0..2000 {
            let mut v = rng_bytes(&mut st);
            v[31] &= 0x7f;
            values.push(v);
        }
        check_filter8(&values, &iv);
    }

    // Field elements at or just above p have a canonical encoding whose first
    // bytes differ from their limbs', which is what the q in the filter is for.
    #[test]
    fn filter_mask_reduces_values_at_and_above_p() {
        if !available() {
            return;
        }
        let mut values = Vec::new();
        for k in 0u8..19 {
            let mut v = [0xffu8; 32];
            v[0] = 0xed + k;
            v[31] = 0x7f;
            values.push(v);
        }
        let mut below = [0xffu8; 32];
        below[0] = 0xec;
        below[31] = 0x7f;
        values.push(below);
        // Canonical first bytes of those are 0..=18, so these intervals are
        // only hit if the reduction happened.
        check_filter8(&values, &[(0, 18u64 << 56)]);
        check_filter8(&values, &[(0, 0)]);
        check_filter8(&values, &[(u64::MAX - 20, 20)]);
    }

    #[target_feature(enable = "avx512f,avx512ifma,avx512dq,avx512bw")]
    unsafe fn run_batch(
        batch: &mut KeyBatch,
        intervals: Option<&[(u64, u64)]>,
    ) {
        batch.run(crate::fast::simd_table(), intervals);
    }

    fn seeded_batch(st: &mut u64) -> Box<KeyBatch> {
        let mut batch = KeyBatch::new();
        for g in 0..SIMD_GROUPS {
            for lane in 0..8 {
                batch.set_next_seed(g, lane, &rng_bytes(st));
            }
        }
        batch
    }

    #[test]
    fn keygen8_matches_dalek() {
        use ed25519_dalek::SigningKey;
        use sha2::{Digest, Sha512};
        if !available() {
            eprintln!("skip: no IFMA");
            return;
        }
        let mut st = 0x5555_1111_aaaa_2222u64;
        let mut batch = seeded_batch(&mut st);
        let mut expect_seeds = vec![[0u8; 32]; SIMD_BATCH];
        for round in 0..20 {
            unsafe { run_batch(&mut batch, None) };
            for g in 0..SIMD_GROUPS {
                assert_eq!(batch.hits[g], 0xff);
                for lane in 0..8 {
                    let seed = batch.used_seed(g, lane);
                    if round > 0 {
                        // The chain: last round's digest upper half.
                        assert_eq!(seed, expect_seeds[g * 8 + lane]);
                    }
                    let theirs = SigningKey::from_bytes(&seed)
                        .verifying_key()
                        .to_bytes();
                    assert_eq!(
                        batch.pubkeys[g][lane], theirs,
                        "round {round} group {g} lane {lane}"
                    );
                    let h: [u8; 64] = Sha512::digest(seed).into();
                    expect_seeds[g * 8 + lane]
                        .copy_from_slice(&h[32..64]);
                }
            }
        }
    }

    // The filtered path against ground truth: every key's real address is
    // computed with dalek and base58, so a filter that drops a real match, or
    // a wrong `pubkeys` slot for one that passes, cannot go unnoticed.
    fn check_grind_finds_all(
        patterns: &[(&str, &str)],
        rounds: usize,
        min_hits: usize,
    ) {
        use ed25519_dalek::SigningKey;
        let target = crate::fast::MatchTargets::new(patterns, false);
        let intervals = target.prefix_filter(u64::MAX);
        assert!(
            intervals.is_some(),
            "filter expected for {patterns:?}"
        );
        let mut st = 0xa5a5_5a5a_1234_4321u64;
        let mut batch = seeded_batch(&mut st);
        let mut hits = 0;
        for round in 0..rounds {
            unsafe { run_batch(&mut batch, intervals.as_deref()) };
            for g in 0..SIMD_GROUPS {
                for lane in 0..8 {
                    let seed = batch.used_seed(g, lane);
                    let real = SigningKey::from_bytes(&seed)
                        .verifying_key()
                        .to_bytes();
                    let addr = fd_bs58::encode_32(real);
                    let matches = patterns.iter().any(|(p, s)| {
                        addr.starts_with(p) && addr.ends_with(s)
                    });
                    let passed = batch.hits[g] & (1 << lane) != 0;
                    if matches {
                        hits += 1;
                        assert!(
                            passed,
                            "filter dropped {addr}, round {round}"
                        );
                    }
                    if passed {
                        assert_eq!(batch.pubkeys[g][lane], real);
                        assert_eq!(
                            target.match_mask_active(&real, u64::MAX)
                                != 0,
                            matches,
                            "exact check disagrees on {addr}"
                        );
                    }
                }
            }
        }
        assert!(hits >= min_hits, "only {hits} hits for {patterns:?}");
    }

    #[test]
    fn grind_filter_finds_every_prefix_hit() {
        if !available() {
            return;
        }
        check_grind_finds_all(&[("C", "")], 40, 100);
    }

    #[test]
    fn grind_filter_finds_every_multi_pattern_hit() {
        if !available() {
            return;
        }
        check_grind_finds_all(
            &[("Ce", ""), ("zz", ""), ("9", "x"), ("Ces", "p")],
            60,
            10,
        );
    }

    // A found key's next chained seed is its nonce prefix, so after a hit
    // that lane must continue from fresh randomness instead.
    #[test]
    fn reseed_replaces_the_chained_seed() {
        use sha2::{Digest, Sha512};
        if !available() {
            return;
        }
        let mut st = 0x7777_8888_9999_aaaau64;
        let mut batch = seeded_batch(&mut st);
        unsafe { run_batch(&mut batch, None) };
        let hit_seed = batch.used_seed(3, 5);
        let other_seed = batch.used_seed(3, 6);
        let fresh = rng_bytes(&mut st);
        batch.set_next_seed(3, 5, &fresh);
        unsafe { run_batch(&mut batch, None) };

        let h: [u8; 64] = Sha512::digest(hit_seed).into();
        assert_eq!(batch.used_seed(3, 5), fresh);
        assert_ne!(batch.used_seed(3, 5).as_slice(), &h[32..64]);
        // Neighbouring lanes are untouched and keep chaining.
        let h: [u8; 64] = Sha512::digest(other_seed).into();
        assert_eq!(batch.used_seed(3, 6).as_slice(), &h[32..64]);
    }
}
