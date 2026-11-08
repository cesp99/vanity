mod check_match;
mod field;
mod group;
#[cfg(target_arch = "aarch64")]
mod neon;
pub mod sha256_simd;
pub mod sha512_simd;
pub mod simd;

pub use check_match::{
    ci_variants, MatchTargets, MAX_CI_VARIANTS, MAX_PATTERNS,
    MAX_PATTERN_LEN,
};
#[cfg(any(test, not(target_arch = "aarch64")))]
use field::{batch_invert, Fe};
#[cfg(any(test, not(target_arch = "aarch64")))]
use group::{edwards_d2, Niels, Point};
pub use sha256_simd::{grind_seeds, SeedJob};

use crate::{credit_kinds, save_keypair, ABORTED, TOTAL_ATTEMPTS};
#[cfg(any(test, not(target_arch = "aarch64")))]
use sha2::{Digest, Sha512};
use std::sync::atomic::Ordering;
#[cfg(any(test, not(target_arch = "aarch64")))]
use std::sync::OnceLock;

pub const BATCH: usize = 512;

// ─── scalar ed25519 keygen ───────────────────────────────────────────────────
//
// Apple Silicon grinds with the NEON batch path. This reference
// implementation stays for every other target, and for tests.

#[cfg(any(test, not(target_arch = "aarch64")))]
const BASE: Point = Point {
    x: Fe([
        1738742601995546,
        1146398526822698,
        2070867633025821,
        562264141797630,
        587772402128613,
    ]),
    y: Fe([
        1801439850948184,
        1351079888211148,
        450359962737049,
        900719925474099,
        1801439850948198,
    ]),
    z: Fe([1, 0, 0, 0, 0]),
    t: Fe([
        1841354044333475,
        16398895984059,
        755974180946558,
        900171276175154,
        1821297809914039,
    ]),
};

// ─── fixed-base comb table ───────────────────────────────────────────────────

#[cfg(any(test, not(target_arch = "aarch64")))]
struct CombTable(Vec<[Niels; 8]>);

#[cfg(any(test, not(target_arch = "aarch64")))]
fn comb_table() -> &'static CombTable {
    static TABLE: OnceLock<CombTable> = OnceLock::new();
    TABLE.get_or_init(|| {
        let d2 = edwards_d2();
        let mut tables: Vec<[Niels; 8]> = Vec::with_capacity(64);
        let mut cur = BASE; // 16^i * B
        for _ in 0..64 {
            let cur_niels = cur.as_niels(&d2);
            let mut multiple = cur; // (k+1) * 16^i * B
            let mut row = [cur.as_niels(&d2); 8];
            for k in 1..8 {
                multiple = multiple.add_niels(&cur_niels);
                row[k] = multiple.as_niels(&d2);
            }
            tables.push(row);
            // cur *= 16
            cur = cur.double().double().double().double();
        }
        CombTable(tables)
    })
}

#[cfg(any(test, not(target_arch = "aarch64")))]
#[inline]
fn to_radix16(bytes: &[u8; 32]) -> [i8; 64] {
    let mut d = [0i8; 64];
    for i in 0..32 {
        d[2 * i] = (bytes[i] & 15) as i8;
        d[2 * i + 1] = ((bytes[i] >> 4) & 15) as i8;
    }
    let mut carry = 0i8;
    for i in 0..63 {
        d[i] += carry;
        carry = (d[i] + 8) >> 4;
        d[i] -= carry << 4;
    }
    d[63] += carry;
    d
}

#[cfg(any(test, not(target_arch = "aarch64")))]
#[inline]
fn scalarmult_base(scalar: &[u8; 32], table: &CombTable) -> Point {
    let digits = to_radix16(scalar);
    let mut acc = Point::IDENTITY;
    for i in 0..64 {
        let di = digits[i];
        if di > 0 {
            acc = acc.add_niels(&table.0[i][(di - 1) as usize]);
        } else if di < 0 {
            acc = acc.sub_niels(&table.0[i][(-di - 1) as usize]);
        }
    }
    acc
}

#[cfg(any(test, not(target_arch = "aarch64")))]
#[inline(always)]
fn clamp(h: &[u8; 64]) -> [u8; 32] {
    let mut s = [0u8; 32];
    s.copy_from_slice(&h[..32]);
    s[0] &= 248;
    s[31] &= 63;
    s[31] |= 64;
    s
}

#[cfg(any(test, not(target_arch = "aarch64")))]
fn batch_compress(points: &[Point], out: &mut [[u8; 32]]) {
    let n = points.len();
    let mut zs: Vec<Fe> = points.iter().map(|p| p.z).collect();
    batch_invert(&mut zs);
    for i in 0..n {
        let zinv = zs[i];
        let x = points[i].x.mul(&zinv);
        let y = points[i].y.mul(&zinv);
        let mut bytes = y.to_bytes();
        bytes[31] |= (x.is_odd() as u8) << 7;
        out[i] = bytes;
    }
}

#[cfg(any(test, not(target_arch = "aarch64")))]
fn keygen_batch(
    seeds: &mut [[u8; 32]; BATCH],
    used: &mut [[u8; 32]; BATCH],
    pubkeys: &mut [[u8; 32]; BATCH],
) {
    let table = comb_table();
    let mut points = [Point::IDENTITY; BATCH];
    for j in 0..BATCH {
        used[j] = seeds[j];
        let h: [u8; 64] = Sha512::digest(seeds[j]).into();
        let scalar = clamp(&h);
        points[j] = scalarmult_base(&scalar, table);
        seeds[j].copy_from_slice(&h[32..64]);
    }
    batch_compress(&points, pubkeys);
}

#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) fn scalarmult_compress_scalar(
    scalar: &[u8; 32],
) -> [u8; 32] {
    let p = scalarmult_base(scalar, comb_table());
    let mut buf = [[0u8; 32]; 1];
    batch_compress(&[p], &mut buf);
    buf[0]
}

// ─── SIMD (AVX-512 IFMA) batch keygen ────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
pub(crate) fn simd_table() -> &'static simd::SimdTable {
    static T: OnceLock<simd::SimdTable> = OnceLock::new();
    T.get_or_init(simd::build_simd_table)
}

/// Build the lookup tables the CPU workers use, so that their cost lands
/// before the grind clock starts rather than on the reported rate.
pub fn init_cpu_tables() {
    #[cfg(target_arch = "x86_64")]
    if simd::available() {
        simd_table();
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    comb_table();
}

fn reseed_lane(seed: &mut [u8; 32]) {
    *seed = rand::random();
}

/// Last gate before a key can be saved: re-derive the public key from the seed
/// with `ed25519_dalek`, and check it against the batch code's result and,
/// through the base58 walk rather than byte ranges, against the patterns.
fn verify_hit(
    target: &MatchTargets,
    seed: &[u8; 32],
    pubkey: &[u8; 32],
    mask: u64,
) -> Result<(), String> {
    let derived = ed25519_dalek::SigningKey::from_bytes(seed)
        .verifying_key()
        .to_bytes();
    if derived != *pubkey {
        return Err(format!(
            "batch keygen produced {} but the seed derives {}",
            fd_bs58::encode_32(*pubkey),
            fd_bs58::encode_32(derived),
        ));
    }
    let encoded = target.match_mask_encoded(&derived, mask);
    if encoded != mask {
        return Err(format!(
            "{} claimed pattern mask {mask:#x} but encodes to {encoded:#x}",
            fd_bs58::encode_32(derived),
        ));
    }
    Ok(())
}

/// Handle a lane whose key matched `mask`. Returns true when the lane must be
/// reseeded: the next chained seed is this key's nonce prefix, so it must not
/// be ground on once the key may be published.
fn take_hit(
    target: &MatchTargets,
    seed: &[u8; 32],
    pubkey: &[u8; 32],
    mask: u64,
    max_count: u32,
) -> bool {
    if let Err(err) = verify_hit(target, seed, pubkey, mask) {
        crate::ui_with_match(|| {
            eprintln!("ERROR: refusing to save a bad keypair: {err}");
            eprintln!("ERROR: this is a bug in vanity; grinding on");
        });
        return true;
    }
    if !credit_kinds(mask, max_count) {
        return false;
    }
    let s = fd_bs58::encode_32(pubkey);
    crate::ui_with_match(|| {
        eprintln!("match: {s}");
        eprintln!("pubkey: {s}");
        save_keypair(seed, pubkey, &s);
    });
    true
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512ifma,avx512dq,avx512bw")]
unsafe fn grind_thread_simd(
    target: &MatchTargets,
    min_count: u32,
    max_count: u32,
) {
    let table = simd_table();
    let mut batch = simd::KeyBatch::new();
    for g in 0..simd::SIMD_GROUPS {
        for lane in 0..8 {
            batch.set_next_seed(g, lane, &rand::random());
        }
    }
    let mut active = 0u64;
    let mut intervals = None;
    let mut local: u64 = 0;

    loop {
        if grind_done(min_count) {
            break;
        }
        // Kinds only ever fill up, so a mask or filter that went stale during
        // the batch is a superset of what is wanted. The credit in `take_hit`
        // has the final say, and no hit can be lost to staleness.
        let unfilled = crate::unfilled_mask(max_count);
        if unfilled != active {
            active = unfilled;
            intervals = target.prefix_filter(active);
        }
        batch.run(table, intervals.as_deref());
        local += simd::SIMD_BATCH as u64;
        if local >= 4096 {
            TOTAL_ATTEMPTS.fetch_add(local, Ordering::Relaxed);
            local = 0;
        }
        for g in 0..simd::SIMD_GROUPS {
            let mut lanes = batch.hits[g];
            while lanes != 0 {
                let lane = lanes.trailing_zeros() as usize;
                lanes &= lanes - 1;
                let pubkey = batch.pubkeys[g][lane];
                let mask = target.match_mask_active(&pubkey, active);
                if mask == 0 {
                    continue;
                }
                let seed = batch.used_seed(g, lane);
                if take_hit(target, &seed, &pubkey, mask, max_count) {
                    batch.set_next_seed(g, lane, &rand::random());
                }
            }
        }
    }
    TOTAL_ATTEMPTS.fetch_add(local, Ordering::Relaxed);
}

#[cfg(target_arch = "aarch64")]
fn grind_thread_neon(
    target: &MatchTargets,
    min_count: u32,
    max_count: u32,
) {
    let mut seeds: [[u8; 32]; BATCH] = [[0u8; 32]; BATCH];
    for s in seeds.iter_mut() {
        *s = rand::random();
    }
    let mut used = [[0u8; 32]; BATCH];
    let mut pubkeys = [[0u8; 32]; BATCH];
    let mut local: u64 = 0;

    loop {
        if grind_done(min_count) {
            break;
        }
        unsafe {
            neon::keygen_batch(&mut seeds, &mut used, &mut pubkeys);
        }
        local += BATCH as u64;
        if local >= 4096 {
            TOTAL_ATTEMPTS.fetch_add(local, Ordering::Relaxed);
            local = 0;
        }
        for j in 0..BATCH {
            let mask = target.match_mask_active(
                &pubkeys[j],
                crate::unfilled_mask(max_count),
            );
            if mask != 0
                && take_hit(
                    target,
                    &used[j],
                    &pubkeys[j],
                    mask,
                    max_count,
                )
            {
                reseed_lane(&mut seeds[j]);
            }
        }
    }
    TOTAL_ATTEMPTS.fetch_add(local, Ordering::Relaxed);
}

#[cfg(not(target_arch = "aarch64"))]
fn grind_thread_scalar(
    target: &MatchTargets,
    min_count: u32,
    max_count: u32,
) {
    let mut seeds: [[u8; 32]; BATCH] = [[0u8; 32]; BATCH];
    for s in seeds.iter_mut() {
        *s = rand::random();
    }
    let mut used = [[0u8; 32]; BATCH];
    let mut pubkeys = [[0u8; 32]; BATCH];
    let mut local: u64 = 0;

    loop {
        if grind_done(min_count) {
            break;
        }
        keygen_batch(&mut seeds, &mut used, &mut pubkeys);
        local += BATCH as u64;
        if local >= 4096 {
            TOTAL_ATTEMPTS.fetch_add(local, Ordering::Relaxed);
            local = 0;
        }
        for j in 0..BATCH {
            let mask = target.match_mask_active(
                &pubkeys[j],
                crate::unfilled_mask(max_count),
            );
            if mask != 0
                && take_hit(
                    target,
                    &used[j],
                    &pubkeys[j],
                    mask,
                    max_count,
                )
            {
                reseed_lane(&mut seeds[j]);
            }
        }
    }
    TOTAL_ATTEMPTS.fetch_add(local, Ordering::Relaxed);
}

#[inline(always)]
fn grind_done(count: u32) -> bool {
    crate::quotas_filled(count) || ABORTED.load(Ordering::Relaxed)
}

#[cfg(test)]
pub fn pubkey_of_seed(seed: &[u8; 32]) -> [u8; 32] {
    let h: [u8; 64] = Sha512::digest(seed).into();
    let scalar = clamp(&h);
    let p = scalarmult_base(&scalar, comb_table());
    let mut buf = [[0u8; 32]; 1];
    batch_compress(&[p], &mut buf);
    buf[0]
}

// ─── multi-threaded grind ────────────────────────────────────────────────────

use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub fn reset_grind(n_kinds: usize) {
    crate::start_grind(n_kinds);
}

pub fn request_abort() {
    ABORTED.store(true, Ordering::SeqCst);
}

#[cfg(feature = "gpu")]
pub fn is_done(count: u32) -> bool {
    grind_done(count)
}

pub fn total_attempts() -> u64 {
    TOTAL_ATTEMPTS.load(Ordering::Relaxed)
}

pub fn backend_name() -> &'static str {
    if cfg!(target_arch = "aarch64") {
        "neon (4-lane)"
    } else if simd::available() {
        "avx512-ifma (8-lane)"
    } else {
        "scalar"
    }
}

/// Run batched CPU keypair workers until `min_count` hits of each
/// pattern or abort. Extra hits are kept up to `max_count`. Call
/// [`reset_grind`] first if coordinating with a GPU thread that shares
/// these counters via [`is_done`].
pub fn run_cpu_workers(
    target: &MatchTargets,
    num_cpus: u32,
    min_count: u32,
    max_count: u32,
) {
    (0..num_cpus)
        .into_par_iter()
        .for_each(|_| {
            #[cfg(target_arch = "x86_64")]
            {
                if simd::available() {
                    unsafe {
                        grind_thread_simd(target, min_count, max_count)
                    };
                } else {
                    grind_thread_scalar(target, min_count, max_count);
                }
            }
            #[cfg(target_arch = "aarch64")]
            grind_thread_neon(target, min_count, max_count);
            #[cfg(not(any(
                target_arch = "x86_64",
                target_arch = "aarch64"
            )))]
            grind_thread_scalar(target, min_count, max_count);
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    #[test]
    fn basepoint_compresses_to_canonical() {
        let mut out = [[0u8; 32]; 1];
        batch_compress(&[BASE], &mut out);
        // Canonical ed25519 base point compressed encoding.
        let mut expect = [0x66u8; 32];
        expect[0] = 0x58;
        assert_eq!(out[0], expect);
    }

    #[test]
    fn matches_dalek_for_random_seeds() {
        for i in 0u32..2000 {
            // Deterministic pseudo-random seeds.
            let h: [u8; 64] = Sha512::digest(i.to_le_bytes()).into();
            let seed: [u8; 32] = h[..32].try_into().unwrap();
            let ours = pubkey_of_seed(&seed);
            let theirs = SigningKey::from_bytes(&seed)
                .verifying_key()
                .to_bytes();
            assert_eq!(ours, theirs, "seed index {i}");
        }
    }

    #[test]
    fn verify_hit_requires_derived_key_and_pattern() {
        let seed = [7u8; 32];
        let pubkey = SigningKey::from_bytes(&seed)
            .verifying_key()
            .to_bytes();
        let addr = fd_bs58::encode_32(pubkey);
        let target = MatchTargets::new(
            &[(&addr[..3], ""), ("zzzzzzzzzzzz", "")],
            false,
        );
        assert!(verify_hit(&target, &seed, &pubkey, 0b01).is_ok());

        // A batch result that is not what the seed derives.
        let mut wrong = pubkey;
        wrong[31] ^= 0x80;
        assert!(verify_hit(&target, &seed, &wrong, 0b01).is_err());
        wrong = pubkey;
        wrong[0] ^= 1;
        assert!(verify_hit(&target, &seed, &wrong, 0b01).is_err());

        // The right key, claiming a pattern it does not have.
        assert!(verify_hit(&target, &seed, &pubkey, 0b10).is_err());
        assert!(verify_hit(&target, &seed, &pubkey, 0b11).is_err());
    }

    // A rejected key must not use up the quota.
    #[test]
    fn take_hit_rejects_without_crediting() {
        let seed = [9u8; 32];
        let pubkey = SigningKey::from_bytes(&seed)
            .verifying_key()
            .to_bytes();
        let addr = fd_bs58::encode_32(pubkey);
        let target = MatchTargets::new(&[(&addr[..3], "")], false);
        let mut wrong = pubkey;
        wrong[5] ^= 1;
        crate::reset_quotas(1);
        assert!(take_hit(&target, &seed, &wrong, 1, 1));
        assert_eq!(crate::unfilled_mask(1), 1);
        for addr in [addr, fd_bs58::encode_32(wrong)] {
            assert!(
                !std::path::Path::new(&format!("{addr}.json")).exists()
            );
        }
    }

    #[test]
    fn batch_matches_singles() {
        let mut seeds = [[0u8; 32]; BATCH];
        for j in 0..BATCH {
            let h: [u8; 64] =
                Sha512::digest((j as u64).to_le_bytes()).into();
            seeds[j] = h[..32].try_into().unwrap();
        }
        let mut used = [[0u8; 32]; BATCH];
        let mut pubs = [[0u8; 32]; BATCH];
        let mut seeds_copy = seeds;
        keygen_batch(&mut seeds_copy, &mut used, &mut pubs);
        for j in 0..BATCH {
            assert_eq!(used[j], seeds[j]);
            let h: [u8; 64] = Sha512::digest(seeds[j]).into();
            assert_eq!(&seeds_copy[j], &h[32..64], "lane {j} chain");
            let theirs = SigningKey::from_bytes(&seeds[j])
                .verifying_key()
                .to_bytes();
            assert_eq!(pubs[j], theirs, "lane {j}");
            reseed_lane(&mut seeds_copy[j]);
            assert_ne!(&seeds_copy[j], &h[32..64], "lane {j} reseed");
        }
    }
}
