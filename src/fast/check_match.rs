use fd_bs58::constants::{
    BINARY_SZ_32, ENC_TABLE_32, INTERMEDIATE_SZ_32, R1_DIV, RAW58_SZ_32,
};
use num_bigint::BigUint;
use num_traits::One;

pub const MAX_PATTERN_LEN: usize = 44;
pub const MAX_PATTERNS: usize = 64;

const ALPHABET: &[u8; 58] =
    b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
const ALPHABET_CI: &[u8; 58] =
    b"123456789abcdefghjkLmnpqrstuvwxyzabcdefghijkmnopqrstuvwxyz";

const fn identity_lut() -> [u8; 58] {
    let mut lut = [0u8; 58];
    let mut i = 0;
    while i < 58 {
        lut[i] = i as u8;
        i += 1;
    }
    lut
}

const fn build_match_lut(alphabet: &[u8; 58]) -> [u8; 58] {
    let mut lut = [0u8; 58];
    let mut i = 0;
    while i < 58 {
        lut[i] = i as u8;
        let mut j = 0;
        while j < i {
            if alphabet[j] == alphabet[i] {
                lut[i] = j as u8;
                break;
            }
            j += 1;
        }
        i += 1;
    }
    lut
}

static MATCH_LUT_CS: [u8; 58] = identity_lut();
static MATCH_LUT_CI: [u8; 58] = build_match_lut(ALPHABET_CI);

#[inline]
pub fn match_lut(case_insensitive: bool) -> &'static [u8; 58] {
    if case_insensitive {
        &MATCH_LUT_CI
    } else {
        &MATCH_LUT_CS
    }
}

struct PatternIdx {
    prefix_ranges: Option<Vec<([u8; 32], [u8; 32])>>,
    suffix_residue: Option<SuffixResidue>,
    prefix_idx: [u8; MAX_PATTERN_LEN],
    prefix_len: u8,
    suffix_idx: [u8; MAX_PATTERN_LEN],
    suffix_len: u8,
}

/// One or more prefix/suffix pairs. A pubkey matches if it matches any pair.
///
/// GPU blob layout (little-endian), shared with CUDA / OpenCL:
/// ```text
/// u32 n
/// u8  max_prefix_len
/// u8  max_suffix_len
/// n times: u8 plen, plen bytes, u8 slen, slen bytes
/// ```
/// Bytes are canonical raw_base58 indices, not ASCII.
pub struct MatchTargets {
    patterns: Vec<PatternIdx>,
    match_lut: &'static [u8; 58],
}

impl MatchTargets {
    pub fn new(
        patterns: &[(&str, &str)],
        case_insensitive: bool,
    ) -> Self {
        assert!(
            patterns.len() <= MAX_PATTERNS,
            "at most {MAX_PATTERNS} patterns"
        );
        let alphabet = if case_insensitive {
            ALPHABET_CI
        } else {
            ALPHABET
        };
        let lut = match_lut(case_insensitive);

        let mut out = Vec::with_capacity(patterns.len());
        for &(prefix, suffix) in patterns {
            debug_assert!(prefix.len() <= MAX_PATTERN_LEN);
            debug_assert!(suffix.len() <= MAX_PATTERN_LEN);
            let mut prefix_idx = [0u8; MAX_PATTERN_LEN];
            for (i, &b) in prefix.as_bytes().iter().enumerate() {
                prefix_idx[i] = char_to_canonical(b, alphabet, lut);
            }
            let mut suffix_idx = [0u8; MAX_PATTERN_LEN];
            for (i, &b) in suffix.as_bytes().iter().enumerate() {
                suffix_idx[i] = char_to_canonical(b, alphabet, lut);
            }
            out.push(PatternIdx {
                prefix_ranges: prefix_ranges(prefix, case_insensitive),
                suffix_residue: SuffixResidue::new(
                    suffix,
                    case_insensitive,
                ),
                prefix_idx,
                prefix_len: prefix.len() as u8,
                suffix_idx,
                suffix_len: suffix.len() as u8,
            });
        }

        Self {
            patterns: out,
            match_lut: lut,
        }
    }

    /// Bit `i` is set when pattern `i` matches. Empty set → no match.
    #[cfg(test)]
    pub fn match_mask(&self, bytes: &[u8; 32]) -> u64 {
        self.match_mask_active(bytes, u64::MAX)
    }

    /// Bit `i` is set when pattern `i` matches and that kind is in `active`.
    #[inline]
    pub fn match_mask_active(
        &self,
        bytes: &[u8; 32],
        active: u64,
    ) -> u64 {
        let mut mask = 0u64;
        let mut encode_active = 0u64;
        for (pi, p) in self.patterns.iter().enumerate() {
            if active & (1u64 << pi) == 0 {
                continue;
            }
            if let Some(ranges) = &p.prefix_ranges {
                if !ranges
                    .iter()
                    .any(|(lo, hi)| bytes >= lo && bytes <= hi)
                {
                    continue;
                }
                if p.suffix_len == 0 {
                    mask |= 1u64 << pi;
                    continue;
                }
            }
            if let Some(r) = &p.suffix_residue {
                if !r.holds(bytes) {
                    continue;
                }
            }
            encode_active |= 1u64 << pi;
        }
        if encode_active != 0 {
            mask |= check_match_32(
                bytes,
                &self.patterns,
                self.match_lut,
                encode_active,
            );
        }
        mask
    }

    /// Like [`Self::match_mask_active`], but always through the base58 walk,
    /// so it can catch a disagreement with the byte ranges.
    pub fn match_mask_encoded(
        &self,
        bytes: &[u8; 32],
        active: u64,
    ) -> u64 {
        check_match_32(bytes, &self.patterns, self.match_lut, active)
    }

    /// Sorted, disjoint `(lo, span)` intervals over a key's first eight bytes
    /// read big-endian. Any key matching an `active` pattern lands in one,
    /// so a key outside them all needs no further checks. None when an active
    /// pattern can't be expressed this way (suffix-only, a leading '1', or
    /// case-insensitive with more than [`MAX_CI_VARIANTS`] case variants),
    /// and every key has to go to the exact check.
    #[cfg(any(test, target_arch = "x86_64"))]
    pub fn prefix_filter(
        &self,
        active: u64,
    ) -> Option<Vec<(u64, u64)>> {
        let mut bounds = Vec::new();
        for (pi, p) in self.patterns.iter().enumerate() {
            if active & (1u64 << pi) == 0 {
                continue;
            }
            for (lo, hi) in p.prefix_ranges.as_ref()? {
                bounds.push((first8(lo), first8(hi)));
            }
        }
        bounds.sort_unstable();
        let mut merged: Vec<(u64, u64)> = Vec::new();
        for (lo, hi) in bounds {
            match merged.last_mut() {
                Some(last) if lo <= last.1.saturating_add(1) => {
                    last.1 = last.1.max(hi);
                }
                _ => merged.push((lo, hi)),
            }
        }
        Some(
            merged
                .iter()
                .map(|&(lo, hi)| (lo, hi - lo))
                .collect(),
        )
    }

    /// Canonical-index blob consumed by `gpu_grind_init` / `gpu_keypair_init`.
    #[cfg(any(test, feature = "gpu"))]
    pub fn gpu_blob(&self) -> Vec<u8> {
        let max_prefix_len = self
            .patterns
            .iter()
            .map(|p| p.prefix_len)
            .max()
            .unwrap_or(0);
        let max_suffix_len = self
            .patterns
            .iter()
            .map(|p| p.suffix_len)
            .max()
            .unwrap_or(0);
        let mut v = Vec::with_capacity(
            6 + self
                .patterns
                .iter()
                .map(|p| {
                    2 + p.prefix_len as usize + p.suffix_len as usize
                })
                .sum::<usize>(),
        );
        v.extend_from_slice(
            &(self.patterns.len() as u32).to_le_bytes(),
        );
        v.push(max_prefix_len);
        v.push(max_suffix_len);
        for p in &self.patterns {
            v.push(p.prefix_len);
            v.extend_from_slice(&p.prefix_idx[..p.prefix_len as usize]);
            v.push(p.suffix_len);
            v.extend_from_slice(&p.suffix_idx[..p.suffix_len as usize]);
        }
        v
    }
}

#[cfg(any(test, target_arch = "x86_64"))]
#[inline]
fn first8(bytes: &[u8; 32]) -> u64 {
    u64::from_be_bytes(bytes[..8].try_into().unwrap())
}

#[inline]
fn char_to_canonical(
    c: u8,
    alphabet: &[u8; 58],
    match_lut: &[u8; 58],
) -> u8 {
    for (k, &ch) in alphabet.iter().enumerate() {
        if ch == c {
            return match_lut[k];
        }
    }
    255
}

/// Most case variants a case-insensitive prefix is expanded into for the
/// byte-range check. There are 2^letters of them, each adding an interval or
/// two to the prefix filter, so longer ones go to the base58 walk.
pub const MAX_CI_VARIANTS: usize = 256;

/// Every case-sensitive pattern that `pattern` (already folded, as
/// [`MatchTargets::new`] takes it) matches case-insensitively: each
/// character becomes every alphabet character that folds to it, so `L` and
/// the digits stay as they are. None when there would be more than `max`.
pub fn ci_variants(pattern: &str, max: usize) -> Option<Vec<String>> {
    let mut out = vec![String::new()];
    for b in pattern.bytes() {
        let choices: Vec<char> = ALPHABET
            .iter()
            .zip(ALPHABET_CI)
            .filter(|&(_, &folded)| folded == b)
            .map(|(&c, _)| c as char)
            .collect();
        if choices.is_empty() || out.len() * choices.len() > max {
            return None;
        }
        out = out
            .iter()
            .flat_map(|v| {
                choices
                    .iter()
                    .map(move |&c| format!("{v}{c}"))
            })
            .collect();
    }
    Some(out)
}

// Prefixes can be checked as byte ranges without base58 conversion; a
// case-insensitive one is the union of its case variants' ranges. Keep the
// encode matcher for a leading '1' and for too many case variants.
fn prefix_ranges(
    prefix: &str,
    ci: bool,
) -> Option<Vec<([u8; 32], [u8; 32])>> {
    if !ci {
        return exact_prefix_ranges(prefix);
    }
    let mut ranges = Vec::new();
    for variant in ci_variants(prefix, MAX_CI_VARIANTS)? {
        ranges.extend(exact_prefix_ranges(&variant)?);
    }
    ranges.sort_unstable();
    Some(ranges)
}

fn exact_prefix_ranges(
    prefix: &str,
) -> Option<Vec<([u8; 32], [u8; 32])>> {
    if prefix.is_empty() || prefix.starts_with('1') {
        return None;
    }
    let mut value = BigUint::from(0u8);
    for b in prefix.bytes() {
        value *= 58u8;
        value += ALPHABET.iter().position(|&c| c == b)?;
    }
    let minimum = BigUint::one() << 248usize;
    let maximum = (BigUint::one() << 256usize) - 1u8;
    let to_bytes = |n: &BigUint| {
        let v = n.to_bytes_be();
        let mut out = [0u8; 32];
        out[32 - v.len()..].copy_from_slice(&v);
        out
    };
    let mut ranges = Vec::new();
    let mut scale = BigUint::one();
    for _ in prefix.len()..=MAX_PATTERN_LEN {
        let lo = (&value * &scale).max(minimum.clone());
        let hi = ((&value + 1u8) * &scale - 1u8).min(maximum.clone());
        if lo <= hi {
            ranges.push((to_bytes(&lo), to_bytes(&hi)));
        }
        scale *= 58u8;
    }
    Some(ranges)
}

/// The last characters of a base58 string are the low base58 digits of the
/// number it encodes, and a leading '1' is a zero digit, so a key can only
/// end in `suffix` if `N mod 58^k` equals the value of its last `k`
/// characters. A necessary condition, checked before the base58 walk.
struct SuffixResidue {
    modulus: u64,
    residue: u64,
    /// `2^(32 * (7 - i)) mod modulus`, the weight of big-endian word `i`.
    scale: [u64; 8],
}

/// Characters the residue covers. 58^4 < 2^24, so each word times its
/// scale stays below 2^56 and the sum of eight fits a u64.
const SUFFIX_RESIDUE_LEN: usize = 4;

impl SuffixResidue {
    fn new(suffix: &str, ci: bool) -> Option<Self> {
        if ci || suffix.is_empty() {
            return None;
        }
        let tail = &suffix.as_bytes()[suffix
            .len()
            .saturating_sub(SUFFIX_RESIDUE_LEN)..];
        let (mut modulus, mut residue) = (1u64, 0u64);
        for b in tail {
            residue = residue * 58
                + ALPHABET.iter().position(|c| c == b)? as u64;
            modulus *= 58;
        }
        let mut scale = [0u64; 8];
        let mut w = 1 % modulus;
        for i in (0..8).rev() {
            scale[i] = w;
            w = (w << 32) % modulus;
        }
        Some(Self {
            modulus,
            residue,
            scale,
        })
    }

    #[inline]
    fn holds(&self, bytes: &[u8; 32]) -> bool {
        let mut sum = 0u64;
        for i in 0..8 {
            let w = u32::from_be_bytes(
                bytes[i * 4..i * 4 + 4]
                    .try_into()
                    .unwrap(),
            );
            sum += u64::from(w) * self.scale[i];
        }
        sum % self.modulus == self.residue
    }
}

#[inline]
fn emit_limb(
    intermediate: &[u64; INTERMEDIATE_SZ_32],
    raw_base58: &mut [u8; RAW58_SZ_32],
    limb: usize,
) {
    let v = intermediate[limb] as u32;
    raw_base58[5 * limb + 4] = (v % 58) as u8;
    raw_base58[5 * limb + 3] = ((v / 58) % 58) as u8;
    raw_base58[5 * limb + 2] = ((v / 3364) % 58) as u8;
    raw_base58[5 * limb + 1] = ((v / 195_112) % 58) as u8;
    raw_base58[5 * limb] = (v / 11_316_496) as u8;
}

#[inline]
fn ensure_limb(
    intermediate: &[u64; INTERMEDIATE_SZ_32],
    raw_base58: &mut [u8; RAW58_SZ_32],
    limbs_done: &mut usize,
    need: usize,
) {
    while *limbs_done <= need {
        emit_limb(intermediate, raw_base58, *limbs_done);
        *limbs_done += 1;
    }
}

#[inline]
fn check_match_32(
    bytes: &[u8; 32],
    patterns: &[PatternIdx],
    match_lut: &[u8; 58],
    active: u64,
) -> u64 {
    let mut in_leading_0s = 0usize;
    while in_leading_0s < 32 && bytes[in_leading_0s] == 0 {
        in_leading_0s += 1;
    }

    let mut binary = [0u32; BINARY_SZ_32];
    for i in 0..BINARY_SZ_32 {
        let o = i * 4;
        binary[i] = u32::from_be_bytes([
            bytes[o],
            bytes[o + 1],
            bytes[o + 2],
            bytes[o + 3],
        ]);
    }

    let mut intermediate = [0u64; INTERMEDIATE_SZ_32];
    for i in 0..BINARY_SZ_32 {
        for j in 0..INTERMEDIATE_SZ_32 - 1 {
            intermediate[j + 1] +=
                u64::from(binary[i]) * ENC_TABLE_32[i][j];
        }
    }

    for i in (1..INTERMEDIATE_SZ_32).rev() {
        intermediate[i - 1] += intermediate[i] / R1_DIV;
        intermediate[i] %= R1_DIV;
    }

    let mut raw_base58 = [0u8; RAW58_SZ_32];
    let mut limbs_done = 0usize;

    ensure_limb(&intermediate, &mut raw_base58, &mut limbs_done, 0);

    let mut raw_leading_0s = 0usize;
    while raw_leading_0s < RAW58_SZ_32 {
        ensure_limb(
            &intermediate,
            &mut raw_base58,
            &mut limbs_done,
            raw_leading_0s / 5,
        );
        if raw_base58[raw_leading_0s] != 0 {
            break;
        }
        raw_leading_0s += 1;
    }

    let skip = raw_leading_0s - in_leading_0s;
    let encoded_length = RAW58_SZ_32 - skip;

    if patterns.is_empty() {
        return 1;
    }

    // Same per-char early-reject walk as the original single-pattern
    // matcher; extra patterns re-use already-emitted limbs.
    let mut mask = 0u64;
    for (pi, p) in patterns.iter().enumerate() {
        if active & (1u64 << pi) == 0 {
            continue;
        }
        let mut ok = true;
        for i in 0..p.prefix_len as usize {
            let rb_idx = skip + i;
            ensure_limb(
                &intermediate,
                &mut raw_base58,
                &mut limbs_done,
                rb_idx / 5,
            );
            if match_lut[raw_base58[rb_idx] as usize] != p.prefix_idx[i]
            {
                ok = false;
                break;
            }
        }
        if ok && p.suffix_len > 0 {
            let suffix_len = p.suffix_len as usize;
            let tail_start = skip + encoded_length - suffix_len;
            let last_limb = (skip + encoded_length - 1) / 5;
            ensure_limb(
                &intermediate,
                &mut raw_base58,
                &mut limbs_done,
                last_limb,
            );
            for i in 0..suffix_len {
                if match_lut[raw_base58[tail_start + i] as usize]
                    != p.suffix_idx[i]
                {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            mask |= 1u64 << pi;
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fast::pubkey_of_seed;
    use sha2::{Digest, Sha512};

    fn encode_matches(
        bytes: &[u8; 32],
        prefix: &str,
        suffix: &str,
        ci: bool,
    ) -> bool {
        let s = fd_bs58::encode_32(*bytes);
        if ci {
            let lc: String =
                s.chars()
                    .map(|c| {
                        if c == 'L' {
                            c
                        } else {
                            c.to_ascii_lowercase()
                        }
                    })
                    .collect();
            (prefix.is_empty() || lc.starts_with(prefix))
                && (suffix.is_empty() || lc.ends_with(suffix))
        } else {
            (prefix.is_empty() || s.starts_with(prefix))
                && (suffix.is_empty() || s.ends_with(suffix))
        }
    }

    #[test]
    fn agrees_with_full_encode_random() {
        let target = MatchTargets::new(&[("zz", "LM")], false);
        for i in 0u32..5000 {
            let h: [u8; 64] = Sha512::digest(i.to_le_bytes()).into();
            let seed: [u8; 32] = h[..32].try_into().unwrap();
            let bytes = pubkey_of_seed(&seed);
            assert_eq!(
                target.match_mask(&bytes) != 0,
                encode_matches(&bytes, "zz", "LM", false),
                "seed {i}"
            );
        }
    }

    #[test]
    fn agrees_with_full_encode_ci() {
        let target = MatchTargets::new(&[("mithriL", "")], true);
        for i in 0u32..2000 {
            let h: [u8; 64] =
                Sha512::digest((i ^ 0xdeadbeef).to_le_bytes()).into();
            let seed: [u8; 32] = h[..32].try_into().unwrap();
            let bytes = pubkey_of_seed(&seed);
            assert_eq!(
                target.match_mask(&bytes) != 0,
                encode_matches(&bytes, "mithriL", "", true),
                "seed {i}"
            );
        }
    }

    #[test]
    fn agrees_on_known_keys() {
        let keys = [
            "XkCriyrNwS3G4rzAXtG5B1nnvb5Ka1JtCku93VqeKAr",
            "11111111111111111111111111111111",
            "1zfbgASTPZHoQ5DhqS5f2bnJk88rxMi137DmZowDztN",
        ];
        for key in keys {
            let bytes = fd_bs58::decode_32(key).unwrap();
            let prefix = &key[..key.len().min(3)];
            let suffix = &key[key.len().saturating_sub(2)..];
            let target = MatchTargets::new(&[(prefix, suffix)], false);
            assert!(target.match_mask(&bytes) != 0, "{key}");
        }
    }

    #[test]
    fn or_of_two_patterns() {
        let key = "XkCriyrNwS3G4rzAXtG5B1nnvb5Ka1JtCku93VqeKAr";
        let bytes = fd_bs58::decode_32(key).unwrap();
        let miss =
            MatchTargets::new(&[("zzz", ""), ("aaa", "qq")], false);
        assert_eq!(miss.match_mask(&bytes), 0);
        let hit =
            MatchTargets::new(&[("zzz", ""), ("XkC", "KAr")], false);
        assert_eq!(hit.match_mask(&bytes), 0b10);
        let pref_only =
            MatchTargets::new(&[("nope", "nope"), ("XkC", "")], false);
        assert_eq!(pref_only.match_mask(&bytes), 0b10);
    }

    #[test]
    fn prefix_ranges_skip_leading_one_and_ci() {
        assert!(prefix_ranges("Harmonic", false).is_some());
        assert!(prefix_ranges("1abc", false).is_none());
        assert!(prefix_ranges("", false).is_none());
        assert!(prefix_ranges("1abc", true).is_none());
        // 2^6 case variants (no uppercase i or o), then 2^9 > 256.
        assert!(prefix_ranges("harmonic", true).is_some());
        assert!(prefix_ranges("abcdefghj", true).is_none());
    }

    #[test]
    fn ci_variants_fold_like_the_matcher() {
        let mut v = ci_variants("cesp99", 64).unwrap();
        v.sort_unstable();
        assert_eq!(v.len(), 16);
        assert_eq!(v[0], "CESP99");
        assert_eq!(v[15], "cesp99");
        // Base58 has no uppercase i or o and no lowercase l.
        assert_eq!(ci_variants("io", 64).unwrap(), ["io"]);
        assert_eq!(ci_variants("L2", 64).unwrap(), ["L2"]);
        let mut v = ci_variants("k", 64).unwrap();
        v.sort_unstable();
        assert_eq!(v, ["K", "k"]);
        assert!(ci_variants("abc", 4).is_none());
        assert_eq!(ci_variants("", 1).unwrap(), [""]);
    }

    /// The case-insensitive byte ranges and filter against a lowercased
    /// encode, on uniform keys and on keys biased toward short addresses.
    #[test]
    fn ci_prefix_filter_agrees_with_encode() {
        let patterns: &[(&str, &str)] =
            &[("ce", ""), ("z", ""), ("b", "q"), ("4h", "")];
        let target = MatchTargets::new(patterns, true);
        let iv = target.prefix_filter(u64::MAX).unwrap();
        let mut hits = [0u32; 4];
        for i in 0u32..100_000 {
            let h: [u8; 64] = Sha512::digest(i.to_le_bytes()).into();
            let mut key: [u8; 32] = h[..32].try_into().unwrap();
            if i % 2 == 0 {
                key[0] = (i % 32) as u8;
            }
            let mut expect = 0u64;
            for (j, (p, sfx)) in patterns.iter().enumerate() {
                if encode_matches(&key, p, sfx, true) {
                    expect |= 1 << j;
                    hits[j] += 1;
                }
            }
            let addr = fd_bs58::encode_32(key);
            assert_eq!(target.match_mask(&key), expect, "{addr}");
            if expect != 0 {
                assert!(passes_filter(&key, &iv), "{addr}");
            }
        }
        assert!(hits.iter().all(|&h| h > 20), "{hits:?}");
    }

    #[test]
    fn prefix_only_range_agrees_with_encode() {
        let key = "XkCriyrNwS3G4rzAXtG5B1nnvb5Ka1JtCku93VqeKAr";
        let bytes = fd_bs58::decode_32(key).unwrap();
        let hit = MatchTargets::new(&[("XkC", "")], false);
        assert_eq!(hit.match_mask(&bytes), 0b1);
        let miss = MatchTargets::new(&[("XkD", "")], false);
        assert_eq!(miss.match_mask(&bytes), 0);
    }

    fn passes_filter(bytes: &[u8; 32], iv: &[(u64, u64)]) -> bool {
        let f = first8(bytes);
        iv.iter()
            .any(|&(lo, span)| f.wrapping_sub(lo) <= span)
    }

    fn to_key(n: &BigUint) -> [u8; 32] {
        let v = n.to_bytes_be();
        let mut out = [0u8; 32];
        out[32 - v.len()..].copy_from_slice(&v);
        out
    }

    /// Exact match and filter against a real base58 encode, for one key.
    fn check_key(
        target: &MatchTargets,
        iv: &[(u64, u64)],
        patterns: &[(&str, &str)],
        key: &[u8; 32],
    ) -> bool {
        let mut expect = 0u64;
        for (i, (p, s)) in patterns.iter().enumerate() {
            if encode_matches(key, p, s, false) {
                expect |= 1 << i;
            }
        }
        assert_eq!(
            target.match_mask(key),
            expect,
            "{}",
            fd_bs58::encode_32(*key)
        );
        assert_eq!(
            target.match_mask_encoded(key, u64::MAX),
            expect,
            "{}",
            fd_bs58::encode_32(*key)
        );
        // Conservative: a real match must always pass the filter.
        if expect != 0 {
            assert!(
                passes_filter(key, iv),
                "{}",
                fd_bs58::encode_32(*key)
            );
        }
        expect != 0
    }

    /// The value base58 decodes `s` to, and 58^len(s).
    fn suffix_value(s: &str) -> (BigUint, BigUint) {
        let mut value = BigUint::from(0u8);
        let mut modulus = BigUint::one();
        for b in s.bytes() {
            value = value * 58u8
                + ALPHABET
                    .iter()
                    .position(|&c| c == b)
                    .unwrap();
            modulus *= 58u8;
        }
        (value, modulus)
    }

    #[test]
    fn prefix_filter_exact_at_range_boundaries() {
        // The largest and smallest filterable keys, as whole addresses.
        let top = fd_bs58::encode_32([0xffu8; 32]);
        let mut bottom_key = [0u8; 32];
        bottom_key[0] = 1;
        let bottom = fd_bs58::encode_32(bottom_key);
        let cases: &[&[(&str, &str)]] = &[
            &[("Cesp99", "")],
            &[("Cesp99", "a")],
            &[("Cesp99", "pq"), ("Cesp", "z")],
            &[("Cesp99", ""), ("zzzz", ""), ("2", "")],
            // A pattern so long that only one address length can hold it.
            &[("Cesp99aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "")],
            &[(top.as_str(), "")],
            &[(bottom.as_str(), "")],
            &[("z", ""), ("2", "")],
        ];
        for patterns in cases {
            let target = MatchTargets::new(patterns, false);
            let iv = target.prefix_filter(u64::MAX).unwrap();
            let max = (BigUint::one() << 256usize) - 1u8;
            let mut boundary_hits = 0;
            for (p, suffix) in patterns.iter() {
                let (want, modulus) = suffix_value(suffix);
                for (lo, hi) in prefix_ranges(p, false).unwrap() {
                    let lo_n = BigUint::from_bytes_be(&lo);
                    let hi_n = BigUint::from_bytes_be(&hi);
                    let mut keys = vec![lo, hi];
                    if lo_n > BigUint::from(0u8) {
                        keys.push(to_key(&(&lo_n - 1u8)));
                    }
                    if hi_n < max {
                        keys.push(to_key(&(&hi_n + 1u8)));
                    }
                    // Same first eight bytes as the bounds, other tails.
                    for b in [lo, hi] {
                        for fill in [0u8, 0xff] {
                            let mut k = [fill; 32];
                            k[..8].copy_from_slice(&b[..8]);
                            keys.push(k);
                        }
                    }
                    // The nearest key to each bound that has the suffix.
                    let up = (&want + &modulus - (&lo_n % &modulus))
                        % &modulus;
                    let first = &lo_n + up;
                    if first <= hi_n {
                        keys.push(to_key(&first));
                    }
                    let down = (&hi_n + &modulus - &want) % &modulus;
                    if down <= hi_n && &hi_n - &down >= lo_n {
                        keys.push(to_key(&(&hi_n - &down)));
                    }
                    for key in &keys {
                        if check_key(&target, &iv, patterns, key) {
                            boundary_hits += 1;
                        }
                    }
                }
            }
            assert!(boundary_hits > 0, "{patterns:?} never matched");
        }
    }

    #[test]
    fn cesp99_ranges_cover_both_address_lengths() {
        let ranges = prefix_ranges("Cesp99", false).unwrap();
        let mut lengths: Vec<usize> = Vec::new();
        for (lo, hi) in &ranges {
            let lo_s = fd_bs58::encode_32(*lo);
            let hi_s = fd_bs58::encode_32(*hi);
            assert!(lo_s.starts_with("Cesp99"), "{lo_s}");
            assert!(hi_s.starts_with("Cesp99"), "{hi_s}");
            assert_eq!(lo_s.len(), hi_s.len());
            lengths.push(lo_s.len());
            // One value below/above leaves the prefix.
            let lo_n = BigUint::from_bytes_be(lo);
            let hi_n = BigUint::from_bytes_be(hi);
            let below = fd_bs58::encode_32(to_key(&(&lo_n - 1u8)));
            let above = fd_bs58::encode_32(to_key(&(&hi_n + 1u8)));
            assert!(!below.starts_with("Cesp99"), "{below}");
            assert!(!above.starts_with("Cesp99"), "{above}");
        }
        lengths.sort_unstable();
        assert_eq!(lengths, [43, 44]);
    }

    #[test]
    fn prefix_filter_agrees_with_encode_on_random_keys() {
        let patterns: &[(&str, &str)] =
            &[("C", ""), ("zz", ""), ("9", "a"), ("Ces", "")];
        let target = MatchTargets::new(patterns, false);
        let iv = target.prefix_filter(u64::MAX).unwrap();
        let mut hits = 0;
        for i in 0u32..50_000 {
            let h: [u8; 64] = Sha512::digest(i.to_le_bytes()).into();
            let key: [u8; 32] = h[..32].try_into().unwrap();
            if check_key(&target, &iv, patterns, &key) {
                hits += 1;
            }
        }
        assert!(hits > 100, "only {hits} hits");
    }

    // Keys biased toward 43-character addresses and leading '1's: a uniform
    // key is a 43-character address only one time in about 30.
    #[test]
    fn prefix_filter_agrees_on_short_addresses() {
        let patterns: &[(&str, &str)] =
            &[("A", ""), ("Z", ""), ("4", ""), ("1", "")];
        let iv_patterns = &patterns[..3];
        let target = MatchTargets::new(iv_patterns, false);
        let iv = target.prefix_filter(u64::MAX).unwrap();
        let mut seen = [false; 3];
        for i in 0u32..50_000 {
            let h: [u8; 64] = Sha512::digest(i.to_le_bytes()).into();
            let mut key: [u8; 32] = h[..32].try_into().unwrap();
            key[0] = (i % 16) as u8;
            let addr = fd_bs58::encode_32(key);
            if addr.len() == 43 {
                seen[0] = true;
            } else if addr.len() == 44 {
                seen[1] = true;
            } else {
                seen[2] = true;
            }
            check_key(&target, &iv, iv_patterns, &key);
        }
        assert_eq!(seen, [true; 3], "did not cover every length");
    }

    #[test]
    fn prefix_filter_only_where_it_is_exact_enough() {
        // No filter once an active pattern needs a real encode.
        let t = MatchTargets::new(&[("abc", ""), ("", "xyz")], false);
        assert!(t.prefix_filter(u64::MAX).is_none());
        // Dropping that kind brings the filter back.
        assert!(t.prefix_filter(0b01).is_some());
        assert!(t.prefix_filter(0b10).is_none());
        let t = MatchTargets::new(&[("1abc", "")], false);
        assert!(t.prefix_filter(u64::MAX).is_none());
        // Case-insensitive: filtered while the case variants are few.
        let t = MatchTargets::new(&[("abc", "")], true);
        assert!(t.prefix_filter(u64::MAX).is_some());
        let t = MatchTargets::new(&[("abcdefghj", "")], true);
        assert!(t.prefix_filter(u64::MAX).is_none());
        // Prefix + suffix filters on the prefix alone.
        let t = MatchTargets::new(&[("abc", "xyz")], false);
        assert!(t.prefix_filter(u64::MAX).is_some());
        // Nothing active: nothing can pass.
        let t = MatchTargets::new(&[("abc", "")], false);
        assert_eq!(t.prefix_filter(0).unwrap(), []);
    }

    #[test]
    fn prefix_filter_merges_overlaps() {
        // "z" contains "zz", so only the wider intervals should remain.
        let wide = MatchTargets::new(&[("z", "")], false)
            .prefix_filter(u64::MAX)
            .unwrap();
        let both = MatchTargets::new(&[("zz", ""), ("z", "")], false)
            .prefix_filter(u64::MAX)
            .unwrap();
        assert_eq!(wide, both);
        for pair in both.windows(2) {
            assert!(pair[0].0 + pair[0].1 < pair[1].0);
        }
    }

    #[test]
    fn suffix_residue_agrees_with_encode() {
        // Every real suffix match must hold the residue; for suffixes of at
        // most four characters, the residue is exact.
        let suffixes =
            ["z", "1", "LM", "11", "zz1", "CooL", "pump", "Cesp99"];
        let mut held = [0u32; 8];
        for i in 0u32..20_000 {
            let h: [u8; 64] = Sha512::digest(i.to_le_bytes()).into();
            let mut key: [u8; 32] = h[..32].try_into().unwrap();
            // Short addresses too, where leading '1's pad the encoding.
            if i % 4 == 0 {
                key[..(i as usize / 4) % 32].fill(0);
            }
            let addr = fd_bs58::encode_32(key);
            for (j, sfx) in suffixes.iter().enumerate() {
                let r = SuffixResidue::new(sfx, false).unwrap();
                let ends = addr.ends_with(sfx);
                if ends {
                    assert!(r.holds(&key), "{addr} {sfx}");
                }
                if sfx.len() <= SUFFIX_RESIDUE_LEN {
                    assert_eq!(r.holds(&key), ends, "{addr} {sfx}");
                }
                held[j] += r.holds(&key) as u32;
            }
        }
        assert!(held[0] > 100 && held[2] > 0, "{held:?}");
        for key in [[0u8; 32], [0xff; 32]] {
            let addr = fd_bs58::encode_32(key);
            let sfx = &addr[addr.len() - 3..];
            assert!(SuffixResidue::new(sfx, false)
                .unwrap()
                .holds(&key));
        }
        assert!(SuffixResidue::new("abc", true).is_none());
        assert!(SuffixResidue::new("", false).is_none());
    }

    #[test]
    fn gpu_blob_roundtrip_header() {
        let t = MatchTargets::new(&[("ab", "xy"), ("Z", "")], false);
        let blob = t.gpu_blob();
        assert_eq!(&blob[..4], &2u32.to_le_bytes());
        assert_eq!(blob[4], 2); // max prefix
        assert_eq!(blob[5], 2); // max suffix
        assert_eq!(blob[6], 2); // plen of first
    }
}
