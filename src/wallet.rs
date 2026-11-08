//! `vanity wallet`: check a keypair file written by grind-keypair and show
//! how to use it. No secret byte is printed unless `--reveal` is given and
//! stdout is a terminal. Errors never quote the file, which holds a secret.

use ed25519_dalek::{Signer, SigningKey};
use std::{
    fs,
    io::{Read, Write},
    path::Path,
};

/// A keypair file is about 300 bytes; anything far larger is not one.
const MAX_FILE_LEN: u64 = 64 * 1024;

const SELF_CHECK_MESSAGE: &[u8] = b"vanity wallet self-check";

/// Read a solana-keygen keypair file: a JSON array of 64 numbers, the
/// ed25519 seed followed by its public key. The public half must be the
/// one the seed derives.
pub fn load_keypair(path: &Path) -> Result<SigningKey, String> {
    let shown = path.display();
    let read_err =
        |e: std::io::Error| format!("cannot read {shown}: {e}");
    // Devices, FIFOs and procfs files report length 0 and can be endless,
    // and opening a FIFO blocks, so check the type before opening.
    let meta = fs::metadata(path).map_err(read_err)?;
    if !meta.is_file() {
        return Err(format!("{shown} is not a regular file"));
    }
    let len = meta.len();
    if len > MAX_FILE_LEN {
        return Err(format!(
            "{shown} is {len} bytes, too large for a keypair file"
        ));
    }
    let mut text = String::new();
    fs::File::open(path)
        .and_then(|f| {
            f.take(MAX_FILE_LEN + 1)
                .read_to_string(&mut text)
        })
        .map_err(read_err)?;
    if text.len() as u64 > MAX_FILE_LEN {
        return Err(format!("{shown} is too large for a keypair file"));
    }
    let bytes = parse_byte_array(&text).ok_or_else(|| {
        format!("{shown} is not a JSON array of numbers 0-255")
    })?;
    let Ok(bytes) = <[u8; 64]>::try_from(bytes.as_slice()) else {
        return Err(format!(
            "{shown} holds {} numbers; a keypair file holds 64",
            bytes.len()
        ));
    };
    let key = SigningKey::from_bytes(bytes[..32].try_into().unwrap());
    if key.verifying_key().as_bytes()[..] != bytes[32..] {
        return Err(format!(
            "the public key stored in {shown} is not the one its secret \
             key derives; the file is damaged or was not written by \
             vanity or solana-keygen"
        ));
    }
    Ok(key)
}

fn parse_byte_array(text: &str) -> Option<Vec<u8>> {
    let inner = text
        .trim()
        .strip_prefix('[')?
        .strip_suffix(']')?;
    if inner.trim().is_empty() {
        return Some(Vec::new());
    }
    inner
        .split(',')
        .map(|n| n.trim().parse().ok())
        .collect()
}

/// Validate the keypair at `path` and print its address and usage hints
/// to `out`. With `reveal`, also print the base58 secret key, but only
/// when `out` is an interactive terminal (`out_is_tty`): never into a pipe,
/// file, log or tool transcript.
pub fn run(
    path: &Path,
    reveal: bool,
    out_is_tty: bool,
    out: &mut impl Write,
) -> Result<(), String> {
    if reveal && !out_is_tty {
        return Err(
            "refusing to --reveal: stdout is not a terminal. The private \
             key is only printed to an interactive terminal, never into a \
             pipe, file or log."
                .into(),
        );
    }
    let key = load_keypair(path)?;
    let signature = key.sign(SELF_CHECK_MESSAGE);
    key.verifying_key()
        .verify_strict(SELF_CHECK_MESSAGE, &signature)
        .map_err(|_| "self-check failed: the key cannot sign")?;

    let address = fd_bs58::encode_32(key.verifying_key().to_bytes());
    let file = std::path::absolute(path)
        .unwrap_or_else(|_| path.to_path_buf());
    let file = file.display();
    let w = |e: std::io::Error| format!("cannot write output: {e}");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(path)
            .map_err(|e| format!("cannot read {file}: {e}"))?
            .permissions()
            .mode()
            & 0o777;
        if mode & 0o077 != 0 {
            writeln!(
                out,
                "warning: {file} is accessible by other users (mode \
                 {mode:03o}); fix with: chmod 600 {file}\n"
            )
            .map_err(w)?;
        }
    }

    writeln!(
        out,
        "address:    {address}\n\
         file:       {file}\n\
         self-check: ok (signed and verified a test message)\n\
         \n\
         Solana CLI:\n\
         \x20 solana-keygen pubkey {file}\n\
         \x20 solana balance {address}\n\
         \x20 solana transfer --from {file} <RECIPIENT> <AMOUNT>\n\
         \n\
         The file is the wallet: anyone with a copy controls the address.\n\
         Back it up offline before you fund the address."
    )
    .map_err(w)?;

    if !reveal {
        writeln!(
            out,
            "\nTo import into Phantom or Solflare, run on a terminal:\n\
             \x20 vanity wallet {file} --reveal"
        )
        .map_err(w)?;
        return Ok(());
    }
    writeln!(
        out,
        "\nWARNING: the next line is the PRIVATE KEY of {address}.\n\
         Anyone who sees it can take everything in the wallet. Paste it \
         only into a wallet's\n\"import private key\" screen, and clear \
         your terminal and its scrollback afterwards.\n"
    )
    .map_err(w)?;
    writeln!(out, "{}", fd_bs58::encode_64(key.to_keypair_bytes()))
        .map_err(w)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Throwaway test key; never used for anything real.
    const SEED: [u8; 32] = [7u8; 32];

    fn key_bytes() -> [u8; 64] {
        SigningKey::from_bytes(&SEED).to_keypair_bytes()
    }

    fn json(bytes: &[u8]) -> String {
        format!("{bytes:?}")
    }

    /// A test file in its own directory, both deleted on drop. Tests run in
    /// parallel, so a shared directory could be removed under another test.
    struct TempFile(PathBuf);

    impl std::ops::Deref for TempFile {
        type Target = Path;
        fn deref(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
            let _ = fs::remove_dir(self.0.parent().unwrap());
        }
    }

    fn temp_file(name: &str, contents: &str) -> TempFile {
        let dir = std::env::temp_dir().join(format!(
            "vanity-wallet-test-{}-{name}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        fs::write(&path, contents).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(
                &path,
                fs::Permissions::from_mode(0o600),
            )
            .unwrap();
        }
        TempFile(path)
    }

    fn run_to_string(
        path: &Path,
        reveal: bool,
        tty: bool,
    ) -> (Result<(), String>, String) {
        let mut out = Vec::new();
        let result = run(path, reveal, tty, &mut out);
        (result, String::from_utf8(out).unwrap())
    }

    /// Every form in which the secret half could leak into output.
    fn secret_forms() -> Vec<String> {
        let kp = key_bytes();
        let hex: String = SEED
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        vec![
            fd_bs58::encode_64(kp),
            fd_bs58::encode_32(SEED),
            json(&SEED),
            json(&kp),
            hex,
            SEED.iter()
                .map(|b| b.to_string())
                .collect::<Vec<_>>()
                .join(","),
        ]
    }

    #[test]
    fn valid_file_shows_address_and_no_secret() {
        let path = temp_file("valid.json", &json(&key_bytes()));
        let (result, out) = run_to_string(&path, false, true);
        assert_eq!(result, Ok(()));
        let address = fd_bs58::encode_32(&key_bytes()[32..]);
        assert!(out.contains(&format!("address:    {address}")));
        assert!(out.contains("solana-keygen pubkey"));
        assert!(out.contains(&format!("solana balance {address}")));
        assert!(out.contains("solana transfer --from"));
        assert!(!out.contains("warning"));
        for secret in secret_forms() {
            assert!(!out.contains(&secret), "output leaks {secret}");
        }
    }

    #[test]
    fn reads_compact_and_spaced_json() {
        let compact = json(&key_bytes()).replace(' ', "");
        let path = temp_file("compact.json", &compact);
        assert!(load_keypair(&path).is_ok());
        let spaced = format!("\n {} \n", json(&key_bytes()));
        let path = temp_file("spaced.json", &spaced);
        assert!(load_keypair(&path).is_ok());
    }

    #[test]
    fn wrong_length_is_rejected() {
        let path = temp_file("short.json", &json(&key_bytes()[..63]));
        let err = load_keypair(&path).unwrap_err();
        assert!(err.contains("63 numbers"), "{err}");
        let path = temp_file("seed-only.json", &json(&SEED));
        assert!(load_keypair(&path).is_err());
    }

    #[test]
    fn mismatched_pubkey_is_rejected() {
        let mut bytes = key_bytes();
        bytes[63] ^= 1;
        let path = temp_file("mismatch.json", &json(&bytes));
        let err = load_keypair(&path).unwrap_err();
        assert!(err.contains("not the one its secret key derives"));
        for secret in secret_forms() {
            assert!(!err.contains(&secret));
        }
    }

    #[test]
    fn non_json_is_rejected_without_quoting_the_file() {
        for (i, bad) in [
            "not json at all",
            "{\"key\": 1}",
            "[1, 2, 256]",
            "[1, 2, -1]",
            "[1, 2,, 3]",
            "",
        ]
        .iter()
        .enumerate()
        {
            let path = temp_file(&format!("bad{i}.json"), bad);
            let err = load_keypair(&path).unwrap_err();
            assert!(err.contains("not a JSON array"), "{err}");
            if !bad.is_empty() {
                assert!(!err.contains(bad), "{err}");
            }
        }
    }

    #[test]
    fn reveal_refuses_when_not_a_terminal() {
        let path = temp_file("refuse.json", &json(&key_bytes()));
        let (result, out) = run_to_string(&path, true, false);
        let err = result.unwrap_err();
        assert!(err.contains("refusing to --reveal"), "{err}");
        assert!(out.is_empty());
        for secret in secret_forms() {
            assert!(!err.contains(&secret));
        }
    }

    #[test]
    fn reveal_on_terminal_prints_importable_key() {
        let path = temp_file("reveal.json", &json(&key_bytes()));
        let (result, out) = run_to_string(&path, true, true);
        assert_eq!(result, Ok(()));
        let warning = out.find("WARNING").unwrap();
        let secret = fd_bs58::encode_64(key_bytes());
        let shown = out.find(&secret).unwrap();
        assert!(warning < shown);
        assert_eq!(fd_bs58::decode_64(&secret).unwrap(), key_bytes());
    }

    #[test]
    fn oversized_file_is_rejected() {
        let path = temp_file(
            "big.json",
            &" ".repeat(MAX_FILE_LEN as usize + 1),
        );
        let err = load_keypair(&path).unwrap_err();
        assert!(err.contains("too large"), "{err}");
    }

    // Length 0 and endless: the size check alone would read until memory
    // runs out.
    #[cfg(unix)]
    #[test]
    fn special_files_are_rejected_without_reading() {
        for dev in ["/dev/zero", "/dev/urandom"] {
            let err = load_keypair(Path::new(dev)).unwrap_err();
            assert!(err.contains("not a regular file"), "{err}");
        }
        let dir = temp_file("dir-marker", "");
        let err = load_keypair(dir.parent().unwrap()).unwrap_err();
        assert!(err.contains("not a regular file"), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn loose_permissions_warn() {
        use std::os::unix::fs::PermissionsExt;
        let path = temp_file("loose.json", &json(&key_bytes()));
        fs::set_permissions(&*path, fs::Permissions::from_mode(0o644))
            .unwrap();
        let (result, out) = run_to_string(&path, false, true);
        assert_eq!(result, Ok(()));
        assert!(out.contains("mode 644"), "{out}");
        assert!(out.contains("chmod 600"));
    }
}
