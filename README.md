# `vanity`

A *bLaZinGlY fAsT* tool for grinding vanity addresses on Solana.

> **This is a fork** of [cavemanloverboy/vanity](https://github.com/cavemanloverboy/vanity)
> by Cavey Cool and Magnetar Fields. All credit for the original tool goes to them.
>
> Changes in this fork:
> - **Faster CPU `grind-keypair`** (AVX-512 IFMA): a wide affine comb table, radix-51 field
>   arithmetic, a specialized SHA-512 and a vectorized prefix filter. About 4-5x the
>   original on a Ryzen AI 9 HX 370, at the cost of about 790 MB of RAM for the table.
> - **Faster CPU `grind`**: precomputed SHA-256 midstates and schedule per seed prefix,
>   16-lane AVX-512 hashing and byte-range matching. About 8x the original.
> - **`vanity wallet`**: check a saved keypair and, on a terminal only, print the base58
>   private key for importing into Phantom or Solflare.
> - Safer key saving, a fix for `grind-doppler` printing the secret seed, and correct
>   case-insensitive odds and ETAs.
>
> `cargo install vanity` installs the original crate from crates.io. To get this fork,
> install it from source:
>
> ```bash
> cargo install --git https://github.com/cesp99/vanity
> ```

## 1) What

Typically, solana developers wishing to obtain a vanity address for their program or token grind out ed25519 keypairs and sign off on a `SystemInstruction::CreateAccount` instruction. However, by using `SystemInstruction::CreateAccountWithSeed`, developers can bypass ed25519 and get extreme speedups on address searches. Although not as generic, this method covers many use cases.

## Install

By default, vanity compiles for cpu:

```bash
cargo install --git https://github.com/cesp99/vanity
```

To compile for an NVIDIA gpu (CUDA) — requires the CUDA toolkit (`nvcc`) on the build machine:

```bash
cargo install --git https://github.com/cesp99/vanity --features=gpu
```

By default the CUDA build emits code for several architectures (Turing through Ada) plus a PTX fallback. For a faster, targeted build set `VANITY_CUDA_ARCH` to your GPU's compute capability — e.g. `86` for an RTX 3090, `89` for an RTX 4090, `80` for an A100, `90` for an H100:

```bash
VANITY_CUDA_ARCH=86 cargo install --git https://github.com/cesp99/vanity --features=gpu
```

If a launch prints `no kernel image is available for execution on the device`, the binary was built without your GPU's architecture — rebuild with the right `VANITY_CUDA_ARCH`.

Keypair CUDA builds use a width-8 affine comb table by default. Wider tables do fewer additions per scalar; `VANITY_COMB_WIDTH` accepts 5 through 12 (8 is ~240M keys/s on a 4090).

On Apple Silicon, build the native Metal backend:

```bash
cargo install --git https://github.com/cesp99/vanity --features=metal
```

Metal kernels compile at runtime with the system Metal framework (macOS only). This is the fast ed25519 path on Apple GPUs.

On other machines without CUDA (AMD / Intel, or no NVIDIA driver), build the OpenCL backend instead:

```bash
cargo install --git https://github.com/cesp99/vanity --features=opencl
```

The OpenCL backend needs only an OpenCL 1.2 ICD loader and headers at build time (`-lOpenCL` on Linux, the system `OpenCL.framework` on macOS); kernels are compiled at runtime for whatever device is present. It exposes the same CLI and `--num-gpus` flag as the CUDA build. The GPU backends are mutually exclusive — if more than one feature is enabled, `metal` is used, otherwise `opencl`, otherwise CUDA.

## Usage

vanity grinds three kinds of vanity output. All three share the same mining flags (`--num-cpus`, `--num-gpus`, `--count`, and for pattern grinds `--max-count`) described in [Common options](#common-options).

| Command | Produces | How the address is derived | Speed |
| --- | --- | --- | --- |
| [`grind`](#grind-a-vanity-seed) | a **seed** for `CreateAccountWithSeed` | `base58(SHA256(base ‖ seed ‖ owner))` | fastest (SHA-256 only) |
| [`grind-keypair`](#grind-a-vanity-keypair) | an **ed25519 keypair** | `base58(ed25519_pubkey(seed))` | slower (ed25519 per attempt) |
| [`grind-doppler`](#grind-a-doppler-keypair) | an **ed25519 keypair** with sign-extendable segments | `ed25519_pubkey(seed)`, matched on bytes | slower (ed25519 per attempt) |

Run `vanity <command> --help` for the full flag list. A fourth command,
[`wallet`](#use-a-found-wallet), checks a keypair file that `grind-keypair`
saved and shows how to use it.

### Grind a vanity seed

`grind` is the fast path. Instead of searching ed25519 keypairs, it searches
*seeds* for `SystemInstruction::CreateAccountWithSeed`, whose resulting address
is `Pubkey::create_with_seed(base, seed, owner) = base58(SHA256(base ‖ seed ‖ owner))`.
No ed25519 is involved, so it's dramatically faster — but the account is
controlled by `base` (the signer), not by a standalone keypair.

A `grind` address has **no private key**. It is a SHA-256 hash, not an
ed25519 public key, so nothing can sign as it: the `base` keypair acts for it
through `CreateAccountWithSeed` and the other `*WithSeed` system instructions.
It cannot be imported into a wallet app, and `vanity wallet` does not apply to
it. For an address you can import and sign with, use
[`grind-keypair`](#grind-a-vanity-keypair).

You supply the `--base` pubkey (the signer for the create instruction), the
program `--owner`, and one or more `--pattern`s to match (base58). A pubkey
hits if it matches **any** pattern; the grind stops after `--count` hits of
**each** pattern. While a harder pattern is still short of `--count`, easier
kinds keep collecting up to `--max-count` (default: `--count`). `...` splits
prefix and suffix (`.` is not in base58):

```bash
vanity grind \
  --base <YOUR_WALLET_PUBKEY> \
  --owner BPFLoaderUpgradeab1e11111111111111111111111 \
  --pattern abc... \
  --num-gpus 1

# several targets in one run (continues until each has --count hits):
vanity grind \
  --base <YOUR_WALLET_PUBKEY> \
  --owner BPFLoaderUpgradeab1e11111111111111111111111 \
  --pattern Cavey...CooL \
  --pattern Harmonic... \
  --num-gpus 1
```

The match line reports the resulting address and the 16-character seed:

```text
gpu 0 match: abcJvoUM9mcmHewxCumNuG8sQGM3u51QB8Wi1RVmats in 0.603s
out seed = [...] -> JcZmEdyCAWJ7q9XI
```

On CPU, `grind` uses AVX-512 when the CPU has it (the `sha2` crate otherwise).
Each thread fixes the first 12 seed characters and walks the last 4, so most
of the SHA-256 work is done once per 14.8M seeds. A seed only names an account
that `base` controls, so CPU seeds come from a counter with a random start per
run rather than from a CSPRNG. They are still 16 alphanumeric characters.

Recompute the address from a `base`/`owner`/`seed` at any time with `verify`:

```bash
vanity verify \
  --base <YOUR_WALLET_PUBKEY> \
  --owner BPFLoaderUpgradeab1e11111111111111111111111 \
  --seed JcZmEdyCAWJ7q9XI
```

To use the seed on-chain, pass it to `create_account_with_seed` (the resulting
address is the `to_pubkey`):

```rust
pub fn create_account_with_seed(
    from_pubkey: &Pubkey,
    // this is the resulting address, obtained via Pubkey::create_with_seed
    to_pubkey: &Pubkey,
    base: &Pubkey,
    seed: &str,
    lamports: u64,
    space: u64,
    owner: &Pubkey,
) -> Instruction
```

### Grind a vanity keypair

`grind-keypair` searches genuine ed25519 keypairs whose **public key** matches
any `--pattern` (same `Prefix...Suffix` syntax as `grind`), and keeps going
until `--count` hits of each pattern (extras of an easy kind are kept up
to `--max-count`). This is the classic
"vanity wallet/mint" case: the result is a standalone keypair you can sign
with. It's slower than `grind` because each attempt computes an ed25519 scalar
multiplication. The CPU path uses a batched custom ed25519 implementation
(AVX-512 IFMA on x86, 4-lane NEON on Apple Silicon); GPUs use the CUDA/OpenCL/Metal keypair kernels.

```bash
vanity grind-keypair --pattern bob...xyz --num-gpus 1
vanity grind-keypair --pattern Cavey...CooL --pattern Harmonic... --num-gpus 1
```

On a match it prints the address and saves the keypair as `<ADDRESS>.json`
in the current directory. The file is a solana-keygen compatible JSON array
of 64 numbers (the 32-byte secret seed, then the public key), readable only
by you (mode 0600). The secret key is not printed:

```text
match: bob...xyz
pubkey: bob...xyz
keypair saved to: /home/you/vanity/bob...xyz.json
```

GPU hits print `gpu 0 match: <ADDRESS> in 0.4s` in place of `match:`. Use
the file with [`vanity wallet`](#use-a-found-wallet).

Before grinding, vanity checks that it can create a file in the current
directory and stops with an error if it cannot. Each key is written to a temp
file, synced to disk, then linked into place, so a key file is either
complete or absent, and an existing file is never overwritten. If the current
directory stops working later (disk full, removed, remounted read-only), the
key goes to `$HOME`, then to the temp directory (`$TMPDIR` or `/tmp`), with a
`FALLBACK` notice and the full path. Only if every location fails does vanity
print the keypair JSON to stderr, so that the key is not lost. That is the one
case where it shows a secret; copy it into a file straight away.

Which prefixes are easy: an address is a 32-byte number in base58, so it is
43 or 44 characters long. About 94% of addresses have 44 characters, and a
44-character address can only start with `1` to `J`. A prefix whose first
character comes after `J` in the base58 alphabet (`K`-`Z`, `a`-`z`) only
occurs in 43-character addresses, which makes it about 58 times rarer than a
prefix of the same length starting with `2` to `H` (`J` itself is about 14
times rarer). `Czzz` is about 58 times easier than `czzz` or `Kzzz`. The
probability and expected attempts that vanity prints at the start already
account for this.

Add `--case-insensitive` to match the prefix/suffix ignoring case (except `L`,
which has no lowercase form in base58). The case variants are not equally
likely, for the reason above: `cesp99` matches 16 variants, and the 8 that
start with `C` make up almost all of its probability, so it is about 8 times
easier than `Cesp99` rather than 16. The printed estimate sums the variants.

### Grind a doppler keypair

`grind-doppler` searches ed25519 keypairs whose **public key** packs into
sign-extendable 32-bit immediates — useful in sBPF / VM contexts where a 64-bit
key segment that sign-extends from an `i32` can be compared with a cheaper
immediate instruction. (Inspired by
[doppler-keygen](https://github.com/blueshift-gg/doppler-keygen).)

The 32-byte pubkey is split into four 8-byte segments. A segment is
*sign-extendable* when its low 4 bytes form a little-endian `i32` and its high
4 bytes are that value's sign extension:

- bit 31 clear (positive `i32`): high 4 bytes are `00 00 00 00`
- bit 31 set (negative `i32`): high 4 bytes are `FF FF FF FF`

`--segments N` (1–4) requires **at least N** of the four segments to be
sign-extendable. Each segment matches with probability 2⁻³², so `--segments 1`
needs ~1 billion keypairs (seconds-to-minutes on a GPU) while `N ≥ 2` is
exponentially rarer.

```bash
# any one segment
vanity grind-doppler --segments 1 --num-gpus 1

# require two sign-extendable segments
vanity grind-doppler --segments 2 --num-gpus 1
```

On a match it prints the address and a per-segment breakdown of the public
key, including the assembly `.equ` constants for each segment, and saves the
keypair file the same way as `grind-keypair`:

```text
gpu 0 match: Gyqq...SrPP in 0.4s
pubkey: Gyqq...SrPP
doppler: 1/4 sign-extendable segment(s)
  seg 0 (bytes 0-7):  0x2a768d76fb0b6fed (not sign-extendable)  =>  .equ EXPECTED_KEY_0, 0x2a768d76fb0b6fed
  seg 1 (bytes 8-15): imm32 149357322 (0x08e7030a)              =>  .equ EXPECTED_KEY_1, 0x08e7030a
  ...
keypair saved to: /home/you/vanity/Gyqq...SrPP.json
```

### Use a found wallet

The `<ADDRESS>.json` file that `grind-keypair` and `grind-doppler` save is a
normal Solana keypair file. `vanity wallet` checks one and shows how to use
it. It requires the file to hold 64 numbers, requires the stored public key
to be the one the secret key derives, signs and verifies a test message, and
warns if other users can read the file. It prints no secret:

```bash
vanity wallet bob...xyz.json
```

```text
address:    bob...xyz
file:       /home/you/vanity/bob...xyz.json
self-check: ok (signed and verified a test message)

Solana CLI:
  solana-keygen pubkey /home/you/vanity/bob...xyz.json
  solana balance bob...xyz
  solana transfer --from /home/you/vanity/bob...xyz.json <RECIPIENT> <AMOUNT>
...
```

**Solana CLI.** The file works as-is anywhere the CLI takes a keypair:

```bash
solana-keygen pubkey bob...xyz.json              # prints the address
solana-keygen verify <ADDRESS> bob...xyz.json    # checks file and address agree
solana balance <ADDRESS>
solana transfer --from bob...xyz.json <RECIPIENT> <AMOUNT>
solana config set --keypair /full/path/bob...xyz.json   # make it the default wallet
```

**Phantom, Solflare and other wallet apps.** Their "Import private key"
screen takes the 64-byte secret key in base58. `vanity wallet <FILE> --reveal`
prints it, after a warning. It refuses unless stdout is an interactive
terminal, so the key cannot end up in a pipe, a log file, or an AI assistant's
transcript. Run it yourself in a terminal, paste the key into the wallet app,
then clear the terminal and its scrollback.

**Keep it safe.**

- The JSON file *is* the wallet. Anyone with a copy controls the address, and
  if every copy is lost, so are the funds. Back it up offline (for example an
  encrypted USB drive or a password manager) before sending anything to the
  address, and keep it at mode 0600 (`chmod 600 <FILE>`).
- Never paste the file or the revealed key into a website, chat or support
  form. Importing the key into a wallet app puts a copy on that device.
- Move funds carefully: send a small test amount first, check it arrived with
  `solana balance <ADDRESS>`, then send the rest.
- vanity makes no network calls. It cannot check balances or send funds; use
  the Solana CLI or a wallet app for that.

### Common options

| Flag | Applies to | Default | Meaning |
| --- | --- | --- | --- |
| `--num-cpus <N>` | all | `0` (= all logical cores) | CPU mining threads |
| `--num-gpus <N>` | all (GPU builds only) | `1` | GPUs to mine on |
| `--count <N>` | `grind`, `grind-keypair` | `1` | stop once **every** `--pattern` has N matches |
| `--max-count <N>` | `grind`, `grind-keypair` | `--count` | keep saving each `--pattern` up to N while waiting; then drop that kind from the search |
| `--count <N>` | `grind-doppler` | `1` | stop after N matching keypairs |
| `--case-insensitive` | `grind`, `grind-keypair` | off | match patterns ignoring case |
| `--pattern` | `grind`, `grind-keypair` | — | repeatable `Prefix...Suffix` / `Prefix...` / `...Suffix`; a hit matches any unfilled kind |
| `--segments <1-4>` | `grind-doppler` | `1` | sign-extendable segments required |

To run purely on CPU (no GPU), build without a GPU feature, or pass `--num-gpus 0`.

## Contributions

yes

## Performance

Approximate single-device throughput:

| Backend | Device | seeds/s | keypairs/s |
| --- | --- | --- | --- |
| CPU    | AMD EPYC 9275F (48 threads, AVX-512 IFMA) | ~201 M | ~33 M |
| CPU    | AMD Ryzen AI 9 HX 370 (24 threads, AVX-512 IFMA; busy desktop, median of 3) | ~383 M | ~27 M |
| CUDA   | RTX 4090   | ~8.6B B | ~240 M |
| OpenCL | Apple Silicon | ~315 M  | ~15 M  |
| Metal  | Apple M5 Pro (20 GPU cores) | unmeasured | ~36 M |

Metal keypair throughput is `grind-doppler` on this machine (`--num-gpus 1 --num-cpus 1`): about 36M keys/s, versus about 13M keys/s for the OpenCL backend on the same GPU. The kernel is a width-12 affine comb stored as 128-byte records (one cache line per table entry). `fe_mul`, `fe_sq`, and the 32-byte SHA-512 stay real calls so their temporaries do not spill the comb.

`grind` is far faster because each attempt is just a SHA-256 hash. Keypair modes (`grind-keypair`/`grind-doppler`) perform full sha512 and ed25519 scalar multiplication per attempt. Throughput scales roughly linearly with `--num-gpus`.


## Acknowledgements, External Libraries

- The sha2 implementation used in this library is taken from [here](https://github.com/mochimodev/cuda-hashing-algos), which is in the public domain.
- The base58 encoding implementation is taken from firedancer with heavy modifications for use in cuda & case insensitive encodings, licensed under APACHE-2.0
