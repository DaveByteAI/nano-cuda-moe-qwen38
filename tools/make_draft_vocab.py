#!/usr/bin/env python3
"""data/draft-vocab.bin: the tokens the MTP draft head scores (int32 ids, ascending).

Scoring all 248,320 tokens costs the draft layer ~0.8 ms per draft; a subset that covers what the model writes in
practice keeps nearly all of the drafts. The rule, from the vocabulary alone:
  - every token of Chinese, Japanese or Korean text (with spaces, digits and ASCII mixed in);
  - ASCII tokens (English, code) whose id is below --ascii-below: byte-level BPE ids follow merge order, so the
    low ids are the common pieces;
  - whitespace and digit tokens, and every special (non-normal) token.
Other scripts are left out: a draft that cannot be made only costs speed (the main model decides every token).

    python3 tools/make_draft_vocab.py models/...-00001-of-00002.gguf [--out data/draft-vocab.bin]

Reads the vocabulary from the GGUF header with the Python standard library.
"""
import argparse
import struct

# GGUF metadata value types
U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)
SIZES = {U8: 1, I8: 1, U16: 2, I16: 2, U32: 4, I32: 4, F32: 4, BOOL: 1, U64: 8, I64: 8, F64: 8}


def read_vocab(path):
    """tokenizer.ggml.tokens and tokenizer.ggml.token_type from a GGUF file."""
    with open(path, "rb") as f:
        def u(fmt):
            return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]

        def string():
            return f.read(u("Q")).decode("utf-8", "replace")

        def value(t):
            if t == STRING:
                return string()
            if t == ARRAY:
                et, n = u("I"), u("Q")
                if et == STRING:
                    return [f.read(u("Q")) for _ in range(n)]
                data = f.read(SIZES[et] * n)
                return list(struct.unpack("<" + {I32: "i", U32: "I"}.get(et, "B") * n, data)) if et in (I32, U32) else data
            return f.read(SIZES[t])

        if f.read(4) != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF file")
        u("I")
        u("Q")
        n_kv = u("Q")
        want = {}
        for _ in range(n_kv):
            key = string()
            v = value(u("I"))
            if key in ("tokenizer.ggml.tokens", "tokenizer.ggml.token_type"):
                want[key] = v
            if len(want) == 2:
                return want["tokenizer.ggml.tokens"], want["tokenizer.ggml.token_type"]
    raise SystemExit(f"{path}: no vocabulary")


def byte_decoder():
    """GPT-2 byte-level BPE stores each byte as a printable character; this maps them back."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def is_cjk(ch):
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0x20000 <= o <= 0x2FA1F or 0x3000 <= o <= 0x30FF or
            0x31F0 <= o <= 0x31FF or 0xFF00 <= o <= 0xFFEF or 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF or
            0x3130 <= o <= 0x318F or 0xF900 <= o <= 0xFAFF)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf")
    ap.add_argument("--out", default="data/draft-vocab.bin")
    ap.add_argument("--ascii-below", type=int, default=40960, help="ASCII tokens with a smaller id are kept")
    a = ap.parse_args()
    tokens, types = read_vocab(a.gguf)
    dec = byte_decoder()
    keep, kinds = [], {"cjk": 0, "ascii": 0, "space/digit": 0, "special": 0}
    for i, (tok, typ) in enumerate(zip(tokens, types)):
        if typ != 1:   # control / user-defined tokens
            keep.append(i)
            kinds["special"] += 1
            continue
        raw = bytes(dec.get(ch, 0x3F) for ch in tok.decode("utf-8", "replace"))
        text = raw.decode("utf-8", "replace")
        chars = [ch for ch in text if not (ch.isspace() or ch.isdigit())]
        if not chars:
            keep.append(i)
            kinds["space/digit"] += 1
        elif all(ord(ch) < 128 for ch in chars):
            if i < a.ascii_below:
                keep.append(i)
                kinds["ascii"] += 1
        elif all(ord(ch) < 128 or is_cjk(ch) for ch in chars) and "�" not in text:
            keep.append(i)
            kinds["cjk"] += 1
    with open(a.out, "wb") as f:
        f.write(struct.pack(f"<{len(keep)}i", *keep))
    print(f"{len(keep)} of {len(tokens)} tokens: " + ", ".join(f"{k} {v}" for k, v in kinds.items()) + f" -> {a.out}")


if __name__ == "__main__":
    main()
