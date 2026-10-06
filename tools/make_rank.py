#!/usr/bin/env python3
"""data/expert-rank.bin from routing traces: every (layer, expert) pair ranked by how often the router chose it.

The engine fills its VRAM expert cache in this order at startup (then the LRU takes over), so the experts used most
start on the GPU. Record a trace by running text through the engine with BL_ROUTE_TRACE=FILE (decode windows only,
so BL_PREFILL=0 to have the prompt traced too), then:

    python3 tools/make_rank.py TRACE [TRACE...] [--out data/expert-rank.bin]

Trace records: int32 sequence, int32 layer, 10 x int32 expert ids. Output ("BLEC" v1): uint32 version, n_layer,
n_expert, count, then count pairs of uint16 (layer, expert), the most used first. Python standard library only.
"""
import argparse
import struct
from collections import Counter

K, N_LAYER, N_EXPERT = 10, 48, 512


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out", default="data/expert-rank.bin")
    a = ap.parse_args()
    use, tokens = Counter(), 0
    rec = struct.Struct(f"<2i{K}i")
    for path in a.traces:
        data = open(path, "rb").read()
        if len(data) % rec.size:
            raise SystemExit(f"{path}: not a whole number of {rec.size}-byte records")
        for v in rec.iter_unpack(data):
            layer = v[1]
            if layer < N_LAYER:
                tokens += layer == 0
                for e in v[2:]:
                    use[(layer, e)] += 1
    ranked = sorted(use, key=lambda p: (-use[p], p))
    with open(a.out, "wb") as f:
        f.write(b"BLEC" + struct.pack("<4I", 1, N_LAYER, N_EXPERT, len(ranked)))
        for layer, e in ranked:
            f.write(struct.pack("<2H", layer, e))
    print(f"{tokens} tokens traced; {len(ranked)} of {N_LAYER * N_EXPERT} experts used; wrote {a.out}")


if __name__ == "__main__":
    main()
