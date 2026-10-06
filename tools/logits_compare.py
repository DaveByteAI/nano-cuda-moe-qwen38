#!/usr/bin/env python3
"""Teacher-forced comparison of two runs' logits: KL(ref || ours), top-1 agreement, per position.
   logits_compare.py REF_DUMP_DIR OURS.logits      (ours: T rows of n_vocab float32, from bl-run --save-logits)
The reference is bl-ref-dump's `result_output` (T rows when the dump asked for logits at every position)."""
import sys
import numpy as np

ref_dir, ours_path = sys.argv[1:3]
rows = [l.split('\t') for l in open(f'{ref_dir}/index.tsv').read().splitlines()[1:]]
e = [r for r in rows if r[0] == 'result_output'][-1]
n_vocab, T = int(e[2]), int(e[3])
ref = np.fromfile(f'{ref_dir}/data.bin', dtype=np.float32, count=n_vocab * T, offset=int(e[6])).reshape(T, n_vocab)
ours = np.fromfile(ours_path, dtype=np.float32).reshape(-1, n_vocab)
if ours.shape[0] > T:   # ours covers more positions: its last T rows are the reference's
    ours = ours[-T:]
if ours.shape[0] != T:
    sys.exit(f'reference has {T} positions, ours {ours.shape[0]}')

def log_softmax(x):
    x = x.astype(np.float64) - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))

lr, lo = log_softmax(ref), log_softmax(ours)
kl = (np.exp(lr) * (lr - lo)).sum(axis=1)
agree = ref.argmax(1) == ours.argmax(1)
print(f'positions {T}: mean KL {kl.mean():.5f} nats, median {np.median(kl):.5f}, max KL {kl.max():.5f} (pos {kl.argmax()}), '
      f'top-1 agreement {agree.mean() * 100:.1f}% ({agree.sum()}/{T})')
