#!/usr/bin/env bash
# Everything the engine needs besides the code, into models/ (or $MODEL_DIR):
#   1. the model: ISTA-DASLab's GSQ-RCO IQ3_XXS GGUF of Qwen3.8-Flash-Next, 2 files, 75.8 GB (resumable, SHA-256
#      checked: SKIP_SHA=1 skips the ~5 minutes of hashing)
#   2. the MTP draft layer (speculative decoding): its tensors from the official BF16 checkpoint, ~5.5 GB by HTTP
#      Range requests (tools/fetch_mtp.py), then packed into models/mtp-q2_0.gguf (build/bl-mtp-pack: build first)
# Mirror: HF_ENDPOINT=https://hf-mirror.com scripts/download_model.sh. A proxy is taken from https_proxy.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_DIR=${MODEL_DIR:-models}
ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
REPO=ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
REV=ed59f92082b1e93c0e96d60a8b11aab089b52f09
FILES=(
  "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf 47039860096 219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15"
  "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf 28800138432 316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"
)

mkdir -p "$MODEL_DIR"
for f in "${FILES[@]}"; do
  read -r name size sha <<<"$f"
  out="$MODEL_DIR/$name"
  have=$( [ -f "$out" ] && stat -c %s "$out" || echo 0 )
  if [ "$have" != "$size" ]; then
    echo "== $name ($((size / 1000000000)) GB)"
    curl -fL --retry 10 --retry-delay 5 -C - -o "$out" "$ENDPOINT/$REPO/resolve/$REV/IQ3_XXS/$name"
  fi
  [ "$(stat -c %s "$out")" = "$size" ] || { echo "$out: wrong size; delete it and run again"; exit 1; }
  if [ "${SKIP_SHA:-0}" != 1 ]; then
    echo "== checking $name"
    echo "$sha  $out" | sha256sum -c --quiet || { echo "$out: SHA-256 mismatch; delete it and run again"; exit 1; }
  fi
done

if [ ! -f "$MODEL_DIR/mtp-q2_0.gguf" ]; then
  [ -x build/bl-mtp-pack ] || { echo "build first (see README: cmake -B build && cmake --build build -j)"; exit 1; }
  echo "== the MTP draft layer"
  python3 tools/fetch_mtp.py --out "$MODEL_DIR/mtp-src"
  build/bl-mtp-pack "$MODEL_DIR/mtp-src" "$MODEL_DIR/mtp-q2_0.gguf"
  echo "(the raw tensors in $MODEL_DIR/mtp-src can be deleted now)"
fi

echo
echo "Ready. Start the server:"
echo "  build/bl-server --model $MODEL_DIR/${FILES[0]%% *} --host 0.0.0.0 --port 8080"
