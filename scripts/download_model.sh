#!/usr/bin/env bash
# Everything the engine needs besides the code, into models/ (or $MODEL_DIR):
#   1. the model: ISTA-DASLab's GSQ-RCO GGUF of Qwen3.8-Flash-Next, 2 files (resumable, SHA-256 checked: SKIP_SHA=1
#      skips the ~5 minutes of hashing). The quantization is the argument, IQ3_XXS by default (the tested one):
#        scripts/download_model.sh [IQ3_XXS | IQ2_XS | Q2_0]     (IQ3_S: not supported yet, see README)
#      The second file (the 28.8 GB n-gram table) is the same in all four: one already here is linked, not fetched.
#   2. the MTP draft layer (speculative decoding): its tensors from the official BF16 checkpoint, ~5.5 GB by HTTP
#      Range requests (tools/fetch_mtp.py), then packed into models/mtp-q2_0.gguf (build/bl-mtp-pack: build first)
# Mirror: HF_ENDPOINT=https://hf-mirror.com scripts/download_model.sh. A proxy is taken from https_proxy.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_DIR=${MODEL_DIR:-models}
ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
REPO=ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
REV=ed59f92082b1e93c0e96d60a8b11aab089b52f09
QUANT=${1:-IQ3_XXS}
case "$QUANT" in   # the first file's size and SHA-256 at REV
  IQ3_XXS) S1="47039860096 219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15" ;;
  IQ3_S)   S1="54817524224 4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3" ;;
  IQ2_XS)  S1="39225954592 92cee27ae5bbadcd732416a0f7a7f0acc092399dbbe8f5a5efa707c2ec0a49d7" ;;
  Q2_0)    S1="37623740192 69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720" ;;
  *) echo "unknown quantization $QUANT: IQ3_XXS (default), IQ3_S, IQ2_XS or Q2_0"; exit 2 ;;
esac
if [ "$QUANT" = IQ3_S ] && [ "${FORCE:-0}" != 1 ]; then   # (FORCE=1 downloads it anyway)
  echo "IQ3_S is not supported yet: one layer's experts are IQ4_XS, which the GPU expert kernels do not handle"; exit 2
fi
[ "$QUANT" = IQ3_XXS ] || echo "note: only IQ3_XXS has been tested with this engine; $QUANT is unverified (the engine checks the formats at start)"
S2="28800138432 316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"
P="Qwen3.8-Flash-Next-GSQ-RCO-$QUANT"
FILES=("$P-00001-of-00002.gguf $S1" "$P-00002-of-00002.gguf $S2")

mkdir -p "$MODEL_DIR"
out2="$MODEL_DIR/$P-00002-of-00002.gguf"
if [ ! -f "$out2" ]; then   # the n-gram table of another quantization: the same bytes, linked instead of fetched
  for o in "$MODEL_DIR"/Qwen3.8-Flash-Next-GSQ-RCO-*-00002-of-00002.gguf; do
    if [ -f "$o" ] && [ "$(stat -L -c %s "$o")" = "${S2%% *}" ]; then echo "== $(basename "$out2"): linked to $o"; ln "$o" "$out2"; break; fi
  done
fi
for f in "${FILES[@]}"; do
  read -r name size sha <<<"$f"
  out="$MODEL_DIR/$name"
  have=$( [ -f "$out" ] && stat -L -c %s "$out" || echo 0 )
  if [ "$have" != "$size" ]; then
    echo "== $name ($((size / 1000000000)) GB)"
    curl -fL --retry 10 --retry-delay 5 -C - -o "$out" "$ENDPOINT/$REPO/resolve/$REV/$QUANT/$name"
  fi
  [ "$(stat -L -c %s "$out")" = "$size" ] || { echo "$out: wrong size; delete it and run again"; exit 1; }
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
