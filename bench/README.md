# Benchmark inputs

Fixed token lists, so every speed and accuracy number in this repository can be measured again on the same input.
Both generators use the engine's own tokenizer and chat template (`build/bl-tok`) and rebuild these files byte for
byte from the sources below.

## `corpus/` - decode speed

Seven inputs of different kinds, at most 1800 tokens each (`bench/make_corpus.py`). Each is fed as a prompt, then
200 tokens are generated (`bl-run --tokens-file ... --gen 200 --spec 3`).

| File | Content | Source (license) |
|---|---|---|
| `chat_en.tokens` | A C++ question, as a chat turn | written here |
| `chat_zh.tokens` | A Chinese question about MoE models, as a chat turn | written here |
| `code_llama.tokens` | C++: llama.cpp's implementation of this model | [llama.cpp](https://github.com/ggml-org/llama.cpp) `src/models/qwen4exp.cpp` at `3cf0325` (MIT) |
| `code_cuda.tokens` | CUDA: this engine's decode kernels | `src/cuda/window.cu` of this repository |
| `en_docs.tokens` | English technical documentation | llama.cpp `docs/build.md` at `3cf0325` (MIT) |
| `zh_readme.tokens` | Chinese technical writing | [nano-metal-moe-qwen36](https://github.com/DaveByteAI/nano-metal-moe-qwen36) `README.zh-CN.md`, `docs/optimization-log.zh-CN.md` (Apache-2.0) |
| `mixed.tokens` | Mixed Chinese, English, code and JSON | nano-metal-moe-qwen36 `scripts/eval/mixed.txt` (Apache-2.0) |

## `prompts/` - prefill speed, long context, accuracy

`1k`, `4k`, `32k`, `128k`: one user turn of exactly that many tokens - a source tree's files, then a code review task
(`bench/make_prompts.py`). The committed prompts were built from llama.cpp's source tree at `3cf0325` (MIT); the
`.txt` files are the decoded text. `mixed128.tokens` and `short.tokens` are small accuracy probes. `chat_template.jinja` is the model's
chat template, as stored in the GGUF.

```bash
python3 bench/make_corpus.py --gguf models/Qwen3.8-...-00001-of-00002.gguf --llama SRC --nano SRC
python3 bench/make_prompts.py --gguf models/Qwen3.8-...-00001-of-00002.gguf --source SRC
```
