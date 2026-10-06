#!/usr/bin/env python3
"""Build the decode benchmark corpus (bench/corpus/*.tokens): seven short inputs of different kinds, 1800 tokens at
most each. Two are chat turns written here; five are the start of existing texts, read from the paths given:

    code_llama  llama.cpp src/models/qwen4exp.cpp at 3cf0325                     (MIT)
    code_cuda   this repository's src/cuda/window.cu                           (Apache-2.0)
    en_docs     llama.cpp docs/build.md at 3cf0325                             (MIT)
    zh_readme   nano-metal-moe-qwen36 README.zh-CN.md + docs/optimization-log.zh-CN.md   (Apache-2.0)
    mixed       nano-metal-moe-qwen36 scripts/eval/mixed.txt                   (Apache-2.0)

    python3 bench/make_corpus.py --gguf models/...-00001-of-00002.gguf --llama SRC --nano SRC

Uses build/bl-tok; Python standard library only.
"""
import argparse
import pathlib

from make_prompts import HERE, Tok

N = 1800
CHATS = {
    "chat_zh": "用中文详细解释一下什么是混合专家模型（MoE），它为什么能在消费级显卡上运行大模型？然后用 Python 写一个最简单的 top-k 路由示例。",
    "chat_en": "Write a C++ function that parses a CSV line with quoted fields, then explain the edge cases it handles.",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--llama", required=True, help="a llama.cpp source tree")
    ap.add_argument("--nano", required=True, help="a nano-metal-moe-qwen36 source tree")
    ap.add_argument("--out", default=str(HERE / "corpus"))
    ap.add_argument("--bl-tok", default=str(HERE.parent / "build" / "bl-tok"))
    a = ap.parse_args()
    tok, out = Tok(a.gguf, a.bl_tok), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    read = lambda root, *paths: "".join((pathlib.Path(root) / p).read_text(encoding="utf-8") for p in paths)

    def save(name, ids):
        (out / f"{name}.tokens").write_text(",".join(map(str, ids[:N])) + "\n")
        print(name, len(ids[:N]))

    save("code_llama", tok.encode(read(a.llama, "src/models/qwen4exp.cpp")))
    save("code_cuda", tok.encode(read(HERE.parent, "src/cuda/window.cu")))
    save("zh_readme", tok.encode(read(a.nano, "README.zh-CN.md", "docs/optimization-log.zh-CN.md")))
    save("en_docs", tok.encode(read(a.llama, "docs/build.md")))
    save("mixed", tok.encode(read(a.nano, "scripts/eval/mixed.txt")))
    for name, q in CHATS.items():
        save(name, tok.encode(tok.chat(q), special=True))


if __name__ == "__main__":
    main()
