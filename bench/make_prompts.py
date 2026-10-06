#!/usr/bin/env python3
"""Build the long benchmark prompts (bench/prompts/{1k,4k,32k,128k}.tokens and .txt).

Each prompt is one user turn: "Here is part of the source tree of an LLM inference engine." + the source files of a
tree (sorted by path, .cpp/.cu/.hpp/.h/.cuh/.py under src, include, serve, tools) cut to exactly N tokens + a code
review task, in the model's chat template with thinking off. The committed prompts were built from llama.cpp's
source tree at 3cf0325 (MIT) - see bench/README.md; any tree works for new prompts.

    python3 bench/make_prompts.py --gguf models/...-00001-of-00002.gguf --source path/to/a/source/tree

Uses build/bl-tok (the engine's own tokenizer and chat template); Python standard library only.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
TARGETS = {"1k": 1000, "4k": 4000, "32k": 32000, "128k": 128000}
SUFFIXES = (".cpp", ".cu", ".hpp", ".h", ".cuh", ".py")
INTRO = "Here is part of the source tree of an LLM inference engine.\n\n"
TASK = ("\n\nTask: review the code above. Name the three most likely bugs or risks, each with the file, the "
        "function and a one-line fix.")
SENTINEL = "\x00DOC\x00"


class Tok:
    """build/bl-tok as a library: encode / decode / chat."""
    def __init__(self, gguf, exe):
        self.gguf, self.exe = gguf, exe

    def _run(self, mode, text, *extra):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as f:
            f.write(text)
        out = subprocess.run([self.exe, self.gguf, mode, f.name, *extra], check=True, capture_output=True).stdout
        pathlib.Path(f.name).unlink()
        return out

    def encode(self, text, special=False):
        line = self._run("--encode", text, *(["--special"] if special else [])).decode().strip()
        return [int(v) for v in line.split(",")] if line else []

    def decode(self, ids):
        return self._run("--decode", ",".join(map(str, ids))).decode("utf-8")

    def chat(self, user_text):
        return self._run("--chat", user_text).decode("utf-8")


def corpus(root):
    files = sorted(p for d in ("src", "include", "serve", "tools") for p in (root / d).rglob("*")
                   if p.is_file() and p.suffix in SUFFIXES)
    return "\n".join(f"=== {p.relative_to(root).as_posix()} ===\n{p.read_text(encoding='utf-8', errors='replace')}"
                     for p in files)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, help="shard 1 of the model (only its vocabulary is read)")
    ap.add_argument("--source", required=True, help="the source tree whose files fill the prompts")
    ap.add_argument("--out", default=str(HERE / "prompts"))
    ap.add_argument("--bl-tok", default=str(HERE.parent / "build" / "bl-tok"))
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = Tok(a.gguf, a.bl_tok)

    head, tail = tok.chat(INTRO + SENTINEL + TASK).split(SENTINEL)
    head_ids, tail_ids = tok.encode(head, special=True), tok.encode(tail, special=True)
    text = corpus(pathlib.Path(a.source))
    doc_ids = tok.encode(text)
    print(f"corpus: {len(text):,} chars, {len(doc_ids):,} tokens; wrapper {len(head_ids)} + {len(tail_ids)} tokens")

    tmpl = (HERE / "prompts" / "chat_template.jinja").read_text(encoding="utf-8")
    manifest = {"tokenizer": pathlib.Path(a.gguf).name, "template_sha256": hashlib.sha256(tmpl.encode()).hexdigest(),
                "head": head, "tail": tail, "prompts": {}}
    for name, n in TARGETS.items():
        k = n - len(head_ids) - len(tail_ids)
        if k > len(doc_ids):
            sys.exit(f"source too short for {name}: need {k} tokens, have {len(doc_ids)}")
        ids = head_ids + doc_ids[:k] + tail_ids
        (out / f"{name}.tokens").write_text(",".join(map(str, ids)) + "\n")
        (out / f"{name}.txt").write_text(tok.decode(ids), encoding="utf-8")
        manifest["prompts"][name] = {"tokens": n, "sha256": hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()}
        print(f"{name}: {n} tokens")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
