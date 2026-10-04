#!/usr/bin/env python
# BSD 3-Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Gemma-4-12B-it 服务级 HTTP 探测：确定性 smoke / 同口径快测 / 并发稳态。

只用标准库（urllib + threading），不引入新依赖。

用法:
  python http_probe.py --base http://127.0.0.1:8000 --mode smoke
  python http_probe.py --mode bench        # 3 次同口径 + TTFT
  python http_probe.py --mode concurrency  # 1/4/8 并发稳态
  python http_probe.py --mode all --out /path/evidence.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.request

DEFAULT_PROMPT = "Explain in one sentence why the sky appears blue."
SMOKE_TOKENS = 32


def post(url: str, payload: dict, timeout: float = 600.0) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def get(url: str, timeout: float = 30.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode()


def completions(base: str, prompt: str, max_tokens: int, temperature: float = 0.0) -> dict:
    return post(
        f"{base}/v1/completions",
        {
            "model": "gemma-4-12B-it",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1.0,
            "top_k": -1,
            "seed": 0,
            # vLLM 扩展：让非流式响应也带回 token_ids，否则无法做逐位确定性比对
            "return_token_ids": True,
        },
    )


def ttft_stream(base: str, prompt: str, max_tokens: int) -> tuple[float, int]:
    """流式取首 token 延迟（TTFT）与总 token 数。"""
    import urllib.request as u

    payload = json.dumps({
        "model": "gemma-4-12B-it", "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "top_p": 1.0, "top_k": -1, "seed": 0,
        "stream": True, "return_token_ids": True,
    }).encode()
    req = u.Request(f"{base}/v1/completions", data=payload,
                    headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    ntok = 0
    with u.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                continue
            for ch in obj.get("choices", []):
                piece = ch.get("text")
                if piece:
                    ntok += 1
                    if first is None:
                        first = time.perf_counter() - t0
    total = time.perf_counter() - t0
    return (first if first is not None else float("nan")), ntok, total


def mode_smoke(base: str, prompt: str) -> dict:
    out = {"runs": []}
    for i in range(2):
        t = time.perf_counter()
        r = completions(base, prompt, SMOKE_TOKENS)
        dt = time.perf_counter() - t
        ch = r["choices"][0]
        # vLLM 返回 token_ids（非流式）；若没有则退回用 text
        toks = ch.get("token_ids")
        out["runs"].append({
            "run": i, "token_ids": toks, "text": ch.get("text"),
            "finish_reason": ch.get("finish_reason"),
            "num_tokens": len(toks) if toks else None,
            "seconds": round(dt, 4),
            "usage": r.get("usage"),
        })
        print(f"[smoke] run{i} finish={ch.get('finish_reason')} "
              f"n={len(toks) if toks else '?'} secs={dt:.4f}")
        if toks:
            print(f"[smoke] run{i} token_ids={toks}")
    a, b = out["runs"][0].get("token_ids"), out["runs"][1].get("token_ids")
    out["identical"] = bool(a) and a == b
    if not a:
        out["identical_note"] = (
            "响应未包含 token_ids（需 return_token_ids=true）；"
            "identical=False 属判据缺失，不代表模型不确定"
        )
    print(f"[smoke] identical={out['identical']}"
          + ("  (token_ids 缺失，判据不成立)" if not a else ""))
    return out


def mode_bench(base: str, prompt: str, repeats: int = 3) -> dict:
    rows = []
    for i in range(repeats):
        ttft, ntok, total = ttft_stream(base, prompt, SMOKE_TOKENS)
        rows.append({"run": i, "ttft_s": round(ttft, 4), "total_s": round(total, 4),
                     "output_tokens": ntok,
                     "tok_per_s": round(ntok / total, 3) if total else None})
        print(f"[bench] run{i} ttft={ttft:.4f}s total={total:.4f}s tok/s={ntok/total:.3f}")
    return {
        "rows": rows,
        "median_ttft_s": round(statistics.median(r["ttft_s"] for r in rows), 4),
        "median_total_s": round(statistics.median(r["total_s"] for r in rows), 4),
        "median_tok_per_s": round(statistics.median(r["tok_per_s"] for r in rows), 3),
    }


def mode_concurrency(base: str, prompt: str, levels=(1, 4, 8), tokens: int = 32) -> dict:
    res = {}
    for n in levels:
        results: list[dict] = []
        lock = threading.Lock()
        barrier = threading.Barrier(n)

        def worker(idx: int) -> None:
            barrier.wait()
            t = time.perf_counter()
            try:
                r = completions(base, prompt, tokens)
                dt = time.perf_counter() - t
                ch = r["choices"][0]
                toks = ch.get("token_ids") or []
                n_tok = len(toks) if toks else int(
                    (r.get("usage") or {}).get("completion_tokens") or 0)
                with lock:
                    results.append({"idx": idx, "seconds": round(dt, 4),
                                    "output_tokens": n_tok,
                                    "token_ids": toks,
                                    "ok": True})
            except Exception as e:  # noqa: BLE001
                with lock:
                    results.append({"idx": idx, "ok": False,
                                    "error": f"{type(e).__name__}: {e}"})

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        wall0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - wall0
        ok = [r for r in results if r["ok"]]
        tot_tok = sum(r["output_tokens"] for r in ok)
        res[str(n)] = {
            "concurrency": n, "wall_s": round(wall, 4),
            "ok": len(ok), "failed": len(results) - len(ok),
            "total_output_tokens": tot_tok,
            "aggregate_tok_per_s": round(tot_tok / wall, 3) if wall else None,
            "per_request": sorted(results, key=lambda r: r["idx"]),
            "errors": [r.get("error") for r in results if not r["ok"]],
        }
        print(f"[conc] n={n} wall={wall:.3f}s ok={len(ok)}/{n} "
              f"tok/s={tot_tok/wall:.3f}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--mode", default="all",
                    choices=["smoke", "bench", "concurrency", "all"])
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out: dict = {"base": args.base, "prompt": args.prompt, "modes": {}}
    try:
        out["models"] = get(f"{args.base}/v1/models")
    except Exception as e:  # noqa: BLE001
        out["models_error"] = f"{type(e).__name__}: {e}"

    if args.mode in ("smoke", "all"):
        out["modes"]["smoke"] = mode_smoke(args.base, args.prompt)
    if args.mode in ("bench", "all"):
        out["modes"]["bench"] = mode_bench(args.base, args.prompt)
    if args.mode in ("concurrency", "all"):
        out["modes"]["concurrency"] = mode_concurrency(args.base, args.prompt)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[http_probe] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
