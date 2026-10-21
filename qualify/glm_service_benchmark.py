"""Immutable, sequential GLM service baselines; Python standard library only.

Create a corpus once, run it unchanged after each adjustment, then compare.
No service settings, sampling defaults, thinking options or caches are changed.
Results retain token IDs for exact comparisons, never response text or API keys.
"""

import argparse
import base64
import hashlib
import http.client
import json
import math
import os
import statistics
import struct
import time
import zlib
from pathlib import Path
from urllib.parse import urlsplit


SCHEMA = "glm-service-benchmark-v1"


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def save(path, value):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def image():
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    rows = b"".join(b"\0" + bytes((220, 30, 30)) * 112 + bytes((30, 30, 220)) * 112 for _ in range(224))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 224, 224, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def corpus(args):
    prompts = [
        ("zh-existing-gate", "请从1开始依次列出100条简短的Python编程建议，每行一条，不要提前结束。", 64),
        ("en-existing-failure", "Write a Python implementation of merge sort, then explain the code line by line in detail.", 64),
        ("long-context", "Read these background notes, then write a detailed numbered list of Python programming advice.\n"
         + "Python programs should use meaningful variable names and clear control flow.\n" * 160, 128),
        ("image", [{"type": "text", "text": "Describe the colors and their positions in this image in detail."},
                   {"type": "image_url", "image_url": {"url": image()}}], 128),
    ]
    cases = [{"id": name, "request": {"model": args.model, "messages": [{"role": "user", "content": prompt}],
             "temperature": 0, "seed": 1234, "max_tokens": count, "stream": True,
             "return_token_ids": True, "return_token_timestamps": True,
             "stream_options": {"include_usage": True}}} for name, prompt, count in prompts]
    save(args.output, {"schema": SCHEMA, "cases": cases,
        "policy": "Existing default thinking; greedy diagnostic workload; one warmup per case before each measured repeat.",
        "qualification": "Full-model ordinary/DFlash2 exact-output gate remains OPEN; matching this corpus cannot clear it."})
    print(json.dumps({"corpus": str(args.output), "cases": len(cases), "sha256": digest(cases)}))


def connect(base, timeout=300):
    url = urlsplit(base)
    if url.scheme not in ("http", "https") or url.path not in ("", "/") or url.username or url.password:
        raise ValueError("base must be an HTTP(S) origin without credentials")
    kind = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    return kind(url.hostname, url.port, timeout=timeout)


def metrics(base):
    client = connect(base, timeout=30)
    try:
        client.request("GET", "/metrics")
        response = client.getresponse()
        if response.status != 200:
            raise RuntimeError(f"metrics HTTP {response.status}")
        result = {}
        for line in response.read().decode().splitlines():
            if line.startswith("flare_"):
                key, value = line.rsplit(" ", 1)
                result[key] = float(value)
        return result
    finally:
        client.close()


def idle(values):
    return values["flare_active_requests"] == values["flare_pending_requests"] == 0


def settled_metrics(base):
    # SSE DONE can precede the scheduler's request-retirement accounting.
    for _ in range(50):
        values = metrics(base)
        if idle(values):
            return values
        time.sleep(.1)
    raise RuntimeError("service did not become idle within five seconds")


def difference(before, after):
    keys = ("flare_requests_completed_total", "flare_requests_failed_total", "flare_generated_tokens_total",
            "flare_accepted_draft_tokens_total", "flare_draft_request_rounds_total", "flare_prefill_tokens_total",
            "flare_reused_prompt_tokens_total")
    return {key: value - before.get(key, 0) for key, value in after.items()
            if key in keys or key.startswith("flare_speculative_") and "_total{" in key}


def request(base, body, key):
    client = connect(base)
    tokens, core, arrivals, finishes, text, usage = [], [], [], [], [], None
    done, first_visible, reused = False, None, None
    try:
        client.connect()
        start = time.perf_counter_ns()
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        client.request("POST", "/v1/chat/completions", encoded(body), headers)
        response = client.getresponse()
        if response.status != 200:
            response.read()
            raise RuntimeError(f"generation HTTP {response.status}")
        while line := response.readline(1 << 20):
            if not line.startswith(b"data:"):
                continue
            elapsed = time.perf_counter_ns() - start
            payload = line[5:].strip()
            if payload == b"[DONE]":
                done = True
                break
            event = json.loads(payload)
            if "error" in event:
                raise RuntimeError("SSE error (response content omitted)")
            usage = event.get("usage") or usage
            reused = event.get("reused_prompt_tokens", reused)
            for choice in event.get("choices", []):
                if choice.get("index", 0) != 0:
                    raise ValueError("benchmark requires one choice")
                ids = choice.get("token_ids") or []
                tokens.extend(ids)
                arrivals.extend([elapsed] * len(ids))
                core.extend(choice.get("token_timestamps_ns") or [])
                delta = choice.get("delta") or {}
                visible = {k: v for k, v in delta.items() if k in ("content", "reasoning_content", "tool_calls") and v}
                if visible:
                    text.append(visible)
                    if first_visible is None:
                        first_visible = elapsed
                if choice.get("finish_reason"):
                    finishes.append(choice["finish_reason"])
        duration = (time.perf_counter_ns() - start) / 1e9
    finally:
        client.close()
    if not done or usage is None or not finishes:
        raise ValueError("incomplete SSE response")
    count = usage["completion_tokens"]
    if count != len(tokens) or count != len(core) or count < 2 or any(b < a for a, b in zip(core, core[1:])):
        raise ValueError("missing/inconsistent token IDs or native timestamps")
    if count != body["max_tokens"] or finishes != ["length"]:
        raise ValueError("completed work differs from the fixed output-token budget")
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens", reused)
    if cached is None:
        raise ValueError("missing cache accounting")
    return {"elapsed_s": duration, "ttft_ms": arrivals[0] / 1e6,
        "visible_ttft_ms": None if first_visible is None else first_visible / 1e6,
        "end_to_end_tokens_s": count / duration,
        "core_tokens_s": (count - 1) * 1e9 / (core[-1] - core[0]),
        "delivery_tokens_s": (count - 1) * 1e9 / (arrivals[-1] - arrivals[0]),
        "usage": usage, "cached_tokens": cached, "finish": finishes,
        "token_ids": tokens, "output_sha256": digest(tokens),
        "text_events_sha256": digest(text)}


def run(args):
    data = json.loads(args.corpus.read_text())
    if data["schema"] != SCHEMA or args.repeats < 1:
        raise ValueError("invalid corpus or repeat count")
    args.output.mkdir(parents=True, exist_ok=False)
    key = args.api_key_file.read_text().strip() if args.api_key_file else None
    before = settled_metrics(args.metrics_base or args.base)
    if not idle(before):
        raise RuntimeError("service busy; no benchmark requests sent")
    result = {"schema": SCHEMA, "label": args.label, "corpus_sha256": digest(data["cases"]),
              "base": args.base, "configuration": json.loads(args.configuration.read_text()) if args.configuration else {},
              "qualification": data["qualification"], "samples": [], "valid": False}
    completed = 0
    try:
        for repetition in range(args.repeats):
            for case in data["cases"]:
                for warmup in (True, False):
                    settled_metrics(args.metrics_base or args.base)
                    sample = request(args.base, case["request"], key)
                    completed += 1
                    sample.update(id=case["id"], repetition=repetition, warmup=warmup)
                    result["samples"].append(sample)
                    if not warmup:
                        print(json.dumps({"case": case["id"], "repeat": repetition,
                            "core_tokens_s": round(sample["core_tokens_s"], 3),
                            "ttft_ms": round(sample["ttft_ms"], 3)}), flush=True)
        after = settled_metrics(args.metrics_base or args.base)
        delta = difference(before, after)
        result.update(metrics_delta=delta, isolation={"idle_after": idle(after), "expected_requests": completed,
            "completed_requests": delta["flare_requests_completed_total"],
            "failed_requests": delta["flare_requests_failed_total"]})
        result["valid"] = (idle(after) and delta["flare_requests_completed_total"] == completed
                           and delta["flare_requests_failed_total"] == 0
                           and delta["flare_generated_tokens_total"] == sum(s["usage"]["completion_tokens"] for s in result["samples"]))
    except Exception as error:
        result["error"] = str(error)
        raise
    finally:
        save(args.output / "results.json", result)
    print(json.dumps(summary(result), ensure_ascii=False))
    if not result["valid"]:
        raise RuntimeError("contaminated baseline; result retained and excluded from valid comparisons")


def percentile(values, q):
    return sorted(values)[max(0, math.ceil(len(values) * q) - 1)]


def summary(result):
    samples = [s for s in result["samples"] if not s["warmup"]]
    cases = {}
    for name in dict.fromkeys(s["id"] for s in samples):
        group = [s for s in samples if s["id"] == name]
        cases[name] = {k: statistics.median([s[k] for s in group])
            for k in ("ttft_ms", "elapsed_s", "core_tokens_s", "end_to_end_tokens_s")}
        cases[name].update(p95_ttft_ms=percentile([s["ttft_ms"] for s in group], .95),
            distinct_outputs=len({s["output_sha256"] for s in group}))
    return {"label": result["label"], "valid": result["valid"], "cases": cases}


def compare(args):
    base, candidate = (json.loads(p.read_text()) for p in (args.baseline, args.candidate))
    if base["schema"] != candidate["schema"] or base["corpus_sha256"] != candidate["corpus_sha256"]:
        raise ValueError("corpus or result schema differs")
    a, b = ([s for s in data["samples"] if not s["warmup"]] for data in (base, candidate))
    if [(s["id"], s["repetition"]) for s in a] != [(s["id"], s["repetition"]) for s in b]:
        raise ValueError("workloads/repetitions differ")
    rows = []
    for old, new in zip(a, b):
        same_work = (old["usage"]["prompt_tokens"] == new["usage"]["prompt_tokens"]
            and old["usage"]["completion_tokens"] == new["usage"]["completion_tokens"]
            and old["cached_tokens"] == new["cached_tokens"] and old["finish"] == new["finish"])
        same_output = old["token_ids"] == new["token_ids"]
        rows.append({"id": old["id"], "repetition": old["repetition"], "matched_work_and_cache": same_work,
            "output_equal": same_output, "first_different_token": next((i for i,(x,y) in enumerate(zip(old["token_ids"],new["token_ids"])) if x!=y), None),
            "core_tokens_s_change_pct": 100 * (new["core_tokens_s"] / old["core_tokens_s"] - 1),
            "ttft_change_pct": 100 * (new["ttft_ms"] / old["ttft_ms"] - 1),
            "end_to_end_change_pct": 100 * (new["end_to_end_tokens_s"] / old["end_to_end_tokens_s"] - 1)})
    report = {"baseline": base["label"], "candidate": candidate["label"], "valid_isolation": base["valid"] and candidate["valid"],
        "all_work_and_cache_match": all(s["matched_work_and_cache"] for s in rows),
        "all_outputs_equal": all(s["output_equal"] for s in rows),
        "qualification": "Diagnostic timings only; existing ordinary/DFlash2 full-model exact-output qualification remains OPEN.",
        "baseline_summary": summary(base), "candidate_summary": summary(candidate), "pairs": rows}
    if args.output:
        save(args.output, report)
    print(json.dumps(report, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("corpus")
    p.add_argument("--model", default="spark/GLM-5.3-Flash-EXL3")
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(function=corpus)
    p = sub.add_parser("run")
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--base", default="http://127.0.0.1:9999")
    p.add_argument("--metrics-base")
    p.add_argument("--api-key-file", type=Path)
    p.add_argument("--configuration", type=Path)
    p.add_argument("--label", required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(function=run)
    p = sub.add_parser("compare")
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.set_defaults(function=compare)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
