import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib import error, request


SYSTEM_PROMPT = (
    "你是中文学术数据标注助手。请根据 context、response 和给定候选节点，"
    "对候选节点进行上下文条件下的重要性重排序。"
)

INSTRUCTION = """要求：
1. 只在给定候选节点内部排序；
2. 基础中心性是先验，不是最终顺序；
3. 若 shift_label=1，优先考虑与新话题更相关的节点；
4. 若 shift_label=0，优先考虑与上下文延续更强的节点；
5. 只输出 JSON；
6. 所有候选节点必须全部出现且不重复。

输出格式：
{
  "reranked_nodes": [...],
  "scores": [{"node": "...", "score": 95}],
  "top_k": [...],
  "reason": "..."
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read candidate JSON -> call local Ollama/OpenAI-compatible API -> save reranked annotation JSON."
    )
    parser.add_argument("--input_json", type=str, default="demo/tiage-1/step1_candidates_top10.json")
    parser.add_argument("--output_json", type=str, default="demo/tiage-1/step2_reranked_annotation.json")

    # Default to local Ollama.
    parser.add_argument("--base_url", type=str, default="http://localhost:11434/v1")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--api_key", type=str, default="ollama")
    parser.add_argument(
        "--auto_pick_model",
        action="store_true",
        help="Auto-pick the first available local Ollama model when --model is empty.",
    )

    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep_sec", type=float, default=0.15)
    parser.add_argument("--with_relevance", action="store_true")
    parser.add_argument("--with_shift_label", action="store_true")
    parser.add_argument("--response_format_json", action="store_true")
    return parser.parse_args()


def _http_post_json(url: str, headers: Dict[str, str], data: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    req = request.Request(
        url=url,
        headers=headers,
        data=json.dumps(data, ensure_ascii=False).encode("utf-8"),
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def _http_get_json(url: str, headers: Dict[str, str], timeout: int) -> Dict[str, Any]:
    req = request.Request(url=url, headers=headers, method="GET")
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def detect_local_ollama_model(base_url: str, timeout: int) -> str:
    # 1) Try OpenAI-compatible models API.
    openai_models_url = f"{base_url.rstrip('/')}/models"
    try:
        data = _http_get_json(openai_models_url, headers={}, timeout=timeout)
        items = data.get("data", [])
        if items:
            return str(items[0].get("id", "")).strip()
    except Exception:
        pass

    # 2) Fallback to native Ollama tags API.
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    tags_url = f"{base}/api/tags"
    data = _http_get_json(tags_url, headers={}, timeout=timeout)
    models = data.get("models", [])
    if not models:
        raise RuntimeError("No local Ollama models found. Please run `ollama pull <model>` first.")
    return str(models[0].get("name", "")).strip()


def _extract_json_str(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()
    return text


def _canonical_node(x: Any) -> str:
    return str(x)


def build_prompt(sample: dict, with_relevance: bool, with_shift_label: bool, top_n: int) -> str:
    candidates = []
    for c in sample["candidates_topk"]:
        item = {
            "node_id": c["node_id"],
            "centrality": c.get("centrality", 0.0),
            "text": c.get("text", ""),
        }
        if with_relevance:
            item["relevance"] = c.get("relevance", 0.0)
        candidates.append(item)

    payload = {
        "context": sample.get("context", []),
        "response": sample.get("response", ""),
        "candidates": candidates,
        "top_n": top_n,
    }
    if with_shift_label:
        payload["shift_label"] = sample.get("shift_label", -1)

    return f"{INSTRUCTION}\n请处理下列样本：\n{json.dumps(payload, ensure_ascii=False, indent=2)}"


def call_chat_completions(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    timeout: int,
    max_retries: int,
    response_format_json: bool,
) -> dict:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    if response_format_json:
        payload["response_format"] = {"type": "json_object"}

    last_err = None
    for i in range(max_retries):
        try:
            return _http_post_json(url, headers, payload, timeout=timeout)
        except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            if i < max_retries - 1:
                time.sleep(1.5 ** i)
    raise RuntimeError(f"API request failed after {max_retries} retries: {last_err}")


def normalize_and_validate(model_out: dict, candidate_nodes: List[str], top_n: int) -> Tuple[dict, bool, str]:
    expected = candidate_nodes[:]
    expected_set = set(expected)

    reranked = model_out.get("reranked_nodes", [])
    reranked = [_canonical_node(x) for x in reranked if _canonical_node(x) in expected_set]

    seen = set()
    uniq = []
    for n in reranked:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    reranked = uniq

    score_map: Dict[str, float] = {}
    for x in model_out.get("scores", []):
        if not isinstance(x, dict):
            continue
        node = _canonical_node(x.get("node"))
        if node not in expected_set:
            continue
        try:
            score_map[node] = float(x.get("score"))
        except (TypeError, ValueError):
            score_map[node] = 0.0

    if len(reranked) != len(expected):
        if len(score_map) == len(expected):
            reranked = sorted(expected, key=lambda n: score_map[n], reverse=True)
        else:
            reranked = reranked + [n for n in expected if n not in reranked]

    if len(score_map) < len(expected):
        base = float(len(expected) * 10)
        for i, n in enumerate(reranked):
            score_map.setdefault(n, base - i)

    normalized = {
        "reranked_nodes": reranked,
        "scores": [{"node": n, "score": score_map[n]} for n in reranked],
        "top_k": reranked[: max(1, min(top_n, len(reranked)))],
        "reason": str(model_out.get("reason", "")).strip(),
    }
    ok = set(reranked) == expected_set and len(reranked) == len(expected)
    return normalized, ok, ("ok" if ok else "recovered")


def rerank_one(sample: dict, args: argparse.Namespace, model: str) -> dict:
    candidate_nodes = [str(c["node_id"]) for c in sample["candidates_topk"]]
    prompt = build_prompt(sample, args.with_relevance, args.with_shift_label, args.top_n)
    raw_resp = call_chat_completions(
        base_url=args.base_url,
        api_key=args.api_key,
        model=model,
        prompt=prompt,
        temperature=args.temperature,
        timeout=args.timeout,
        max_retries=args.max_retries,
        response_format_json=args.response_format_json,
    )
    content = raw_resp["choices"][0]["message"]["content"]
    model_out = json.loads(_extract_json_str(content))
    norm, valid, status = normalize_and_validate(model_out, candidate_nodes, args.top_n)

    out = dict(sample)
    out["llm_rerank"] = norm
    out["llm_meta"] = {"valid_directly": valid, "status": status, "model": model}
    return out


def main() -> None:
    args = parse_args()

    model = args.model.strip()
    if not model:
        if not args.auto_pick_model:
            raise ValueError("Missing --model. Use --auto_pick_model to auto-detect local Ollama model.")
        model = detect_local_ollama_model(args.base_url, args.timeout)
        print(f"[INFO] auto-picked model: {model}")

    input_path = Path(args.input_json).resolve()
    output_path = Path(args.output_json).resolve()

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    output = {
        "config": {
            "source_input": str(input_path),
            "base_url": args.base_url,
            "model": model,
            "top_n": args.top_n,
            "temperature": args.temperature,
            "with_relevance": bool(args.with_relevance),
            "with_shift_label": bool(args.with_shift_label),
            "response_format_json": bool(args.response_format_json),
            "generated_at": int(time.time()),
        },
        "splits": {"train": [], "dev": [], "test": []},
    }

    for split in ("train", "dev", "test"):
        samples = data.get("splits", {}).get(split, [])
        for i, s in enumerate(samples, start=1):
            try:
                out = rerank_one(s, args, model)
                output["splits"][split].append(out)
            except Exception as e:
                failed = dict(s)
                failed["llm_rerank_error"] = str(e)
                output["splits"][split].append(failed)
            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)
            print(f"[{split}] {i}/{len(samples)} done")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] {output_path}")


if __name__ == "__main__":
    main()
