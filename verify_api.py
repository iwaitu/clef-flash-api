"""HTTP acceptance test and timings. Standard library; no GPU/model imports."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import math
import os
import statistics
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
BASE = os.environ.get("CLEF_API_BASE_URL", f"http://127.0.0.1:{CONFIG['port']}").rstrip("/")
BASELINE = json.loads((HERE / "test-fixtures.json").read_text(encoding="utf-8-sig"))


def call(path, body=None):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urlopen(req, timeout=120) as r:
            payload = r.read()
            elapsed = (time.perf_counter() - start) * 1000
            content_type = r.headers.get("Content-Type", "")
            value = json.loads(payload) if "json" in content_type else payload.decode()
            return r.status, value, elapsed, dict(r.headers)
    except HTTPError as e:
        return e.code, json.loads(e.read()), (time.perf_counter() - start) * 1000, dict(e.headers)


def percentile(values, p):
    values = sorted(values)
    at = (len(values) - 1) * p
    lo = int(at)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (at - lo)


def predictions(body):
    return {k: (v["choice"] if v["type"] == "choice" else v["noul"] >= .5)
            for k, v in body["answers"].items()}


if __name__ == "__main__":
    status, health, _, _ = call("/health")
    assert status == 200 and health["ready"] and health["gpu_workers"] == 1, health
    assert call("/docs")[0] == 200
    spec = call("/openapi.json")[1]
    assert "/v1/systemone" in spec["paths"]
    assert call("/v1/models")[1]["data"][0]["id"] == "clef-flash"
    cases = []
    request_ids = set()
    probability_max_delta = 0
    probability_parity_tolerance = .001
    for case in BASELINE["cases"]:
        for _ in range(3):
            assert call("/v1/systemone", case["request"])[0] == 200
        times, infer_times, queue_times = [], [], []
        case_probability_max_delta = 0
        for _ in range(30):
            status, body, elapsed, headers = call("/v1/systemone", case["request"])
            assert status == 200, body
            assert predictions(body) == case["expected"], (case["name"], body)
            assert body["usage"]["input_tokens"] == case["input_tokens"]
            assert body["usage"]["output_tokens"] == 0
            for key, answer in body["answers"].items():
                previous = case["answer"][key]
                if answer["type"] == "choice":
                    assert abs(sum(answer["probabilities"].values()) - 1) < .001
                    delta = max(abs(p - previous["probabilities"][o])
                                for o, p in answer["probabilities"].items())
                else:
                    delta = abs(answer["noul"] - previous["noul"])
                probability_max_delta = max(probability_max_delta, delta)
                case_probability_max_delta = max(case_probability_max_delta, delta)
            request_id = headers.get("X-Request-ID") or headers.get("x-request-id")
            assert request_id and request_id not in request_ids
            request_ids.add(request_id)
            assert headers.get("Server-Timing") or headers.get("server-timing")
            times.append(elapsed)
            infer_times.append(body["timing"]["inference_ms"])
            queue_times.append(body["timing"]["queue_ms"])
        result = {"name": case["name"], "correct": True, "requests": 30,
                  "http_ms": {"p50": statistics.median(times), "p95": percentile(times, .95),
                              "min": min(times), "max": max(times)},
                  "inference_ms": {"p50": statistics.median(infer_times), "p95": percentile(infer_times, .95)},
                  "queue_ms_p50": statistics.median(queue_times), "answer": body["answers"],
                  "baseline_max_probability_delta": case_probability_max_delta}
        cases.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    example = json.loads((HERE / "example-request.json").read_text(encoding="utf-8"))
    status, three_types, _, _ = call("/v1/systemone", example)
    assert status == 200 and three_types["answers"]["answer"]["choice"] == "five"
    assert three_types["answers"]["more_than_four"]["noul"] > .5
    score = three_types["answers"]["quantity"]
    assert score["type"] == "score" and 0 <= score["score"] <= 2
    assert set(score["probabilities"]) == {"0", "1", "2"}
    assert all(math.isfinite(v) for v in score["probabilities"].values())

    invalid = [{}, {**example, "model": "not-clef"}, {**example, "questions": {}},
               {**example, "images": ["not-supported"]},
               {**example, "questions": {"q": {"type": "choice", "criteria": ["a", "b"]}}},
               {**example, "questions": {"q": {"type": "noul", "criteria": {"unknown": "missing"}}}},
               {**example, "questions": {"q": {"type": "score", "criteria": []}}},
               {**example, "state": None}]
    for bad in invalid:
        assert call("/v1/systemone", bad)[0] == 422
    too_long = {**example, "state": " evidence " * 15000}
    status, overflow, _, _ = call("/v1/systemone", too_long)
    assert status == 422 and overflow["detail"]["code"] == "context_length_exceeded", overflow
    assert call("/v1/systemone", {**example, "state": "x" * 300000})[0] == 413

    def concurrent_job(i):
        request = copy.deepcopy(BASELINE["cases"][i % len(BASELINE["cases"])] ["request"])
        status, body, elapsed, _ = call("/v1/systemone", request)
        assert status == 200, body
        assert predictions(body) == BASELINE["cases"][i % len(BASELINE["cases"])]["expected"]
        return {"http_ms": elapsed, **body["timing"]}

    with ThreadPoolExecutor(max_workers=8) as pool:
        concurrent = list(pool.map(concurrent_job, range(8)))
    assert call("/health")[1]["pending_requests"] == 0
    report = {"base_url": BASE, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "client": "Python urllib HTTP, new connection per request",
              "health": health, "cases": cases, "three_types": three_types,
              "validation": {"sequential_requests": 240, "correct_cases": len(cases),
                             "baseline_max_probability_delta": probability_max_delta,
                             "baseline_probability_parity_tolerance": probability_parity_tolerance,
                             "baseline_probability_parity": probability_max_delta <= probability_parity_tolerance,
                             "invalid_requests_rejected": len(invalid),
                             "overflow_rejected_without_truncation": True, "oversized_body_rejected": True,
                             "concurrent_requests_correct": len(concurrent), "docs_and_schema": True},
              "concurrency_8": concurrent,
              "median_of_case_http_medians_ms": statistics.median(c["http_ms"]["p50"] for c in cases)}
    report["functional_result"] = "PASS"
    report["probability_comparison"] = "MATCH" if report["validation"]["baseline_probability_parity"] else "DIFFERENT"
    (HERE / "api-test-results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"functional_result": "PASS",
                      "probability_comparison": "MATCH" if report["validation"]["baseline_probability_parity"] else "DIFFERENT",
                      "validation": report["validation"],
                      "http_p50_ms": report["median_of_case_http_medians_ms"]}, ensure_ascii=False))
