"""Resident, text/JSON Clef NVFP4 API. One GPU worker; no input truncation."""
from __future__ import annotations

import asyncio
import importlib.metadata
import json
import logging
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
for _key, _name, _convert in (
    ("model_path", "CLEF_MODEL_PATH", str), ("host", "CLEF_HOST", str),
    ("port", "CLEF_PORT", int), ("max_model_len", "CLEF_MAX_MODEL_LEN", int),
    ("kv_cache_gib", "CLEF_KV_CACHE_GIB", float),
    ("max_pending_requests", "CLEF_MAX_PENDING_REQUESTS", int),
    ("max_body_bytes", "CLEF_MAX_BODY_BYTES", int),
):
    if _name in os.environ:
        CONFIG[_key] = _convert(os.environ[_name])
if not (1 <= CONFIG["port"] <= 65535 and 256 <= CONFIG["max_model_len"] <= 16384
        and CONFIG["kv_cache_gib"] > 0 and CONFIG["max_pending_requests"] >= 1
        and CONFIG["max_body_bytes"] > 0):
    raise ValueError("Invalid Clef service configuration.")
MODEL_NAME = "clef-flash"
VERSION = "0.1.0"
LOG = logging.getLogger("clef-api")


class ChoiceQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["choice"]
    instructions: str | None = Field(default=None, min_length=1, max_length=12000)
    criteria: dict[str, str] = Field(min_length=2, max_length=64)

    @model_validator(mode="after")
    def validate_options(self):
        if any(not key.strip() or len(key) > 128 or not value.strip()
               or len(value) > 12000 for key, value in self.criteria.items()):
            raise ValueError("Option IDs and descriptions must be nonempty and bounded.")
        return self


class NoulQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["noul"]
    instructions: str | None = Field(default=None, min_length=1, max_length=12000)
    criteria: dict[str, str] | None = None

    @model_validator(mode="after")
    def validate_options(self):
        if self.criteria is not None and (
            set(self.criteria) - {"true", "false"}
            or any(not v.strip() or len(v) > 12000 for v in self.criteria.values())
        ):
            raise ValueError("noul criteria can only customize true and false.")
        return self


class ScoreQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["score"]
    instructions: str | None = Field(default=None, min_length=1, max_length=12000)
    criteria: list[str] = Field(min_length=2, max_length=64)

    @model_validator(mode="after")
    def validate_options(self):
        if any(not v.strip() or len(v) > 12000 for v in self.criteria):
            raise ValueError("Score levels must have nonempty, bounded descriptions.")
        return self


Question = Annotated[ChoiceQuestion | NoulQuestion | ScoreQuestion, Field(discriminator="type")]


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "model": MODEL_NAME,
        "state": "小明有3个苹果，又买了2个苹果。",
        "questions": {"answer": {"type": "choice", "instructions": "现在共有几个苹果？",
                                  "criteria": {"three": "3个", "five": "5个", "six": "6个"}}}
    }})
    model: Literal["clef-flash", "clef-flash-NVFP4"] = MODEL_NAME
    state: JsonValue
    questions: dict[str, Question] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_question_ids(self):
        if any(not q.strip() or len(q) > 128 for q in self.questions):
            raise ValueError("Question IDs must be nonempty and at most 128 characters.")
        if self.state is None:
            raise ValueError("state must be text or a JSON value, not null.")
        return self


class Runtime:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clef-gpu")
        self.lock = asyncio.Lock()
        self.pending = 0
        self.completed = 0
        self.failed = 0
        self.ready = False
        self.created = time.perf_counter()
        self.model = None
        self.load_seconds = None
        self.warmup_seconds = None
        self.gpu = {}

    def load(self):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["MAX_JOBS"] = "4"
        os.environ["CUDNN_FRONTEND_CUDART_LIB_NAME"] = "libcudart.so.13"
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        os.environ.pop("PYTORCH_ALLOC_CONF", None)
        try:
            dist = importlib.metadata.distribution("nvidia-cuda-nvcc")
            nvcc = next(dist.locate_file(p) for p in dist.files if str(p).endswith("/nvcc"))
        except (importlib.metadata.PackageNotFoundError, StopIteration):
            import shutil
            nvcc_path = shutil.which("nvcc")
            if not nvcc_path:
                raise RuntimeError("CUDA compiler nvcc is missing from the image.") from None
            nvcc = Path(nvcc_path).resolve()
        cuda_home = nvcc.parent.parent
        os.environ["CUDA_HOME"] = str(cuda_home)
        os.environ["CUDA_PATH"] = str(cuda_home)
        os.environ["CUDACXX"] = str(nvcc)
        os.environ["PATH"] = (str(Path(sys.executable).parent) + os.pathsep + str(nvcc.parent)
                              + os.pathsep + os.environ["PATH"])
        os.environ["LD_LIBRARY_PATH"] = str(cuda_home / "lib") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        model_path = Path(CONFIG["model_path"])
        if not (model_path / "joint_head.safetensors").is_file():
            raise RuntimeError(f"Missing local Clef model/head: {model_path}")
        os.environ["CLEF_MODEL_PATH"] = str(model_path)
        sys.path.insert(0, str(model_path))
        import torch
        from clef_vllm import ClefVLLM
        from joint_schema_model import encode_record, systemone_answer

        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU available. Start the container with --gpus all.")
        capability = torch.cuda.get_device_capability()
        if capability[0] < 10:
            raise RuntimeError("This NVFP4 checkpoint requires a Blackwell-class NVIDIA GPU.")
        self.torch = torch
        self.encode_record = encode_record
        self.systemone_answer = systemone_answer
        start = time.perf_counter()
        self.model = ClefVLLM(
            model_path, max_model_len=CONFIG["max_model_len"],
            kv_cache_gib=CONFIG["kv_cache_gib"], max_images=0, max_videos=0,
            max_num_seqs=4, max_num_batched_tokens=CONFIG["max_model_len"],
            language_model_only=True,
            kernel_config={"linear_backend_per_quant": {"nvfp4_w4a4": "flashinfer_cudnn"},
                           "enable_flashinfer_autotune": False},
        )
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - start
        warm = json.loads((HERE / "example-request.json").read_text(encoding="utf-8"))
        start = time.perf_counter()
        for _ in range(3):
            self.infer(warm, time.perf_counter())
        self.warmup_seconds = time.perf_counter() - start
        self.gpu = {"name": torch.cuda.get_device_name(),
                    "compute_capability": list(capability),
                    "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
                    "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 3)}
        LOG.info("Model ready: load=%.2fs warmup=%.2fs gpu=%s", self.load_seconds,
                 self.warmup_seconds, self.gpu)

    def infer(self, payload: dict, submitted: float):
        started = time.perf_counter()
        # The bundled encoder truncates state at max_length. Encode fully instead,
        # reject overflow, and pass the exact encoded request to the joint head.
        encoded = self.encode_record(self.model.tokenizer, payload, max_length=sys.maxsize,
                                     processor=self.model.processor)
        tokens = len(encoded.input_ids)
        if tokens > CONFIG["max_model_len"]:
            raise HTTPException(422, detail={"code": "context_length_exceeded",
                "input_tokens": tokens, "max_input_tokens": CONFIG["max_model_len"],
                "message": "Input was rejected, not truncated. Split the material into smaller units."})
        self.torch.cuda.synchronize()
        hidden = self.model.hidden_states([payload], [encoded])[0]
        probabilities = self.model.decide(encoded, hidden)
        self.torch.cuda.synchronize()
        answers = {qid: self.systemone_answer(payload["questions"][qid], p)
                   for qid, p in probabilities.items()}
        finished = time.perf_counter()
        return {"model": payload["model"], "answers": answers,
                "usage": {"input_tokens": tokens, "output_tokens": 0},
                "timing": {"queue_ms": round((started - submitted) * 1000, 3),
                           "inference_ms": round((finished - started) * 1000, 3),
                           "total_ms": round((finished - submitted) * 1000, 3)}}


runtime = Runtime()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.get_running_loop().run_in_executor(runtime.executor, runtime.load)
    runtime.ready = True
    try:
        yield
    finally:
        runtime.ready = False
        runtime.executor.shutdown(wait=True, cancel_futures=True)
        runtime.model = None


app = FastAPI(title="Clef Flash NVFP4 本地判断 API", version=VERSION, lifespan=lifespan,
              description="常驻并预热的结构化判断服务。支持文字和 JSON；输入超长时拒绝，保留原文完整性。")


@app.middleware("http")
async def request_metadata(request: Request, call_next):
    request_id = uuid.uuid4().hex
    started = time.perf_counter()
    # Read incrementally so an oversized request is rejected before buffering it.
    if request.method in {"POST", "PUT", "PATCH"}:
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > CONFIG["max_body_bytes"]:
                return JSONResponse(status_code=413, content={"detail": f"Request body exceeds {CONFIG['max_body_bytes']} bytes."},
                                    headers={"X-Request-ID": request_id})
        request._body = bytes(body)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Elapsed-Ms"] = f"{(time.perf_counter() - started) * 1000:.3f}"
    return response


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse("/docs")


@app.get("/health")
async def health(response: Response):
    if not runtime.ready:
        response.status_code = 503
    return {"status": "ok" if runtime.ready else "unavailable", "ready": runtime.ready,
            "service": "clef-flash-api", "version": VERSION, "model": MODEL_NAME,
            "max_input_tokens": CONFIG["max_model_len"], "supports": ["text", "json"],
            "question_types": ["choice", "noul", "score"], "gpu_workers": 1,
            "pending_requests": runtime.pending, "max_pending_requests": CONFIG["max_pending_requests"],
            "completed_requests": runtime.completed, "failed_requests": runtime.failed,
            "uptime_seconds": round(time.perf_counter() - runtime.created, 2),
            "model_load_seconds": runtime.load_seconds, "warmup_seconds": runtime.warmup_seconds,
            "gpu": runtime.gpu, "nvfp4_backend": "flashinfer_cudnn"}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model",
            "owned_by": "local", "capabilities": ["systemone", "choice", "noul", "score"]}]}


@app.post("/v1/systemone")
async def systemone(body: DecisionRequest, response: Response):
    if not runtime.ready:
        raise HTTPException(503, "Model is not ready.")
    if runtime.pending >= CONFIG["max_pending_requests"]:
        raise HTTPException(429, "Inference queue is full.", headers={"Retry-After": "1"})
    submitted = time.perf_counter()
    runtime.pending += 1
    try:
        async with runtime.lock:
            future = asyncio.get_running_loop().run_in_executor(
                runtime.executor, runtime.infer, body.model_dump(exclude_none=True), submitted)
            try:
                result = await asyncio.shield(future)
            except asyncio.CancelledError:
                # Finish GPU work before releasing the inference lock.
                await asyncio.shield(future)
                raise
            runtime.completed += 1
            response.headers["Server-Timing"] = (
                f"queue;dur={result['timing']['queue_ms']},infer;dur={result['timing']['inference_ms']}")
            return result
    except HTTPException:
        runtime.failed += 1
        raise
    except Exception:
        runtime.failed += 1
        LOG.exception("Inference failed")
        raise HTTPException(500, "Inference failed; inspect service.log.") from None
    finally:
        runtime.pending -= 1


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(app, host=CONFIG["host"], port=CONFIG["port"], workers=1,
                timeout_keep_alive=30, timeout_graceful_shutdown=30, access_log=True)
