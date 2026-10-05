# syntax=docker/dockerfile:1
# Official vLLM 0.30.0, pinned linux/amd64 manifest. Model weights are mounted.
FROM vllm/vllm-openai@sha256:5f5e535216848d0c52159c8c13a0af04be5f6fe1a84e79914300610796f76d40

USER root
WORKDIR /app
COPY requirements-api.txt /app/requirements-api.txt
RUN python3 -m pip install --no-cache-dir --no-deps -r /app/requirements-api.txt \
    && python3 -c "import fastapi,uvicorn,pydantic,vllm,torch; print(fastapi.__version__,uvicorn.__version__,pydantic.__version__,vllm.__version__,torch.__version__)"

COPY app.py config.json example-request.json healthcheck.py /app/
ENV CLEF_MODEL_PATH=/models/clef-flash \
    CLEF_HOST=0.0.0.0 \
    CLEF_PORT=8000 \
    CLEF_MAX_MODEL_LEN=4096 \
    CLEF_KV_CACHE_GIB=0.5 \
    CLEF_MAX_PENDING_REQUESTS=16 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    CUDNN_FRONTEND_CUDART_LIB_NAME=libcudart.so.13 \
    PYTHONUNBUFFERED=1

LABEL org.opencontainers.image.title="clef-flash-api" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.description="Local Clef Flash NVFP4 structured decision API; mount model at /models/clef-flash"
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=900s --retries=3 \
    CMD ["python3", "/app/healthcheck.py"]
STOPSIGNAL SIGTERM
ENTRYPOINT ["python3", "-u", "/app/app.py"]
