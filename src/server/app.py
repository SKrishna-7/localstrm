"""
FastAPI OpenAI-Compatible REST Server
-------------------------------------
Exposes OpenAI-compatible HTTP endpoints:
  - POST /v1/chat/completions (Streaming SSE & Non-streaming)
  - POST /v1/completions (Legacy prompt format)
  - GET  /v1/models
  - GET  /health
  - GET  /metrics (Prometheus scrape endpoint)
"""

import os
import sys
import time
import json
import uuid
import asyncio
from typing import AsyncGenerator
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.server.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionChunk,
    CompletionResponseStreamChoice,
    UsageInfo,
    ModelCard,
    ModelList,
    ErrorDetail,
    ErrorResponse,
    KVCacheMemoryError
)
from src.server.serving_engine import AsyncServingEngine
from src.engine.sampling import SamplingParams
from src.config import get_logger

logger = get_logger("mini_vllm.server")


def create_app(engine: AsyncServingEngine) -> FastAPI:
    """Builds and configures the FastAPI application with an active serving engine."""
    app = FastAPI(
        title="Mini-vLLM OpenAI-Compatible API Server",
        description="High-Throughput, Memory-Efficient LLM Serving Engine",
        version="1.0.0"
    )

    # Enable wide-open CORS so local UIs (Open WebUI, LibreChat, Continue.dev, browser apps) connect seamlessly
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # =====================================================================
    # Exception Handlers (OpenAI Format)
    # =====================================================================
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        errors = exc.errors()
        messages = []
        first_param = None
        for err in errors:
            loc = err.get("loc", [])
            loc_parts = [str(l) for l in loc if l != "body"]
            param_name = ".".join(loc_parts) if loc_parts else None
            if first_param is None and param_name:
                first_param = param_name
            msg = err.get("msg", "Validation error")
            messages.append(f"{param_name}: {msg}" if param_name else msg)
        full_message = "; ".join(messages) if messages else "Validation error"
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error=ErrorDetail(
                    message=full_message,
                    type="invalid_request_error",
                    param=first_param,
                    code=422
                )
            ).model_dump()
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        err_type = "invalid_request_error" if exc.status_code < 500 else "api_error"
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=ErrorDetail(
                    message=str(exc.detail),
                    type=err_type,
                    param=None,
                    code=exc.status_code
                )
            ).model_dump()
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled server exception: %s", exc, exc_info=True)
        if isinstance(exc, KVCacheMemoryError):
            return JSONResponse(
                status_code=503,
                content=ErrorResponse(
                    error=ErrorDetail(
                        message="KV cache capacity exceeded. The request cannot fit in the available KV cache pool.",
                        type="server_error",
                        param=None,
                        code="kv_cache_full"
                    )
                ).model_dump()
            )
        is_oom = (
            isinstance(exc, (torch.cuda.OutOfMemoryError, getattr(torch, "OutOfMemoryError", ())))
            or "out of memory" in str(exc).lower()
        )
        if is_oom:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return JSONResponse(
                status_code=503,
                content=ErrorResponse(
                    error=ErrorDetail(
                        message="GPU out of memory during inference. Reduce max_tokens or prompt length.",
                        type="server_error",
                        param=None,
                        code="cuda_out_of_memory"
                    )
                ).model_dump()
            )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    message=f"Internal server error: {type(exc).__name__}",
                    type="api_error",
                    param=None,
                    code=500
                )
            ).model_dump()
        )

    # =====================================================================
    # Health & Meta Endpoints
    # =====================================================================
    @app.get("/")
    async def root():
        return {
            "name": "Mini-vLLM Inference Server",
            "version": "1.0.0",
            "model": engine.model_name,
            "device": engine.device_str,
            "status": "online",
            "docs": "/docs",
            "metrics": "/metrics"
        }

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "model": engine.model_name,
            "device": engine.device_str,
            "running_requests": engine.metrics.num_requests_running
        }

    @app.get("/v1/models", response_model=ModelList)
    async def list_models():
        return ModelList(
            data=[
                ModelCard(
                    id=engine.model_name,
                    created=int(time.time()),
                    owned_by="mini-vllm"
                ),
                ModelCard(
                    id="mini-vllm",
                    created=int(time.time()),
                    owned_by="mini-vllm"
                )
            ]
        )

    # =====================================================================
    # Prometheus Metrics Exposition
    # =====================================================================
    @app.get("/metrics")
    async def get_metrics():
        metrics_text = engine.metrics.export_prometheus_text()
        return PlainTextResponse(metrics_text, media_type="text/plain; version=0.0.4; charset=utf-8")

    # =====================================================================
    # POST /v1/chat/completions
    # =====================================================================
    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        # 1. Format messages into model prompt
        messages_dict = [{"role": m.role, "content": m.content} for m in req.messages]
        prompt = engine.format_chat_prompt(messages_dict)

        # 2. Parse sampling parameters
        stop_strs = []
        if req.stop:
            stop_strs = [req.stop] if isinstance(req.stop, str) else list(req.stop)

        sampling_params = SamplingParams(
            temperature=req.temperature if req.temperature is not None else 0.7,
            top_p=req.top_p if req.top_p is not None else 1.0,
            top_k=req.top_k if req.top_k is not None else -1,
            min_p=req.min_p if req.min_p is not None else 0.0,
            repetition_penalty=req.repetition_penalty if req.repetition_penalty is not None else 1.0,
            frequency_penalty=req.frequency_penalty if req.frequency_penalty is not None else 0.0,
            presence_penalty=req.presence_penalty if req.presence_penalty is not None else 0.0,
            stop_strings=stop_strs
        )
        max_tokens = req.max_tokens if req.max_tokens is not None else 256

        # 3. Handle Streaming (Server-Sent Events)
        if req.stream:
            async def sse_stream_generator() -> AsyncGenerator[str, None]:
                req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                created_ts = int(time.time())

                # Initial chunk: specify role assistant
                first_chunk = ChatCompletionChunk(
                    id=req_id,
                    created=created_ts,
                    model=req.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(role="assistant", content=""),
                            finish_reason=None
                        )
                    ]
                )
                yield f"data: {first_chunk.model_dump_json()}\n\n"

                prompt_token_count = len(engine.tokenizer.encode(prompt)) if getattr(engine, "tokenizer", None) is not None else 0
                completion_token_count = 0

                try:
                    async for stream_item in engine.generate_stream(prompt, sampling_params, max_tokens):
                        token_text, _, is_last, finish_reason = stream_item
                        if hasattr(stream_item, "completion_tokens"):
                            completion_token_count = stream_item.completion_tokens
                        else:
                            completion_token_count += 1
                        if hasattr(stream_item, "prompt_tokens") and stream_item.prompt_tokens > 0:
                            prompt_token_count = stream_item.prompt_tokens

                        chunk = ChatCompletionChunk(
                            id=req_id,
                            created=created_ts,
                            model=req.model,
                            choices=[
                                ChatCompletionChunkChoice(
                                    index=0,
                                    delta=ChatCompletionChunkDelta(content=token_text),
                                    finish_reason=finish_reason if is_last else None
                                )
                            ]
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"

                    if req.stream_options and req.stream_options.include_usage:
                        usage_chunk = ChatCompletionChunk(
                            id=req_id,
                            created=created_ts,
                            model=req.model,
                            choices=[],
                            usage=UsageInfo(
                                prompt_tokens=prompt_token_count,
                                completion_tokens=completion_token_count,
                                total_tokens=prompt_token_count + completion_token_count
                            )
                        )
                        yield f"data: {usage_chunk.model_dump_json()}\n\n"

                    yield "data: [DONE]\n\n"
                except (GeneratorExit, asyncio.CancelledError):
                    raise
                except KVCacheMemoryError as e:
                    logger.error("SSE chat stream KV cache full: %s", e)
                    err_json = json.dumps({
                        "error": {
                            "message": "KV cache capacity exceeded. The request cannot fit in the available KV cache pool.",
                            "type": "server_error",
                            "param": None,
                            "code": "kv_cache_full"
                        }
                    })
                    yield f"data: {err_json}\n\n"
                except (torch.cuda.OutOfMemoryError, getattr(torch, "OutOfMemoryError", ())) as e:
                    logger.error("SSE chat stream CUDA OOM: %s", e)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    err_json = json.dumps({
                        "error": {
                            "message": "GPU out of memory during inference. Reduce max_tokens or prompt length.",
                            "type": "server_error",
                            "param": None,
                            "code": "cuda_out_of_memory"
                        }
                    })
                    yield f"data: {err_json}\n\n"
                except Exception as e:
                    logger.error("SSE chat stream error: %s", e)
                    err_json = json.dumps({
                        "error": {
                            "message": f"Generation error: {type(e).__name__}",
                            "type": "server_error",
                            "param": None,
                            "code": 500
                        }
                    })
                    yield f"data: {err_json}\n\n"

            return StreamingResponse(
                sse_stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )

        # 4. Handle Non-Streaming JSON Response
        try:
            full_text, p_tokens, gen_tokens, finish_reason = await engine.generate_complete(prompt, sampling_params, max_tokens)
        except KVCacheMemoryError as e:
            logger.error("Chat completion KV cache full: %s", e)
            return JSONResponse(
                status_code=503,
                content=ErrorResponse(
                    error=ErrorDetail(
                        message="KV cache capacity exceeded. The request cannot fit in the available KV cache pool.",
                        type="server_error",
                        param=None,
                        code="kv_cache_full"
                    )
                ).model_dump()
            )
        except (torch.cuda.OutOfMemoryError, getattr(torch, "OutOfMemoryError", ())) as e:
            logger.error("Chat completion CUDA OOM: %s", e)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return JSONResponse(
                status_code=503,
                content=ErrorResponse(
                    error=ErrorDetail(
                        message="GPU out of memory during inference. Reduce max_tokens or prompt length.",
                        type="server_error",
                        param=None,
                        code="cuda_out_of_memory"
                    )
                ).model_dump()
            )
        except Exception as e:
            logger.error("Chat completion generation error: %s", e)
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error=ErrorDetail(
                        message=f"Internal server error: {type(e).__name__}",
                        type="api_error",
                        param=None,
                        code=500
                    )
                ).model_dump()
            )

        return ChatCompletionResponse(
            model=req.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=full_text),
                    finish_reason=finish_reason
                )
            ],
            usage=UsageInfo(
                prompt_tokens=p_tokens,
                completion_tokens=gen_tokens,
                total_tokens=p_tokens + gen_tokens
            )
        )

    # =====================================================================
    # POST /v1/completions (Legacy text completion)
    # =====================================================================
    @app.post("/v1/completions")
    async def legacy_completions(req: CompletionRequest):
        prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
        stop_strs = [req.stop] if isinstance(req.stop, str) else (req.stop or [])

        sampling_params = SamplingParams(
            temperature=req.temperature if req.temperature is not None else 0.7,
            top_p=req.top_p if req.top_p is not None else 1.0,
            repetition_penalty=req.repetition_penalty if req.repetition_penalty is not None else 1.0,
            stop_strings=stop_strs
        )
        max_tokens = req.max_tokens if req.max_tokens is not None else 128

        if req.stream:
            async def sse_text_generator() -> AsyncGenerator[str, None]:
                req_id = f"cmpl-{uuid.uuid4().hex[:12]}"
                created_ts = int(time.time())
                prompt_token_count = len(engine.tokenizer.encode(prompt)) if getattr(engine, "tokenizer", None) is not None else 0
                completion_token_count = 0
                try:
                    async for stream_item in engine.generate_stream(prompt, sampling_params, max_tokens):
                        token_text, _, is_last, finish_reason = stream_item
                        if hasattr(stream_item, "completion_tokens"):
                            completion_token_count = stream_item.completion_tokens
                        else:
                            completion_token_count += 1
                        if hasattr(stream_item, "prompt_tokens") and stream_item.prompt_tokens > 0:
                            prompt_token_count = stream_item.prompt_tokens

                        chunk = CompletionChunk(
                            id=req_id,
                            created=created_ts,
                            model=req.model,
                            choices=[
                                CompletionResponseStreamChoice(
                                    index=0,
                                    text=token_text,
                                    finish_reason=finish_reason if is_last else None
                                )
                            ]
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"

                    if req.stream_options and req.stream_options.include_usage:
                        usage_chunk = CompletionChunk(
                            id=req_id,
                            created=created_ts,
                            model=req.model,
                            choices=[],
                            usage=UsageInfo(
                                prompt_tokens=prompt_token_count,
                                completion_tokens=completion_token_count,
                                total_tokens=prompt_token_count + completion_token_count
                            )
                        )
                        yield f"data: {usage_chunk.model_dump_json()}\n\n"

                    yield "data: [DONE]\n\n"
                except (GeneratorExit, asyncio.CancelledError):
                    raise
                except KVCacheMemoryError as e:
                    logger.error("SSE legacy text stream KV cache full: %s", e)
                    err_json = json.dumps({
                        "error": {
                            "message": "KV cache capacity exceeded. The request cannot fit in the available KV cache pool.",
                            "type": "server_error",
                            "param": None,
                            "code": "kv_cache_full"
                        }
                    })
                    yield f"data: {err_json}\n\n"
                except (torch.cuda.OutOfMemoryError, getattr(torch, "OutOfMemoryError", ())) as e:
                    logger.error("SSE legacy text stream CUDA OOM: %s", e)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    err_json = json.dumps({
                        "error": {
                            "message": "GPU out of memory during inference. Reduce max_tokens or prompt length.",
                            "type": "server_error",
                            "param": None,
                            "code": "cuda_out_of_memory"
                        }
                    })
                    yield f"data: {err_json}\n\n"
                except Exception as e:
                    logger.error("SSE legacy text stream error: %s", e)
                    err_json = json.dumps({
                        "error": {
                            "message": f"Generation error: {type(e).__name__}",
                            "type": "server_error",
                            "param": None,
                            "code": 500
                        }
                    })
                    yield f"data: {err_json}\n\n"

            return StreamingResponse(sse_text_generator(), media_type="text/event-stream")

        try:
            full_text, p_tokens, gen_tokens, finish_reason = await engine.generate_complete(prompt, sampling_params, max_tokens)
        except KVCacheMemoryError as e:
            logger.error("Legacy completion KV cache full: %s", e)
            return JSONResponse(
                status_code=503,
                content=ErrorResponse(
                    error=ErrorDetail(
                        message="KV cache capacity exceeded. The request cannot fit in the available KV cache pool.",
                        type="server_error",
                        param=None,
                        code="kv_cache_full"
                    )
                ).model_dump()
            )
        except (torch.cuda.OutOfMemoryError, getattr(torch, "OutOfMemoryError", ())) as e:
            logger.error("Legacy completion CUDA OOM: %s", e)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return JSONResponse(
                status_code=503,
                content=ErrorResponse(
                    error=ErrorDetail(
                        message="GPU out of memory during inference. Reduce max_tokens or prompt length.",
                        type="server_error",
                        param=None,
                        code="cuda_out_of_memory"
                    )
                ).model_dump()
            )
        except Exception as e:
            logger.error("Legacy completion generation error: %s", e)
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error=ErrorDetail(
                        message=f"Internal server error: {type(e).__name__}",
                        type="api_error",
                        param=None,
                        code=500
                    )
                ).model_dump()
            )
        return CompletionResponse(
            model=req.model,
            choices=[
                CompletionResponseChoice(
                    index=0,
                    text=full_text,
                    finish_reason=finish_reason
                )
            ],
            usage=UsageInfo(
                prompt_tokens=p_tokens,
                completion_tokens=gen_tokens,
                total_tokens=p_tokens + gen_tokens
            )
        )

    return app
