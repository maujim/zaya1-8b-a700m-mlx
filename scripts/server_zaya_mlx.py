#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from run_zaya_mlx import MODEL_ID, QUANT_CHOICES, Profiler, enable_moe_decode_fast_path, generate_from_messages, load_model

SERVER_MODEL_ID = "zaya-mlx"


class ChatMessage(BaseModel):
    role: str
    content: Any = ""


class ChatCompletionRequest(BaseModel):
    model: str = SERVER_MODEL_ID
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None)
    max_completion_tokens: int | None = Field(default=None)
    temperature: float | None = 0.0
    stream: bool = False


def normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text = item.get("text", "")
                    if isinstance(text, str):
                        chunks.append(text)
                elif isinstance(item.get("content"), str):
                    chunks.append(item["content"])
        return "\n".join(chunks)
    if content is None:
        return ""
    return str(content)


def openai_chunk(completion_id: str, content: str | None, finish_reason: str | None = None) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": SERVER_MODEL_ID,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    if content is not None:
        payload["choices"][0]["delta"] = {"content": content}
    return f"data: {json.dumps(payload)}\n\n"


def create_app(quant: str = "full", moe_decode_fast_path: bool = True, use_cache: bool = True) -> FastAPI:
    app = FastAPI(title="ZAYA MLX OpenAI-compatible server")

    profiler = Profiler(enabled=True)
    with profiler.span("resolve_model_path"):
        model_path = Path(snapshot_download(MODEL_ID))
    print(f"MLX model path: {model_path}", flush=True)

    model = load_model(model_path, profiler, quant=quant)
    if moe_decode_fast_path:
        enable_moe_decode_fast_path(model)
    with profiler.span("final_parameter_sync", force_eval=model.parameters()):
        pass
    print("MLX model loaded and synchronized", flush=True)

    with profiler.span("load_tokenizer"):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    profiler.print_report()

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": SERVER_MODEL_ID,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                    "context_window": 8192,
                    "max_tokens": 2048,
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest):
        if request.model != SERVER_MODEL_ID:
            raise HTTPException(status_code=404, detail=f"unknown model: {request.model}")

        messages = [{"role": msg.role, "content": normalize_content(msg.content)} for msg in request.messages]
        max_tokens = request.max_completion_tokens or request.max_tokens or 500
        temperature = 0.0 if request.temperature is None else request.temperature
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"

        if request.stream:
            def events():
                yield openai_chunk(completion_id, "")
                for token in generate_from_messages(model, tokenizer, messages, max_tokens, temperature, use_cache=use_cache):
                    text = tokenizer.decode([token], skip_special_tokens=True)
                    if text:
                        yield openai_chunk(completion_id, text)
                yield openai_chunk(completion_id, None, "stop")
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                events(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        pieces = []
        for token in generate_from_messages(model, tokenizer, messages, max_tokens, temperature, use_cache=use_cache):
            pieces.append(tokenizer.decode([token], skip_special_tokens=True))
        content = "".join(pieces)
        return JSONResponse(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": SERVER_MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": len(pieces), "total_tokens": len(pieces)},
            }
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local ZAYA MLX port via an OpenAI-compatible API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--quant", choices=QUANT_CHOICES, default="full", help="Weight mode: full BF16 weights or quick dynamic Q8 quantization after load.")
    parser.add_argument("--cache", dest="cache", action="store_true", default=True, help="Use KV/CCA cached generation (default).")
    parser.add_argument("--no-cache", dest="cache", action="store_false", help="Disable KV/CCA cached generation.")
    parser.add_argument(
        "--moe-decode-fast-path",
        dest="moe_decode_fast_path",
        action="store_true",
        default=True,
        help="Use single-token MoE expert short-circuit during decode (default).",
    )
    parser.add_argument(
        "--no-moe-decode-fast-path",
        dest="moe_decode_fast_path",
        action="store_false",
        help="Disable single-token MoE expert short-circuit.",
    )
    args = parser.parse_args()

    uvicorn.run(
        create_app(args.quant, moe_decode_fast_path=args.moe_decode_fast_path, use_cache=args.cache),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
