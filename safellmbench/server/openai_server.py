"""
Minimal OpenAI-compatible server that serves ONE HuggingFace causal-LM model.

Endpoints implemented (non-streaming, plenty for the benchmark loop):
    GET  /v1/models
    POST /v1/chat/completions
    POST /v1/completions
    GET  /health

Compatible with the `openai` Python client — the benchmark runner talks to
this server just like it would talk to OpenAI's real API.
"""

from __future__ import annotations

import time
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..models.target import TargetModel, load_target
from .. import config


# ---- Request / response schemas -------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9


# ---- App factory -----------------------------------------------------------
def build_app(model_id: str) -> FastAPI:
    app = FastAPI(title="SafeLLMBench Target Server", version="0.1.0")
    target: TargetModel = load_target(model_id)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": target.model_id}

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{
                "id": target.model_id, "object": "model",
                "created": int(time.time()), "owned_by": "safellmbench",
            }],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        if req.stream:
            raise HTTPException(400, "streaming not supported by SafeLLMBench server")
        # Extract the (optional) system prompt and the final user message.
        # For a safety benchmark we only care about the last user turn — the
        # target model's chat template is applied internally in TargetModel.
        system_prompt = None
        user_content = ""
        for m in req.messages:
            if m.role == "system":
                system_prompt = m.content
            elif m.role == "user":
                user_content = m.content
        text = target.generate(
            user_content, system_prompt=system_prompt,
            max_new_tokens=req.max_tokens or 256,
            temperature=req.temperature or 0.7,
            top_p=req.top_p or 0.9,
        )
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": target.model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        text = target.generate(
            req.prompt, system_prompt=None,
            max_new_tokens=req.max_tokens or 256,
            temperature=req.temperature or 0.7,
            top_p=req.top_p or 0.9,
        )
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": target.model_id,
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    return app


def serve(model_id: str,
          host: str = config.DEFAULT_HOST,
          port: int = config.DEFAULT_PORT) -> None:
    """Blocking call — start the uvicorn server. Ctrl-C to stop."""
    import uvicorn
    app = build_app(model_id)
    print(f"[server] serving {model_id} on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
