"""
OpenAI-Compatible REST API Protocols & Data Models
--------------------------------------------------
Implements strict Pydantic models conforming to the OpenAI REST specification:
  - POST /v1/chat/completions (Chat format, streaming & non-streaming)
  - POST /v1/completions (Raw text completion, streaming & non-streaming)
  - GET  /v1/models (Model listing)
"""

import time
import uuid
from typing import List, Optional, Union, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator


# =========================================================================
# Error Protocols (OpenAI Specification)
# =========================================================================
class ErrorDetail(BaseModel):
    message: str
    type: str = "server_error"
    param: Optional[str] = None
    code: Optional[Union[int, str]] = 500


class ErrorResponse(BaseModel):
    error: ErrorDetail


class KVCacheMemoryError(MemoryError):
    """Raised when KV cache capacity is exhausted (admission rejection or pool exhaustion)."""
    pass


# =========================================================================
# Chat Completion Protocols
# =========================================================================
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "function"]
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage] = Field(..., min_length=1, description="A list of messages comprising the conversation so far.")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=-1, ge=-1)
    min_p: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=256, gt=0)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    repetition_penalty: Optional[float] = Field(default=1.0, gt=0.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    user: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponseChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "abort"]] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


# =========================================================================
# Streaming Chunk Protocols (Server-Sent Events)
# =========================================================================
class ChatCompletionChunkDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[Literal["stop", "length", "abort"]] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]


# =========================================================================
# Legacy / Text Completion Protocols
# =========================================================================
class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = Field(default=128, gt=0)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    repetition_penalty: Optional[float] = Field(default=1.0, gt=0.0)

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("prompt cannot be empty")
        elif isinstance(v, list):
            if len(v) == 0:
                raise ValueError("prompt list cannot be empty")
            for i, p in enumerate(v):
                if not isinstance(p, str) or not p.strip():
                    raise ValueError(f"prompt[{i}] cannot be empty")
        else:
            raise ValueError("prompt must be a string or list of strings")
        return v


class CompletionResponseChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: Optional[str] = "stop"


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:12]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


# =========================================================================
# Model Registry Protocols
# =========================================================================
class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mini-vllm"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: List[Dict[str, Any]] = []


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]
