from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


JobType = Literal["train", "lora", "rag_index", "reasoning", "bpe_train"]


class JobSubmit(BaseModel):
    type: JobType
    config: dict[str, Any] = Field(default_factory=dict)


class ChatReq(BaseModel):
    prompt: str = ""
    max_tokens: int = Field(200, ge=1, le=8192)
    temperature: float = Field(0.8, ge=0.0, le=5.0)
    top_k: int = Field(40, ge=0)
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    use_rag: bool = False
    rag_k: int = 4
    use_reasoning_bias: bool = False
    verify: bool = False


class LoadModelReq(BaseModel):
    ckpt: str
    device: str = "auto"


class SignalsReq(BaseModel):
    prompt: str
    ckpt: Optional[str] = None
    device: str = "auto"


class RagLoadReq(BaseModel):
    path: str


class RagSearchReq(BaseModel):
    path: str
    query: str
    k: int = Field(4, ge=1, le=64)


class RagAddReq(BaseModel):
    kb: str
    corpus: list[str]
    chunk_size: int = 1000
    chunk_overlap: int = 200
    encoder: Literal["hidden", "sentence_transformer"] = "hidden"
    ckpt: Optional[str] = None


class GraphLoadReq(BaseModel):
    path: str


class GraphApplyReq(BaseModel):
    path: Optional[str] = None
    facts: list[str]


class GraphStatsReq(BaseModel):
    path: Optional[str] = None
    top: int = 10


class FileMkdirReq(BaseModel):
    path: str


class ShutdownReq(BaseModel):
    confirm: bool = False
