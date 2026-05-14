from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OpenAIBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class ModelCard(OpenAIBaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "kvserve"


class ListModelsResponse(OpenAIBaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class FunctionTool(OpenAIBaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    strict: bool | None = None


class ChatTool(OpenAIBaseModel):
    type: Literal["function"]
    function: FunctionTool


class ToolCallFunction(OpenAIBaseModel):
    name: str
    arguments: str


class ChatToolCall(OpenAIBaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ChatMessage(OpenAIBaseModel):
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ChatToolCall] | None = None


class ResponseFormat(OpenAIBaseModel):
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: dict[str, Any] | None = None


class StreamOptions(OpenAIBaseModel):
    include_usage: bool = True


class ChatCompletionRequest(OpenAIBaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool = False
    stream_options: StreamOptions = Field(default_factory=StreamOptions)
    stop: str | list[str] | None = None
    tools: list[ChatTool] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: ResponseFormat | None = None
    user: str | None = None
    seed: int | None = None

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        if not value:
            raise ValueError("messages must not be empty")
        return value

    @property
    def output_token_limit(self) -> int:
        return self.max_completion_tokens or self.max_tokens or 256


class CompletionUsage(OpenAIBaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatChoice(OpenAIBaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None = None


class ChatCompletionResponse(OpenAIBaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: CompletionUsage


class ChatDelta(OpenAIBaseModel):
    role: Literal["assistant", "tool"] | None = None
    content: str | None = None
    tool_calls: list[ChatToolCall] | None = None


class ChatChunkChoice(OpenAIBaseModel):
    index: int
    delta: ChatDelta
    finish_reason: str | None = None


class ChatCompletionChunk(OpenAIBaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatChunkChoice]
    usage: CompletionUsage | None = None


class EmbeddingEncodingFormat(StrEnum):
    FLOAT = "float"
    BASE64 = "base64"


class EmbeddingsRequest(OpenAIBaseModel):
    model: str
    input: str | list[str] | list[int] | list[list[int]]
    encoding_format: EmbeddingEncodingFormat = EmbeddingEncodingFormat.FLOAT
    dimensions: int | None = None
    user: str | None = None


class EmbeddingObject(OpenAIBaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: list[float] | str
    index: int


class EmbeddingsResponse(OpenAIBaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingObject]
    model: str
    usage: CompletionUsage


class RerankRequest(OpenAIBaseModel):
    model: str
    query: str
    documents: list[str]
    top_n: int | None = None
    return_documents: bool = True
    user: str | None = None

    @field_validator("documents")
    @classmethod
    def documents_not_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("documents must not be empty")
        return value


class RerankResult(OpenAIBaseModel):
    index: int
    relevance_score: float
    document: str | None = None


class RerankResponse(OpenAIBaseModel):
    id: str
    object: Literal["list"] = "list"
    model: str
    results: list[RerankResult]
    usage: CompletionUsage


class ErrorBody(OpenAIBaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(OpenAIBaseModel):
    error: ErrorBody
