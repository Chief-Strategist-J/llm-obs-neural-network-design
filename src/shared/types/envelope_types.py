"""Centralized result envelopes and trace models."""

from dataclasses import dataclass
from typing import Any, Generic, Mapping, TypeVar

T = TypeVar("T")
E = TypeVar("E", bound=Exception)


@dataclass(frozen=True)
class TraceContext:
    trace_id: str
    span_id: str = ""


@dataclass(frozen=True)
class Result(Generic[T, E]):
    value: T | None
    error: E | None
    is_ok: bool

    @classmethod
    def ok(cls, val: T) -> "Result[T, Any]":
        return cls(value=val, error=None, is_ok=True)

    @classmethod
    def err(cls, error: E) -> "Result[Any, E]":
        return cls(value=None, error=error, is_ok=False)


@dataclass(frozen=True)
class ApiResponse(Generic[T]):
    success: bool
    data: T | None
    meta: Mapping[str, Any]
    error: Mapping[str, Any] | None = None
