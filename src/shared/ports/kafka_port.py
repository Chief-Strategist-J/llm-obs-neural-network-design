from __future__ import annotations
from typing import Protocol, Mapping


class KafkaProducerPort(Protocol):
    def produce(
        self,
        topic: str,
        key: str,
        value: bytes,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        ...

    def flush(self, timeout: float = 10.0) -> None:
        ...


class KafkaConsumerPort(Protocol):
    def subscribe(self, topics: list[str]) -> None:
        ...

    def poll(self, timeout: float = 1.0) -> object | None:
        ...

    def commit(self, asynchronous: bool = True) -> None:
        ...

    def close(self) -> None:
        ...
