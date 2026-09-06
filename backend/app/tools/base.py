"""Base tool interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT")


class BaseTool(ABC, Generic[InputT, OutputT]):
    """Abstract base class for Agent tools."""

    name: str = ""
    description: str = ""
    input_schema: type[BaseModel] = BaseModel

    @abstractmethod
    async def execute(self, input_data: InputT) -> OutputT:
        """Execute the tool with validated input."""
        ...

    async def __call__(self, **kwargs: Any) -> OutputT:
        input_data = self.input_schema(**kwargs)
        return await self.execute(input_data)

    def to_langchain_tool(self) -> dict[str, Any]:
        """Convert to LangChain-compatible tool spec."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema.model_json_schema(),
        }
