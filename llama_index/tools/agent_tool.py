from typing import Any, Optional, Callable, Type

from pydantic import BaseModel
from llama_index.tools.types import BaseTool, ToolMetadata, ToolOutput
from llama_index.tools.utils import create_schema_from_function


class AgentTool(BaseTool):
    """Agent Tool
    
    a tool that wraps a LlamaIndex Agent and converts it to a tool
    """
    
    def __init__(
        self,
        agent: BaseAgent,
        metadata: ToolMetadata
    ) -> None:
        self._agent = agent
        self._metadata = metadata
        
    @classmethod
    def from_defaults(
        cls,
        agent: BaseAgent,
        name: str,
        description: str,
        fn_schema: Optional[Type[BaseModel]] = None,
    ) -> "FunctionTool":
        if fn_schema is None:
            fn_schema = create_schema_from_function(
                f"{name}", agent.chat, additional_fields=None
            )
        metadata = ToolMetadata(name=name, description=description, fn_schema=fn_schema)
        return cls(agent=agent, metadata=metadata)
    

    @property
    def metadata(self) -> ToolMetadata:
        """Metadata."""
        return self._metadata

    @property
    def fn(self) -> Callable[..., Any]:
        """Function."""
        return self._agent.chat

    def __call__(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Call."""
        tool_output = self._agent.chat(*args, **kwargs)
        return ToolOutput(
            content=str(tool_output),
            tool_name=self.metadata.name,
            raw_input={"args": args, "kwargs": kwargs},
            raw_output=tool_output,
        )
