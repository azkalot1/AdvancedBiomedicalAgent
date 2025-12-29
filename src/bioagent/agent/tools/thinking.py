# tools.py
from __future__ import annotations

import uuid
from typing import Annotated, Any, Literal
import dotenv
from pydantic import Field
from langchain_core.tools import tool
from bioagent.biomcp.thinking import mark_thinking_used
dotenv.load_dotenv()





@tool("think", return_direct=False)
async def think(
    thought: Annotated[str, Field(description="Your thinking, reasoning, or plan")],
) -> str:
    """
    Use this tool to think through a problem step-by-step before acting.
    
    Good for:
    - Breaking down complex tasks
    - Planning which tools to use
    - Reasoning about intermediate results
    - Adjusting approach based on new information
    
    Args:
        thought: Your current thinking, reasoning, or plan
    """
    mark_thinking_used()
    return "Thought recorded. Continue thinking or proceed to action."