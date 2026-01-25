import json
import inspect
from functools import wraps

def robust_unwrap_llm_inputs(func):
    """
    Decorator to normalize LLM outputs into Python types.
    
    Handles:
    - JSON strings: '["a", "b"]' → ["a", "b"]
    - Null strings: "null", "None" → None
    - Wrapped dicts: {"value": [...]} → [...]
    - Single values: "aspirin" → ["aspirin"] (when list expected)
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        def process_value(value, param_name=None):
            if value is None:
                return None
            
            # Unwrap {"value": ...} or {"type": ..., "value": ...}
            if isinstance(value, dict) and "value" in value:
                value = value["value"]
                return process_value(value, param_name)
            
            # Handle string representations
            if isinstance(value, str):
                stripped = value.strip().lower()
                
                # Null-like strings
                if stripped in ("null", "none", ""):
                    return None
                
                # JSON arrays/objects
                if value.strip().startswith(("[", "{")):
                    try:
                        return json.loads(value)
                    except json.JSONDecodeError:
                        pass
                
                # Boolean strings
                if stripped == "true":
                    return True
                if stripped == "false":
                    return False
                
                # Return as-is
                return value
            
            return value

        processed_kwargs = {k: process_value(v, k) for k, v in kwargs.items()}
        
        if inspect.iscoroutinefunction(func):
            return await func(*args, **processed_kwargs)
        return func(*args, **processed_kwargs)

    return wrapper