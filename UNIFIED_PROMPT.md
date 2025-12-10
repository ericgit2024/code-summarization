# Unified Prompt Instruction for All Files

## Standard Instruction (to be used everywhere)

```python
STANDARD_INSTRUCTION = (
    "Generate a concise docstring summary for this code.\\n"
    "Write 1-3 sentences explaining what the code does.\\n"
    "Do NOT use markdown, bullet points, or structured sections."
)
```

## Files to Update

1. **trainer.py** (line 67-73)
2. **reflective_agent.py** (line 65-83)
3. **inference.py** (line 204-215)
4. **prompt.py** (line 33-43)

## Rationale

CodeSearchNet docstrings are simple, like:
- "Validates user input and creates a product."
- "Returns all orders for a given customer ID."
- "Generates a new authentication token with TTL expiration."

Keep it SIMPLE - the model is over-instructed and producing garbage.
