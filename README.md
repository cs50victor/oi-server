# Run server

- `brew install uv pre-commit`
- `pre-commit install`
- `uv run examples/basic.py`
- `uv run pytest tests -q`

if using the oi_server on a unix system, you can import uvloop and add this to the top of you server file

```python
import uvloop # uvloop makes asyncio 2-4x faster.
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
```
