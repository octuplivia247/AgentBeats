"""Green agent module - responsible for managing evaluations."""


def start_green_agent(*args, **kwargs):
    """Lazy import to avoid requiring uvicorn at module load time."""
    from .agent import start_green_agent as _start_green_agent
    return _start_green_agent(*args, **kwargs)


__all__ = ["start_green_agent"]
