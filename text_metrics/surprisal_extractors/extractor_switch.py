"""Compatibility wrapper for legacy get_surp_extractor imports."""

from psycholing_metrics.surprisal import create_surprisal_extractor


def get_surp_extractor(*args, **kwargs):
    """Return a psycholing_metrics surprisal extractor using the legacy name."""
    return create_surprisal_extractor(*args, **kwargs)


__all__ = ["get_surp_extractor"]

