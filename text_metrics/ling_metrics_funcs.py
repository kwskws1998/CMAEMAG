"""Compatibility exports for legacy text_metrics metric imports."""

from psycholing_metrics.metrics import get_metrics as _get_metrics


def get_metrics(*args, **kwargs):
    """Call psycholing_metrics.get_metrics and normalize legacy column names."""
    metrics = _get_metrics(*args, **kwargs)
    rename_map = {}
    for column in metrics.columns:
        if column.endswith("_cat_Surprisal"):
            rename_map[column] = column.replace("_cat_Surprisal", "_Surprisal")
    if rename_map:
        metrics = metrics.rename(columns=rename_map)
    return metrics


__all__ = ["get_metrics"]

