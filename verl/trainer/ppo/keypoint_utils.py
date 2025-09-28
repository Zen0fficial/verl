"""
Utilities for extracting and injecting key points for PPO training.

This module provides helpers to extract GSM8K-style key points from
extra_info['answer'] (e.g., lines beginning with a specific prefix like
"#### ") and to store the results back into data containers.
"""

from __future__ import annotations

import re
from typing import List, Optional, Iterable

import numpy as np
import pandas as pd


def extract_keypoints_from_answer_str(answer: Optional[str], prefix: str = "####") -> List[str]:
    """Extract key points from a GSM8K-style answer string.

    The function searches for a line beginning with the given prefix and
    returns a list with a single formatted key point string like
    "#### 18". If nothing is found, returns an empty list.

    Args:
        answer: The full answer text.
        prefix: The prefix used to mark the final answer line.

    Returns:
        A list of extracted key point strings.
    """
    if not isinstance(answer, str):
        return []

    try:
        pattern = rf"^\s*{re.escape(prefix)}\s*(.+)$"
        match = re.search(pattern, answer, flags=re.MULTILINE)
        if not match:
            return []
        value = match.group(1).strip()
        if not value:
            return []
        return [f"{prefix} {value}"]
    except Exception:
        return []


def inject_keypoints_into_extra_info(extra_info: dict, prefix: str = "####") -> dict:
    """Return a new extra_info dict with keypoints extracted from 'answer'.

    If 'keypoints' already exists and is a list, it is preserved.
    """
    info = {} if extra_info is None else dict(extra_info)
    if isinstance(info.get("keypoints"), list):
        return info
    keypoints = extract_keypoints_from_answer_str(info.get("answer"), prefix=prefix)
    info["keypoints"] = keypoints
    return info


def annotate_dataframe_with_keypoints(
    df: pd.DataFrame,
    extra_info_col: str = "extra_info",
    keypoints_col: Optional[str] = None,
    prefix: str = "####",
    inplace: bool = False,
) -> pd.DataFrame:
    """Annotate a pandas DataFrame with extracted key points.

    - Reads per-row dicts from `extra_info_col` (each should contain 'answer').
    - Writes back 'keypoints' into those dicts.
    - Optionally writes the extracted list to a separate column `keypoints_col`.

    Returns the modified DataFrame (or a copy if inplace=False).
    """
    if not inplace:
        df = df.copy()

    if extra_info_col not in df.columns:
        raise KeyError(f"Column '{extra_info_col}' not found in DataFrame")

    def _process_info(x):
        return inject_keypoints_into_extra_info(x, prefix=prefix)

    # Apply to the column; accept scalars or arrays of dicts
    if isinstance(df[extra_info_col].iloc[0], (list, tuple, np.ndarray)):
        # If extra_info is a sequence per row, process element-wise
        def _process_seq(seq: Iterable):
            seq = [] if seq is None else list(seq)
            return [inject_keypoints_into_extra_info(elem, prefix=prefix) for elem in seq]

        df[extra_info_col] = df[extra_info_col].apply(_process_seq)
    else:
        df[extra_info_col] = df[extra_info_col].apply(_process_info)

    if keypoints_col is not None:
        def _extract_kps_from_info(x):
            if isinstance(x, dict):
                return x.get("keypoints", [])
            if isinstance(x, (list, tuple, np.ndarray)):
                return [d.get("keypoints", []) for d in x]
            return []

        df[keypoints_col] = df[extra_info_col].apply(_extract_kps_from_info)

    return df

