"""
Utilities for extracting and injecting key points into PPO training batches.

This module provides helpers to extract GSM8K-style key points from
extra_info['answer'] (e.g., lines beginning with a specific prefix like
"#### ") and to store the results in both extra_info['keypoints'] and
top-level non-tensor batch under key 'keypoints'.
"""

from __future__ import annotations

import re
from typing import List, Optional

import numpy as np

from verl import DataProto


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


def inject_keypoints_from_extra_info(batch: DataProto, prefix: str = "####") -> None:
    """Inject extracted key points into the batch in-place.

    This function looks for per-sample "answer" strings under
    batch.non_tensor_batch['extra_info'] and creates per-sample key points
    by calling `extract_keypoints_from_answer_str`. The resulting list is
    stored in both extra_info['keypoints'] and the top-level
    non_tensor_batch['keypoints'].

    If top-level 'keypoints' already exists, this function is a no-op.

    Args:
        batch: The DataProto batch to modify in place.
        prefix: The prefix used in answer strings to denote the final answer.
    """
    if "keypoints" in batch.non_tensor_batch:
        return

    extras = batch.non_tensor_batch.get("extra_info", None)
    if extras is None:
        return

    batch_size = len(batch.batch)
    if hasattr(extras, "tolist"):
        extras_list = extras.tolist()
    else:
        extras_list = list(extras)

    keypoints_per_sample: List[List[str]] = []
    for i in range(batch_size):
        info = extras_list[i] if i < len(extras_list) else {}
        info = {} if info is None else dict(info)

        if "keypoints" in info and isinstance(info["keypoints"], list):
            kps = info["keypoints"]
        else:
            ans = info.get("answer", None)
            kps = extract_keypoints_from_answer_str(ans, prefix=prefix)
            info["keypoints"] = kps

        keypoints_per_sample.append(kps)
        extras_list[i] = info

    batch.non_tensor_batch["extra_info"] = np.array(extras_list, dtype=object)
    batch.non_tensor_batch["keypoints"] = np.array(keypoints_per_sample, dtype=object)


