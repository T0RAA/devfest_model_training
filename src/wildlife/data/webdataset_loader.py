"""WebDataset `.tar` shard loader for T2/T3/distillation training (§6.4 & §7.5)."""

from __future__ import annotations

import json
from pathlib import Path
import tarfile
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np

from wildlife.data.clean import decode_image_bytes


def iter_webdataset_shard(shard_tar_path: Path) -> Iterator[Tuple[np.ndarray, int, Dict[str, Any]]]:
    """Iterate over `(rgb_uint8, class_idx, metadata_dict)` records stored in a WebDataset `.tar` shard."""
    shard_tar_path = Path(shard_tar_path)
    grouped: Dict[str, Dict[str, bytes]] = {}
    order: List[str] = []

    with tarfile.open(shard_tar_path, "r") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = Path(member.name).name
            if "." not in name:
                continue
            key, ext = name.rsplit(".", 1)
            fobj = tar.extractfile(member)
            if fobj is None:
                continue
            if key not in grouped:
                grouped[key] = {}
                order.append(key)
            grouped[key][ext] = fobj.read()

    for key in order:
        parts = grouped[key]
        img_bytes = parts.get("png") or parts.get("jpg")
        if img_bytes is None:
            continue
        rgb, _ = decode_image_bytes(img_bytes)
        cls_idx = int(parts.get("cls", b"-1").decode("utf-8").strip())
        meta = (
            json.loads(parts["json"].decode("utf-8"))
            if "json" in parts
            else {"image_id": key}
        )
        yield rgb, cls_idx, meta
