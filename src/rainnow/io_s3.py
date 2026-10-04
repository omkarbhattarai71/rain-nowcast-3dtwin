"""Read-only S3 helpers. Nothing in this project writes to, or creates, S3 buckets."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def client(region: str = "eu-north-1"):
    import boto3
    from botocore.config import Config

    return boto3.client("s3", region_name=region, config=Config(max_pool_connections=64, retries={"max_attempts": 8}))


def list_keys(bucket: str, prefix: str = "", region: str = "eu-north-1", delimiter: str | None = None) -> list[dict]:
    """List all objects under a prefix (handles pagination). Returns [{'Key','Size'}, ...]."""
    s3 = client(region)
    out: list[dict] = []
    token = None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if delimiter:
            kw["Delimiter"] = delimiter
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        out += [{"Key": o["Key"], "Size": o["Size"]} for o in resp.get("Contents", [])]
        if not resp.get("IsTruncated"):
            return out
        token = resp["NextContinuationToken"]


def read_bytes(bucket: str, key: str, region: str = "eu-north-1") -> bytes:
    return client(region).get_object(Bucket=bucket, Key=key)["Body"].read()


def download(bucket: str, key: str, dest: Path, region: str = "eu-north-1", skip_existing: bool = True) -> Path:
    dest = Path(dest)
    if skip_existing and dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    client(region).download_file(bucket, key, str(tmp))
    tmp.replace(dest)
    return dest


def download_many(
    bucket: str,
    items: list[tuple[str, Path]],
    region: str = "eu-north-1",
    workers: int = 16,
    skip_existing: bool = True,
) -> list[Path]:
    """Download (key, dest) pairs in parallel. Failures are logged and skipped."""
    done: list[Path] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(download, bucket, k, d, region, skip_existing): k for k, d in items}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                done.append(fut.result())
            except Exception as exc:  # noqa: BLE001 - keep going, report at the end
                log.warning("download failed %s: %s", futs[fut], exc)
            if i % 500 == 0:
                log.info("  downloaded %d / %d", i, len(items))
    return done
