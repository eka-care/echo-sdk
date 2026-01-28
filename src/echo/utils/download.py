"""Utilities for downloading and sanitising file content."""

import logging

import httpx

logger = logging.getLogger(__name__)


def _strip_multipart_wrapper(data: bytes) -> bytes:
    """Extract actual file bytes from a multipart form-data wrapper.

    When a client uploads a file to an S3 presigned URL using
    ``multipart/form-data`` instead of a raw binary PUT, S3 stores the
    entire multipart envelope (boundary + headers + payload + closing
    boundary).  This helper detects that case and returns only the file
    payload.

    If *data* is not multipart-wrapped it is returned unchanged.
    """
    if not data.startswith(b"--") or data.startswith(b"--\r\n"):
        return data

    boundary_end = data.find(b"\r\n")
    if boundary_end == -1:
        return data
    boundary = data[:boundary_end]

    # Payload starts after the first blank line (\r\n\r\n)
    header_end = data.find(b"\r\n\r\n", boundary_end)
    if header_end == -1:
        return data
    payload_start = header_end + 4

    # Closing boundary is the same token followed by '--'
    close_boundary = boundary + b"--"
    payload_end = data.rfind(close_boundary)
    if payload_end > payload_start:
        payload = data[payload_start:payload_end].rstrip(b"\r\n")
    else:
        payload = data[payload_start:]

    logger.debug(
        "Stripped multipart wrapper: %d bytes -> %d bytes",
        len(data),
        len(payload),
    )
    return payload


def download_url_as_bytes(url: str) -> bytes:
    """Download content from a URL and return raw file bytes.

    Automatically strips multipart form-data wrappers that are present
    when a file was uploaded to S3 via multipart POST instead of a raw
    binary PUT.
    """
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(str(url))
        resp.raise_for_status()
        return _strip_multipart_wrapper(resp.content)
