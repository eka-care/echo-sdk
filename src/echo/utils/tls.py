"""TLS verification settings for self-hosted / on-prem HTTPS endpoints.

Such endpoints often sit behind a private CA or a plain self-signed cert.
``resolve_ssl_verify`` turns env vars into what httpx / the OpenAI SDK accept
for ``verify``: ``False`` (verification disabled — dev/testing only), an
``ssl.SSLContext`` trusting a CA bundle (the production-grade option), or
``True`` (default trust store).
"""

from __future__ import annotations

import logging
import os
import ssl
from typing import Iterable, Union

logger = logging.getLogger(__name__)

DEFAULT_HTTP_TIMEOUT_S = 600.0

_FALSY = ("0", "false", "no", "off")


def resolve_ssl_verify(
    verify_vars: Iterable[str], ca_vars: Iterable[str]
) -> Union[bool, ssl.SSLContext]:
    """Resolve TLS verification from env.

    - the first env var in ``verify_vars`` set to a falsy value
      (0/false/no/off) disables verification and wins over any CA bundle;
    - otherwise the first set env var in ``ca_vars`` (path to a PEM file) gives
      an ``ssl.SSLContext`` trusting that CA;
    - otherwise ``True`` (default trust store).
    """
    for var in verify_vars:
        raw = os.getenv(var)
        if raw is not None and raw.strip():
            if raw.strip().lower() in _FALSY:
                logger.warning(
                    "TLS certificate verification DISABLED via %s — use a CA "
                    "bundle instead for anything beyond local testing.",
                    var,
                )
                return False
            break
    for var in ca_vars:
        path = os.getenv(var)
        if path:
            return ssl.create_default_context(cafile=path)
    return True


def build_custom_http_client(verify_vars, ca_vars, timeout_s=DEFAULT_HTTP_TIMEOUT_S):
    """Sync ``httpx.Client`` honoring the TLS env vars, or None for SDK defaults."""
    verify = resolve_ssl_verify(verify_vars, ca_vars)
    if verify is True:
        return None
    import httpx

    return httpx.Client(verify=verify, timeout=timeout_s)
