"""Network helpers."""

import socket

_original = socket.getaddrinfo


def ipv4_only() -> None:
    """Resolve hosts to IPv4 only.

    Some networks advertise IPv6 without routing it; Python then hangs on every
    download. Call this before fetching pretrained weights on such a network.
    """
    socket.getaddrinfo = lambda host, port, family=0, *a, **kw: _original(host, port, socket.AF_INET, *a, **kw)
