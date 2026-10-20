"""Importing this module adds "Authorization: Bearer $KIT_API_KEY" to every urllib request when KIT_API_KEY is set
(servers started with --api-key). No effect otherwise."""
import os
import urllib.request

_KEY = os.environ.get("KIT_API_KEY")
if _KEY:
    _orig = urllib.request.urlopen

    def _urlopen(req, *a, **kw):
        if isinstance(req, str):
            req = urllib.request.Request(req)
        if not req.has_header("Authorization"):
            req.add_header("Authorization", "Bearer " + _KEY)
        return _orig(req, *a, **kw)

    urllib.request.urlopen = _urlopen
