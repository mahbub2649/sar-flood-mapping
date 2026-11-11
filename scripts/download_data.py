"""Download the hand-labelled part of Sen1Floods11 (Bonafilia et al., CVPRW 2020).

Public Google Cloud bucket, plain HTTPS, no account needed. About 1 GB.
Resumable: files already present with the right size are skipped.
Licence: CC BY 4.0 (Cloud to Street). See README for the citation.
"""

import json
import socket
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Some networks advertise IPv6 but cannot route it, and Python then hangs on every request.
# The bucket is reachable over IPv4 everywhere, so only IPv4 addresses are used.
_getaddrinfo = socket.getaddrinfo
socket.getaddrinfo = lambda host, port, family=0, *a, **kw: _getaddrinfo(host, port, socket.AF_INET, *a, **kw)

BUCKET = "sen1floods11"
ROOT = Path(__file__).resolve().parent.parent / "data"
PREFIXES = {
    "v1.1/data/flood_events/HandLabeled/S1Hand/": "S1Hand",
    "v1.1/data/flood_events/HandLabeled/LabelHand/": "LabelHand",
    "v1.1/splits/flood_handlabeled/": "splits",
}


def list_objects(prefix):
    items, token = [], None
    while True:
        q = {"prefix": prefix, "fields": "items(name,size),nextPageToken", "maxResults": "1000"}
        if token:
            q["pageToken"] = token
        url = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?{urllib.parse.urlencode(q)}"
        with urllib.request.urlopen(url, timeout=60) as r:
            page = json.load(r)
        items += page.get("items", [])
        token = page.get("nextPageToken")
        if not token:
            return items


def fetch(item, folder):
    dest = ROOT / folder / Path(item["name"]).name
    if dest.exists() and dest.stat().st_size == int(item["size"]):
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://storage.googleapis.com/{BUCKET}/{urllib.parse.quote(item['name'])}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                f.write(r.read())
            tmp.rename(dest)
            return "ok"
        except Exception as exc:  # retry transient network errors
            if attempt == 3:
                return f"failed: {exc}"


def main():
    jobs = []
    for prefix, folder in PREFIXES.items():
        items = [i for i in list_objects(prefix) if not i["name"].endswith("/")]
        print(f"{folder}: {len(items)} files, {sum(int(i['size']) for i in items) / 1e6:.0f} MB")
        jobs += [(i, folder) for i in items]
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda j: fetch(*j), jobs))
    failed = [r for r in results if r.startswith("failed")]
    print(f"done: {results.count('ok')} downloaded, {results.count('skip')} already present, {len(failed)} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
