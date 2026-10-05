import json
import os
from urllib.request import urlopen

with urlopen(f"http://127.0.0.1:{os.environ.get('CLEF_PORT', '8000')}/health", timeout=4) as response:
    body = json.load(response)
    if response.status != 200 or not body.get("ready") or body.get("service") != "clef-flash-api":
        raise SystemExit(1)
