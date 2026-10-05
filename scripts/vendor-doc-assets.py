"""Vendor pinned npm documentation assets. Run only when maintaining the source."""
import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile
from urllib.request import urlopen

DEST = Path(__file__).resolve().parent.parent / "static" / "docs"
REDOC_LOGO_URL = "https://cdn.redoc.ly/redoc/logo-mini.svg"
REDOC_LOGO_SHA256 = "1dab4315f8de2177b2b2726b29e84a26314c69cd27d692228947c0058793d5a7"
PACKAGES = {
    "swagger-ui-dist": ("5.33.1", {
        "package/swagger-ui-bundle.js": "swagger-ui-bundle.js",
        "package/swagger-ui.css": "swagger-ui.css",
        "package/LICENSE": "swagger-ui-LICENSE",
    }),
    "redoc": ("2.5.4", {
        "package/bundles/redoc.standalone.js": "redoc.standalone.js",
        "package/LICENSE": "redoc-LICENSE",
    }),
}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = {"packages": {}, "files": {}}
    with urlopen(REDOC_LOGO_URL, timeout=60) as response:
        logo = response.read()
    if b"<svg" not in logo or b"</svg>" not in logo:
        raise RuntimeError("Unexpected ReDoc logo format")
    logo_digest = hashlib.sha256(logo).hexdigest()
    if logo_digest != REDOC_LOGO_SHA256:
        raise RuntimeError("ReDoc logo changed; review and pin its new digest before updating")
    (DEST / "redoc-logo-mini.svg").write_bytes(logo)
    manifest["files"]["redoc-logo-mini.svg"] = logo_digest
    manifest["redoc_logo"] = {"source": REDOC_LOGO_URL, "sha256": logo_digest,
                              "patch": "Embed the upstream footer logo as a data URI in redoc.standalone.js"}
    for name, (version, files) in PACKAGES.items():
        metadata_url = f"https://registry.npmjs.org/{name}/{version}"
        with urlopen(metadata_url, timeout=60) as response:
            metadata = json.load(response)
        with urlopen(metadata["dist"]["tarball"], timeout=120) as response:
            archive = response.read()
        algorithm, digest = metadata["dist"]["integrity"].split("-", 1)
        if hashlib.new(algorithm, archive).digest() != base64.b64decode(digest):
            raise RuntimeError(f"Integrity check failed for {name}@{version}")
        manifest["packages"][name] = {
            "version": version, "license": metadata["license"],
            "metadata_url": metadata_url, "tarball": metadata["dist"]["tarball"],
            "integrity": metadata["dist"]["integrity"],
        }
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as package:
            for member, filename in files.items():
                payload = package.extractfile(member).read()
                if filename == "redoc.standalone.js":
                    if payload.count(REDOC_LOGO_URL.encode()) != 1:
                        raise RuntimeError("ReDoc footer logo location changed; review before updating")
                    embedded = b"data:image/svg+xml;base64," + base64.b64encode(logo)
                    payload = payload.replace(REDOC_LOGO_URL.encode(), embedded)
                (DEST / filename).write_bytes(payload)
                manifest["files"][filename] = hashlib.sha256(payload).hexdigest()
        print(f"Vendored {name}@{version}; npm archive integrity verified.")
    manifest["files"]["favicon.svg"] = hashlib.sha256((DEST / "favicon.svg").read_bytes()).hexdigest()
    (DEST / "assets-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
