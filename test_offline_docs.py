"""Documentation packaging tests; no model loading or external network access."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import unittest
from urllib.parse import urlsplit

from fastapi.testclient import TestClient
import app as service


class Resources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if (tag, name) in {("script", "src"), ("link", "href"), ("redoc", "spec-url")}:
                self.urls.append(value)


class OfflineDocsTests(unittest.TestCase):
    def test_assets_match_manifest(self):
        directory = service.HERE / "static" / "docs"
        manifest = json.loads((directory / "assets-manifest.json").read_text())
        self.assertGreaterEqual(len(manifest["files"]), 6)
        for name, expected in manifest["files"].items():
            self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), expected, name)
        redoc_bundle = (directory / "redoc.standalone.js").read_text()
        self.assertNotIn("https://cdn.redoc.ly/redoc/logo-mini.svg", redoc_bundle)
        self.assertIn("data:image/svg+xml;base64,", redoc_bundle)

    def test_docs_use_only_local_resources_including_proxy_prefix(self):
        # Do not enter the client lifespan: these tests deliberately avoid loading the GPU model.
        for prefix in ("", "/clef-api"):
            with self.subTest(prefix=prefix):
                client = TestClient(service.app, root_path=prefix)
                try:
                    for endpoint in ("/docs", "/redoc"):
                        response = client.get(prefix + endpoint)
                        self.assertEqual(response.status_code, 200)
                        parser = Resources()
                        parser.feed(response.text)
                        self.assertGreaterEqual(len(parser.urls), 3)
                        for url in parser.urls:
                            self.assertFalse(urlsplit(url).scheme or urlsplit(url).netloc, url)
                            self.assertTrue(url.startswith(prefix + "/"), url)
                            asset = client.get(url)
                            self.assertEqual(asset.status_code, 200, url)
                            self.assertGreater(len(asset.content), 100, url)
                        self.assertIn(prefix + "/openapi.json", response.text)
                        self.assertNotIn("fonts.googleapis.com", response.text)
                        self.assertNotIn("cdn.jsdelivr.net", response.text)
                    self.assertIn('"validatorUrl": null', client.get(prefix + "/docs").text)
                    self.assertEqual(client.get(prefix + "/docs/oauth2-redirect").status_code, 200)
                finally:
                    client.close()

    def test_documentation_does_not_pollute_api_schema(self):
        schema = service.app.openapi()
        self.assertIn("/v1/systemone", schema["paths"])
        self.assertNotIn("/docs", schema["paths"])
        self.assertNotIn("/redoc", schema["paths"])
        self.assertNotIn("/docs/oauth2-redirect", schema["paths"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
