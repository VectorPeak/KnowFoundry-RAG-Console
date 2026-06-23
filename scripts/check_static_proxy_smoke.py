"""Smoke-check homepage and proxied static assets."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class ExpectedResponse:
    name: str
    path: str
    content_type_fragment: str | None = None


CHECKS = (
    ExpectedResponse("home", ""),
    ExpectedResponse("css", "/static/css/app.css", "text/css"),
    ExpectedResponse("js", "/static/js/app.js", "javascript"),
)


def build_url(base_url: str, path: str) -> str:
    """Append a path to a base URL that may already include a public prefix."""
    return base_url.rstrip("/") + path


def fetch_head_or_get(url: str) -> tuple[int, str]:
    req = urlrequest.Request(url, method="HEAD")
    try:
        with urlrequest.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
            return response.status, response.headers.get("Content-Type", "")
    except HTTPError as exc:
        if exc.code != 405:
            return exc.code, exc.headers.get("Content-Type", "")
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"{url}: {exc}") from exc

    req = urlrequest.Request(url, method="GET")
    try:
        with urlrequest.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
            response.read(1)
            return response.status, response.headers.get("Content-Type", "")
    except HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type", "")
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"{url}: {exc}") from exc


def check_url(base_url: str, expected: ExpectedResponse) -> dict[str, object]:
    url = build_url(base_url, expected.path)
    status, content_type = fetch_head_or_get(url)
    ok = status == 200
    if expected.content_type_fragment:
        ok = ok and expected.content_type_fragment.lower() in content_type.lower()

    return {
        "name": expected.name,
        "url": url,
        "status": status,
        "content_type": content_type,
        "expected_content_type_fragment": expected.content_type_fragment,
        "ok": ok,
    }


def run_checks(base_url: str) -> dict[str, object]:
    results = [check_url(base_url, expected) for expected in CHECKS]
    return {
        "report_type": "static_proxy_smoke",
        "ok": all(bool(result["ok"]) for result in results),
        "base_url": base_url,
        "checks": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check that a base URL serves the homepage and static CSS/JS assets. "
            "The base URL may include a public path prefix."
        )
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"Default: {DEFAULT_BASE_URL}")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        payload = run_checks(args.base_url)
    except RuntimeError as exc:
        print(f"Static proxy smoke check failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
