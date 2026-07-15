#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Small OpenAI-compatible endpoint for Dopamine GRM integration smoke tests."""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeGRMHandler(BaseHTTPRequestHandler):
    scores = ("+35%", "+55%", "-25%")
    request_count = 0
    lock = threading.Lock()

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            self.send_error(400, "messages must be a non-empty list")
            return

        with self.lock:
            score = self.scores[self.request_count % len(self.scores)]
            type(self).request_count += 1
        response = json.dumps(
            {"choices": [{"message": {"content": f"<score>{score}</score>"}}]}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format_string, *args):
        print(f"[fake-grm] {format_string % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), FakeGRMHandler)
    print(
        f"Fake Dopamine GRM listening on http://{args.host}:{args.port}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
