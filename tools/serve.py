"""Static dev server for IceFjord, plus POST /save?name=xxx to store a canvas capture in shots/.

    python tools/serve.py [port]
"""
import base64
import http.server
import os
import re
import sys
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOTS = os.path.join(ROOT, "shots")
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 5261


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        u = urlparse(self.path)
        name = parse_qs(u.query).get("name", [""])[0]
        if u.path != "/save" or not re.fullmatch(r"[a-z0-9_-]{1,64}", name):
            self.send_error(400)
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("ascii")
        data = base64.b64decode(body.split(",", 1)[1])
        os.makedirs(SHOTS, exist_ok=True)
        with open(os.path.join(SHOTS, name + ".jpg"), "wb") as f:
            f.write(data)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, fmt, *args):
        if "POST" in (args[0] if args else ""):
            super().log_message(fmt, *args)


http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
