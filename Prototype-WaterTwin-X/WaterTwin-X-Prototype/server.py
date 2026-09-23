"""
WaterTwin-X prototype - local API server.

Uses only Python's standard library (http.server) so there's nothing extra
to install beyond scikit-learn/numpy, which are already needed for engine.py.

Run:  python3 server.py
Then open frontend/index.html in a browser (it talks to http://localhost:8765).
"""

import json
import threading
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from engine import WaterTwin, GRID_N, PARAMS

warnings.filterwarnings("ignore")  # quiet GP hyperparameter-bound convergence chatter

PORT = 8765
twin = WaterTwin(seed=7, n_initial=7)              # the "hero" twin - priority sampling
shadow_twin = WaterTwin(seed=7, n_initial=7)        # baseline - random sampling, for the live comparison chart
error_log = {"priority": [twin.total_error()], "random": [shadow_twin.total_error()]}
state_lock = threading.Lock()  # ThreadingHTTPServer means concurrent requests are possible -
                                # guard the shared twin/shadow_twin/error_log against races


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def do_OPTIONS(self):
        self._send_json({})

    def do_GET(self):
        try:
            if self.path == "/state":
                with state_lock:
                    self._send_json({"twin": twin.state(), "error_log": error_log})
            else:
                self._send_json({"error": "not found"}, status=404)
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    def do_POST(self):
        global twin, shadow_twin, error_log
        try:
            body = self._read_json_body()

            if self.path == "/step":
                with state_lock:
                    idx = twin.step("priority")
                    shadow_twin.step("random")
                    if idx is None:
                        self._send_json({"error": "grid fully sampled - nothing left to check", "twin": twin.state(), "error_log": error_log}, status=409)
                        return
                    error_log["priority"].append(twin.total_error())
                    error_log["random"].append(shadow_twin.total_error())
                    self._send_json({"twin": twin.state(), "error_log": error_log})

            elif self.path == "/manual":
                row, col, field, value = body.get("row"), body.get("col"), body.get("field"), body.get("value")
                if row is None or col is None or field is None or value is None:
                    self._send_json({"error": "row, col, field, value required"}, status=400)
                    return
                row, col, value = int(row), int(col), float(value)
                if not (0 <= row < GRID_N and 0 <= col < GRID_N):
                    self._send_json({"error": f"row/col must be within 0..{GRID_N-1}"}, status=400)
                    return
                if field not in PARAMS:
                    self._send_json({"error": f"field must be one of {list(PARAMS)}"}, status=400)
                    return
                with state_lock:
                    twin.manual_edit(row, col, field, value)
                    self._send_json({"twin": twin.state(), "error_log": error_log})

            elif self.path == "/reset":
                seed = int(body.get("seed", 7))
                with state_lock:
                    twin = WaterTwin(seed=seed, n_initial=7)
                    shadow_twin = WaterTwin(seed=seed, n_initial=7)
                    error_log = {"priority": [twin.total_error()], "random": [shadow_twin.total_error()]}
                    self._send_json({"twin": twin.state(), "error_log": error_log})

            else:
                self._send_json({"error": "not found"}, status=404)
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    def log_message(self, fmt, *args):
        print("[server]", fmt % args)


if __name__ == "__main__":
    print(f"WaterTwin-X backend running on http://localhost:{PORT}")
    print("Endpoints: GET /state | POST /step | POST /manual {row,col,value} | POST /reset {seed}")
    ThreadingHTTPServer(("localhost", PORT), Handler).serve_forever()

