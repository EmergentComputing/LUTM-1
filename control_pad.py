"""Local browser control pad for live LUTM-1 island-GA training."""

from __future__ import annotations

import argparse
from copy import deepcopy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any
from urllib.parse import parse_qs, urlparse
import webbrowser

from control_pad_backend import (
    DEFAULT_DEFINITION,
    LiveSettings,
    LiveTrainingSession,
    list_saved_runs,
    parse_live_settings,
    parse_new_run_payload,
    suggested_run_id,
    validate_run_id,
)
from taichi_backend import initialize_taichi_cuda


ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "control_pad.html"


class ControlPadController:
    """Coordinate the HTTP threads with one main-thread CUDA training loop."""

    def __init__(self, runs_root: Path) -> None:
        self.runs_root = runs_root.resolve()
        self.condition = threading.Condition(threading.RLock())
        self.status = "idle"
        self.session: LiveTrainingSession | None = None
        self.snapshot: dict[str, Any] | None = None
        self.pending_start: tuple[str, Any] | None = None
        self.pending_settings: LiveSettings | None = None
        self.pause_requested = False
        self.stop_requested = False
        self.save_requested = False
        self.last_error: str | None = None
        self.last_message = "Ready"
        self.shutting_down = False

    def _refresh_snapshot(self) -> None:
        self.snapshot = (
            None if self.session is None else self.session.public_state()
        )

    def state(self) -> dict[str, Any]:
        with self.condition:
            return {
                "status": self.status,
                "message": self.last_message,
                "error": self.last_error,
                "session": deepcopy(self.snapshot),
                "pending_settings": (
                    None
                    if self.pending_settings is None
                    else self.pending_settings.to_json()
                ),
            }

    def defaults(self) -> dict[str, Any]:
        result = deepcopy(DEFAULT_DEFINITION)
        base = suggested_run_id()
        run_id = base
        suffix = 2
        while LiveTrainingSession.checkpoint_path(
            self.runs_root, run_id
        ).exists():
            run_id = f"{base}-{suffix}"
            suffix += 1
        result["run_id"] = run_id
        return result

    def request_start(self, payload: dict[str, Any]) -> None:
        definition, settings = parse_new_run_payload(payload)
        checkpoint = LiveTrainingSession.checkpoint_path(
            self.runs_root, definition.run_id
        )
        if checkpoint.exists():
            raise FileExistsError(
                f"run {definition.run_id!r} already has a checkpoint; "
                "load it or choose a different run ID"
            )
        with self.condition:
            if self.status in {"starting", "loading", "running", "pausing",
                               "stopping"}:
                raise RuntimeError("another run is currently active")
            self.pending_start = ("new", (definition, settings))
            self.pending_settings = None
            self.status = "starting"
            self.last_error = None
            self.last_message = f"Starting {definition.run_id}"
            self.condition.notify_all()

    def request_load(self, run_id: object) -> None:
        validated = validate_run_id(run_id)
        with self.condition:
            if self.status in {"starting", "loading", "running", "pausing",
                               "stopping"}:
                raise RuntimeError("another run is currently active")
            self.pending_start = ("load", validated)
            self.pending_settings = None
            self.status = "loading"
            self.last_error = None
            self.last_message = f"Loading {validated}"
            self.condition.notify_all()

    def request_settings(self, payload: dict[str, Any]) -> int:
        with self.condition:
            if self.session is None:
                raise RuntimeError("no run is loaded")
            settings = parse_live_settings(payload, self.session.definition)
            self.pending_settings = settings
            effective = self.session.generation + 1
            self.last_message = (
                f"Settings queued for generation {effective:,}"
            )
            self.condition.notify_all()
            return effective

    def request_pause(self) -> None:
        with self.condition:
            if self.status != "running":
                raise RuntimeError("training is not running")
            self.pause_requested = True
            self.status = "pausing"
            self.last_message = "Pausing after the active generation"
            self.condition.notify_all()

    def request_continue(self) -> None:
        with self.condition:
            if self.session is None:
                raise RuntimeError("no run is loaded")
            if self.status not in {"paused", "stopped", "solved"}:
                raise RuntimeError("the run is not paused or stopped")
            self.pause_requested = False
            self.stop_requested = False
            self.status = "running"
            self.last_error = None
            self.last_message = "Training"
            self.condition.notify_all()

    def request_stop(self) -> None:
        with self.condition:
            if self.status not in {"running", "pausing"}:
                raise RuntimeError("training is not running")
            self.stop_requested = True
            self.pause_requested = False
            self.status = "stopping"
            self.last_message = "Stopping and saving after the active generation"
            self.condition.notify_all()

    def request_new_run(self) -> None:
        with self.condition:
            if self.status != "stopped":
                raise RuntimeError(
                    "a new run can be configured only after Stop & save"
                )
            self.session = None
            self.snapshot = None
            self.pending_start = None
            self.pending_settings = None
            self.pause_requested = False
            self.stop_requested = False
            self.save_requested = False
            self.status = "idle"
            self.last_error = None
            self.last_message = "Ready for a new run"

    def request_save(self) -> None:
        with self.condition:
            if self.session is None:
                raise RuntimeError("no run is loaded")
            self.save_requested = True
            self.last_message = "Checkpoint requested"
            self.condition.notify_all()

    def history(self, start: int, islands: list[int]) -> dict[str, Any]:
        with self.condition:
            session = self.session
        if session is None:
            return {
                "start": start,
                "records": [],
                "diagnostics": [],
                "islands": {},
                "total_records": 0,
            }
        return session.history_payload(start, islands)

    def saved_runs(self) -> list[dict[str, Any]]:
        return list_saved_runs(self.runs_root)

    def _save(self, reason: str) -> None:
        if self.session is None:
            raise RuntimeError("no session is available to save")
        path = self.session.save_checkpoint(self.runs_root, reason=reason)
        with self.condition:
            self.save_requested = False
            self._refresh_snapshot()
            self.last_message = (
                f"Saved generation {self.session.generation:,} to {path}"
            )

    def _apply_pending_settings(self) -> None:
        with self.condition:
            settings = self.pending_settings
            self.pending_settings = None
        if settings is not None:
            if self.session is None:
                raise RuntimeError("no session is available")
            self.session.apply_settings(settings)
            with self.condition:
                self._refresh_snapshot()
                self.last_message = (
                    f"Settings active for generation "
                    f"{self.session.generation + 1:,}"
                )

    def _finish_initialization(self, session: LiveTrainingSession) -> None:
        self.session = session
        with self.condition:
            self._refresh_snapshot()
            if session.should_stop_exact():
                self.status = "solved"
                self.last_message = "Exact program found"
            elif session.reached_generation_target():
                self.status = "paused"
                self.last_message = "Generation target reached"
            else:
                self.status = "running"
                self.last_message = "Training"
        if self.status in {"solved", "paused"}:
            self._save(self.status)

    def _initialize_request(self, request: tuple[str, Any]) -> None:
        mode, value = request
        if mode == "new":
            definition, settings = value
            session = LiveTrainingSession.create(definition, settings)
        elif mode == "load":
            session = LiveTrainingSession.load_checkpoint(
                self.runs_root, value
            )
        else:
            raise RuntimeError(f"unknown start mode: {mode}")
        self._finish_initialization(session)
        if mode == "load":
            with self.condition:
                self.status = "paused"
                self.last_message = (
                    f"Loaded {session.definition.run_id} at generation "
                    f"{session.generation:,}"
                )

    def _handle_boundary_requests(self) -> bool:
        """Handle controls. Return True when a generation may run."""

        self._apply_pending_settings()
        with self.condition:
            session = self.session
            pause = self.pause_requested
            stop = self.stop_requested
            save = self.save_requested
            status = self.status
        if session is None:
            return False
        if stop:
            self._save("stopped")
            with self.condition:
                self.stop_requested = False
                self.status = "stopped"
                self.last_message = "Stopped and saved"
            return False
        if pause:
            self._save("paused")
            with self.condition:
                self.pause_requested = False
                self.status = "paused"
                self.last_message = "Paused and saved"
            return False
        if save:
            self._save("manual")
        if status != "running":
            return False
        if session.should_stop_exact():
            self._save("solved")
            with self.condition:
                self.status = "solved"
                self.last_message = "Exact program found"
            return False
        if session.reached_generation_target():
            self._save("generation-target")
            with self.condition:
                self.status = "paused"
                self.last_message = "Generation target reached"
            return False
        return True

    def run_forever(self) -> None:
        """Run on the process main thread so all Taichi work stays there."""

        while True:
            with self.condition:
                while (
                    not self.shutting_down
                    and self.pending_start is None
                    and self.status not in {"running", "pausing", "stopping"}
                    and not self.save_requested
                ):
                    self.condition.wait()
                if self.shutting_down:
                    return
                request = self.pending_start
                self.pending_start = None
            try:
                if request is not None:
                    self._initialize_request(request)
                    continue
                if not self._handle_boundary_requests():
                    continue
                if self.session is None:
                    raise RuntimeError("training session disappeared")
                record = self.session.step()
                with self.condition:
                    self._refresh_snapshot()
                    self.last_message = (
                        f"Generation {self.session.generation:,}: "
                        f"best {record['current_best']['fitness']:.6f}"
                    )
                if (
                    self.session.generation
                    % self.session.settings.checkpoint_interval
                    == 0
                ):
                    self._save("periodic")
            except Exception as error:
                with self.condition:
                    self.status = "error"
                    self.last_error = f"{type(error).__name__}: {error}"
                    self.last_message = "Training stopped with an error"
                    self.pause_requested = False
                    self.stop_requested = False
                    self.save_requested = False
                    if self.session is not None:
                        self._refresh_snapshot()

    def shutdown(self) -> None:
        with self.condition:
            self.shutting_down = True
            self.condition.notify_all()


class ControlPadHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        controller: ControlPadController,
    ) -> None:
        self.controller = controller
        super().__init__(address, ControlPadRequestHandler)


class ControlPadRequestHandler(BaseHTTPRequestHandler):
    server: ControlPadHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        if self.command != "GET" or self.path == "/":
            super().log_message(format, *args)

    def _json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _error(self, error: Exception) -> None:
        status = (
            HTTPStatus.NOT_FOUND
            if isinstance(error, FileNotFoundError)
            else HTTPStatus.CONFLICT
            if isinstance(error, (FileExistsError, RuntimeError))
            else HTTPStatus.BAD_REQUEST
        )
        self._json(
            {"error": f"{type(error).__name__}: {error}"},
            status,
        )

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("request body is required")
        length = int(raw_length)
        if not 0 < length <= 1_000_000:
            raise ValueError("request body size is invalid")
        parsed = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(parsed, dict):
            raise TypeError("request body must be a JSON object")
        return parsed

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = FRONTEND.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header(
                    "Content-Type", "text/html; charset=utf-8"
                )
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if parsed.path == "/api/defaults":
                self._json(self.server.controller.defaults())
                return
            if parsed.path == "/api/state":
                self._json(self.server.controller.state())
                return
            if parsed.path == "/api/runs":
                self._json({"runs": self.server.controller.saved_runs()})
                return
            if parsed.path == "/api/history":
                query = parse_qs(parsed.query)
                start = int(query.get("start", ["0"])[0])
                islands_text = query.get("islands", [""])[0]
                islands = (
                    []
                    if not islands_text
                    else [int(value) for value in islands_text.split(",")]
                )
                self._json(
                    self.server.controller.history(start, islands)
                )
                return
            self._json(
                {"error": "not found"}, HTTPStatus.NOT_FOUND
            )
        except Exception as error:
            self._error(error)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            path = urlparse(self.path).path
            if path == "/api/start":
                self.server.controller.request_start(payload)
                self._json({"accepted": True}, HTTPStatus.ACCEPTED)
            elif path == "/api/load":
                self.server.controller.request_load(payload.get("run_id"))
                self._json({"accepted": True}, HTTPStatus.ACCEPTED)
            elif path == "/api/settings":
                generation = self.server.controller.request_settings(payload)
                self._json(
                    {
                        "accepted": True,
                        "effective_generation": generation,
                    },
                    HTTPStatus.ACCEPTED,
                )
            elif path == "/api/pause":
                self.server.controller.request_pause()
                self._json({"accepted": True}, HTTPStatus.ACCEPTED)
            elif path == "/api/continue":
                self.server.controller.request_continue()
                self._json({"accepted": True}, HTTPStatus.ACCEPTED)
            elif path == "/api/stop":
                self.server.controller.request_stop()
                self._json({"accepted": True}, HTTPStatus.ACCEPTED)
            elif path == "/api/new":
                self.server.controller.request_new_run()
                self._json({"accepted": True}, HTTPStatus.ACCEPTED)
            elif path == "/api/save":
                self.server.controller.request_save()
                self._json({"accepted": True}, HTTPStatus.ACCEPTED)
            else:
                self._json(
                    {"error": "not found"}, HTTPStatus.NOT_FOUND
                )
        except Exception as error:
            self._error(error)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local LUTM-1 training control pad."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=ROOT / "runs",
    )
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.host not in {"127.0.0.1", "localhost"}:
        raise ValueError(
            "the control pad intentionally binds only to the local machine"
        )
    if not FRONTEND.is_file():
        raise FileNotFoundError(f"frontend not found: {FRONTEND}")

    initialize_taichi_cuda()
    controller = ControlPadController(arguments.runs_dir)
    server = ControlPadHTTPServer(
        (arguments.host, arguments.port),
        controller,
    )
    server_thread = threading.Thread(
        target=server.serve_forever,
        name="control-pad-http",
        daemon=True,
    )
    server_thread.start()
    url = f"http://{arguments.host}:{server.server_port}/"
    print(f"LUTM-1 control pad: {url}", flush=True)
    print("Press Ctrl+C here to close the server.", flush=True)
    if not arguments.no_browser:
        webbrowser.open(url)
    try:
        controller.run_forever()
    except KeyboardInterrupt:
        print("\nClosing control pad.", flush=True)
    finally:
        controller.shutdown()
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
