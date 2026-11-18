"""The heavy work of the bot, in a process of its own that only lives while there is work.

The bot keeps nothing but the Telegram connection in memory. Reading price lists, Gemini, building output
files, screenshots and the brain all live in `pricebot.service`, which only the engine process imports.
The engine starts on the first job (or earlier, while a file is still downloading), runs jobs side by side
in threads, and quits after `config.ENGINE_IDLE_SECONDS` without work: its memory goes back to the system
with it, all of it. If it ever dies (out of memory on a huge file), only the jobs it was running fail and
the next job starts a fresh one.

Results come back through `light_loads`, which only accepts plain data and the bot's own light classes,
so the bot process can never end up importing numpy, OpenCV, PyMuPDF, Gemini or the brain by accident.

    analysis = await ENGINE.call("analyze", src, filename, workdir, progress, False, stage)

Callables among the top-level arguments (progress callbacks) are called back in the bot process.
"""
from __future__ import annotations

import asyncio
import io
import itertools
import logging
import os
import pickle
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from multiprocessing.connection import Client, Connection, Listener

from . import config

log = logging.getLogger("engine")

PROTOCOL = pickle.HIGHEST_PROTOCOL
START_TIMEOUT = 120                 # seconds the engine may take to start and connect
MAX_TRIES = 3                       # engines a job is handed to when they die before beginning it
# what the bot process accepts from the engine: plain data and the bot's own light modules
LIGHT_MODULES = {"builtins", "copyreg", "collections", "datetime", "decimal", "fractions", "zoneinfo", "_zoneinfo",
                 "pricebot.models", "pricebot.numfmt", "pricebot.commands", "pricebot.errors", "pricebot.ai",
                 "pricebot.shot.values", "pricebot.engine"}


class EngineError(RuntimeError):
    """The engine process stopped or could not start, or a job failed in an unexpected way."""


@dataclass
class _Callback:
    index: int


class _LightUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module in LIGHT_MODULES or module == "pathlib" or module.startswith("pathlib."):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"{module}.{name} is not light: it cannot go to the bot process")


def light_loads(data: bytes):
    return _LightUnpickler(io.BytesIO(data)).load()


# =================================================================== bot side ==

class _Call:
    __slots__ = ("loop", "future", "callbacks", "payload", "started", "tries")

    def __init__(self, loop: asyncio.AbstractEventLoop, future: asyncio.Future, callbacks: list, payload: bytes):
        self.loop, self.future, self.callbacks, self.payload = loop, future, callbacks, payload
        self.started = False        # the engine said it began the job: if it dies now, the job failed
        self.tries = 0


class Engine:
    def __init__(self, idle: float = config.ENGINE_IDLE_SECONDS):
        self.idle = idle
        self._lock = threading.RLock()
        self._conn: Connection | None = None
        self._proc: subprocess.Popen | None = None
        self._calls: dict[int, _Call] = {}
        self._ids = itertools.count(1)
        self._last = time.monotonic()
        self._starting = False
        self._watcher: threading.Thread | None = None

    # ----------------------------------------------------------------- public --
    def running(self) -> bool:
        with self._lock:
            return self._conn is not None

    def warm(self) -> None:
        """Start the engine in the background (a file is downloading: the engine is ready when the file is)."""
        with self._lock:
            self._last = time.monotonic()
            if self._conn is not None or self._starting:
                return
            self._starting = True
        threading.Thread(target=self._warm, name="engine-warm", daemon=True).start()

    async def call(self, name: str, *args, **kwargs):
        """Run `pricebot.service.<name>(*args, **kwargs)` in the engine and return its result."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        callbacks = [a for a in args if callable(a)]
        marked = tuple(_Callback(callbacks.index(a)) if callable(a) else a for a in args)
        payload = pickle.dumps((name, marked, kwargs), PROTOCOL)
        await asyncio.to_thread(self._submit, _Call(loop, future, callbacks, payload))
        return await future

    def stop(self) -> None:
        with self._lock:
            conn, proc = self._conn, self._proc
            self._conn = self._proc = None
            calls, self._calls = self._calls, {}
        self._fail(calls, EngineError("ربات در حال خاموش شدن است."))
        if conn is not None:
            _send_quit(conn)
        _close(conn, proc)

    # --------------------------------------------------------------- internal --
    def _warm(self) -> None:
        try:
            with self._lock:
                self._starting = False
                self._ensure()
        except Exception as exc:  # noqa: BLE001 - the job itself reports a failed start
            log.warning("engine warm-up failed: %s", exc)

    def _submit(self, call: _Call) -> None:
        while True:
            with self._lock:
                self._ensure()
                cid = next(self._ids)
                self._calls[cid] = call
                self._last = time.monotonic()
                call.tries += 1
                conn = self._conn
                try:
                    conn.send_bytes(pickle.dumps(("call", cid, call.payload), PROTOCOL))
                    return
                except (OSError, ValueError) as exc:
                    self._calls.pop(cid, None)
                    error = exc
            self._dead(conn)
            if call.tries >= MAX_TRIES:
                raise EngineError(f"engine connection lost: {error}")

    def _resubmit(self, calls: list[_Call]) -> None:
        for call in calls:
            try:
                self._submit(call)
            except Exception as exc:  # noqa: BLE001
                _resolve(call, False, exc if isinstance(exc, EngineError) else EngineError(str(exc)))

    def _ensure(self) -> None:
        """With the lock held: make sure an engine process is up and connected."""
        if self._conn is not None:
            if self._proc is not None and self._proc.poll() is None:
                return
            # it died between two jobs and the reader has not noticed yet: never hand it a new job
            self._dead(self._conn)
            if self._conn is not None:
                return
        if self._watcher is None:
            self._watcher = threading.Thread(target=self._watch, name="engine-idle", daemon=True)
            self._watcher.start()
        self._starting = True
        try:
            key = os.urandom(32)
            listener = Listener(("127.0.0.1", 0), authkey=key)
            try:
                env = {**os.environ, "PRICEBOT_ENGINE": f"{listener.address[1]}:{key.hex()}"}
                proc = subprocess.Popen([sys.executable, "-m", "pricebot.engine"], env=env,
                                        cwd=str(config.BASE_DIR))
                got: dict = {}

                def accept() -> None:
                    try:
                        got["conn"] = listener.accept()
                    except Exception as exc:  # noqa: BLE001
                        got["error"] = exc

                waiter = threading.Thread(target=accept, daemon=True)
                waiter.start()
                waiter.join(START_TIMEOUT)
            finally:
                listener.close()
            if "conn" not in got:
                proc.kill()
                raise EngineError(f"موتور پردازش بالا نیامد ({got.get('error', 'timeout')}).")
            self._conn, self._proc = got["conn"], proc
            self._last = time.monotonic()
            log.info("engine started (pid %s)", proc.pid)
            threading.Thread(target=self._read, args=(self._conn,), name="engine-read", daemon=True).start()
        finally:
            self._starting = False

    def _read(self, conn: Connection) -> None:
        while True:
            try:
                kind, cid, a, b = light_loads(conn.recv_bytes())
            except (EOFError, OSError):
                break
            except Exception:  # noqa: BLE001 - cannot happen with our own engine; never stop reading for it
                log.exception("unreadable message from the engine")
                continue
            if kind == "start":
                call = self._calls.get(cid)
                if call is not None:
                    call.started = True
            elif kind == "cb":
                call = self._calls.get(cid)
                if call is not None:
                    try:
                        call.callbacks[a](*light_loads(b))
                    except Exception:  # noqa: BLE001 - a progress callback never stops the job
                        log.debug("callback failed", exc_info=True)
            elif kind == "done":
                with self._lock:
                    call = self._calls.pop(cid, None)
                    self._last = time.monotonic()
                if call is None:
                    continue
                try:
                    _resolve(call, a, light_loads(b))
                except Exception as exc:  # noqa: BLE001 - a result the bot must not load fails that job only
                    log.error("result of job %s cannot be loaded here: %s", cid, exc)
                    _resolve(call, False, EngineError(str(exc)))
        self._dead(conn)

    def _dead(self, conn: Connection | None) -> None:
        """`conn` is gone. Jobs the engine had begun fail; jobs it never began go to a fresh engine."""
        with self._lock:
            if conn is None or conn is not self._conn:
                return
            proc = self._proc
            self._conn = self._proc = None
            calls, self._calls = self._calls, {}
        running = {cid: c for cid, c in calls.items() if c.started or c.tries >= MAX_TRIES}
        waiting = [c for cid, c in calls.items() if cid not in running]
        if running:
            log.warning("engine stopped with %d job(s) running", len(running))
        self._fail(running, EngineError("پردازش نیمه‌کاره ماند (موتور پردازش متوقف شد)؛ دوباره امتحان کن."))
        if waiting:
            threading.Thread(target=self._resubmit, args=(waiting,), daemon=True).start()
        threading.Thread(target=_close, args=(None, proc), daemon=True).start()

    @staticmethod
    def _fail(calls: dict[int, _Call], exc: Exception) -> None:
        for call in calls.values():
            _resolve(call, False, exc)

    def _watch(self) -> None:
        """Quit the engine once it has been idle long enough: all of its memory goes back to the system."""
        while True:
            time.sleep(2)
            with self._lock:
                if (self._conn is None or self._calls or self._starting
                        or time.monotonic() - self._last < self.idle):
                    continue
                conn, proc = self._conn, self._proc
                self._conn = self._proc = None
            log.info("engine idle for %ds: stopped to free its memory", self.idle)
            _send_quit(conn)
            threading.Thread(target=_close, args=(conn, proc), daemon=True).start()


def _send_quit(conn: Connection) -> None:
    try:
        conn.send_bytes(pickle.dumps(("quit", 0, None, None), PROTOCOL))
    except (OSError, ValueError):
        pass


def _resolve(call: _Call, ok: bool, value) -> None:
    def setter() -> None:
        if call.future.done():
            return
        if ok:
            call.future.set_result(value)
        else:
            call.future.set_exception(value if isinstance(value, BaseException) else EngineError(str(value)))

    try:
        call.loop.call_soon_threadsafe(setter)
    except RuntimeError:   # the event loop is closed (the bot is shutting down)
        pass


def _close(conn: Connection | None, proc: subprocess.Popen | None) -> None:
    if proc is not None:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    if conn is not None:
        try:
            conn.close()
        except OSError:
            pass


ENGINE = Engine()


# ================================================================ engine side ==

def _light_error(exc: BaseException) -> BaseException:
    """An error the bot can load: the bot's own errors as they are, anything else as text."""
    if type(exc).__module__ in LIGHT_MODULES:
        try:
            light_loads(pickle.dumps(exc, PROTOCOL))
            return exc
        except Exception:  # noqa: BLE001
            pass
    return EngineError(f"{type(exc).__name__}: {exc}")


def serve(port: int, key: bytes) -> None:
    from concurrent.futures import ThreadPoolExecutor

    conn = Client(("127.0.0.1", port), authkey=key)
    lock = threading.Lock()

    def send(kind: str, cid: int, a, value) -> None:
        msg = pickle.dumps((kind, cid, a, pickle.dumps(value, PROTOCOL)), PROTOCOL)
        with lock:
            conn.send_bytes(msg)

    def run(cid: int, payload: bytes) -> None:
        name = "?"
        try:
            send("start", cid, None, None)      # from now on, if this process dies, the job failed
            name, args, kwargs = pickle.loads(payload)
            args = [(lambda *a, _i=x.index: send("cb", cid, _i, a)) if isinstance(x, _Callback) else x
                    for x in args]
            from . import service
            value = getattr(service, name)(*args, **kwargs)
            data = pickle.dumps(value, PROTOCOL)
            light_loads(data)                   # what the bot could not load is a bug: report it, never send it
        except Exception as exc:  # noqa: BLE001 - every failure goes back to the job that asked for it
            if type(exc).__module__ in LIGHT_MODULES:
                log.info("job %s failed: %s", name, exc)
            else:
                log.exception("job %s failed", name)
            send("done", cid, False, _light_error(exc))
            return
        msg = pickle.dumps(("done", cid, True, data), PROTOCOL)
        with lock:
            conn.send_bytes(msg)

    # import the heavy parts while the first job is still on its way
    threading.Thread(target=lambda: __import__("pricebot.service"), daemon=True).start()
    pool = ThreadPoolExecutor(max_workers=config.ENGINE_THREADS, thread_name_prefix="job")
    while True:
        try:
            msg = pickle.loads(conn.recv_bytes())
        except (EOFError, OSError):
            break
        if msg[0] == "quit":
            break
        pool.submit(run, msg[1], msg[2])
    pool.shutdown(wait=True)


if __name__ == "__main__":
    logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(levelname)s engine: %(message)s")
    for _noisy in ("httpx", "httpcore", "google_genai", "urllib3"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)
    from pricebot import engine as _module    # run under the real module name: its classes match the bot's
    _port, _key = os.environ["PRICEBOT_ENGINE"].split(":")
    _module.serve(int(_port), bytes.fromhex(_key))
