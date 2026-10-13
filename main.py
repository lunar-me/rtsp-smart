"""
rtsp-smart — real-time object watcher (person, dog, ...) for an RTSP (H.264) camera.

A background thread reads the RTSP stream and keeps only the most recent
frame (so latency never accumulates). Each analysed frame is first cropped
to the ACTIVE_WINDOW (a percent-of-frame-size region, default: the full
frame) and run through YOLO restricted to the TRIGGER_CLASS classes
(default: person), and while any trigger object is visible with confidence
>= CONFIDENCE_THRESHOLD one screenshot is saved every
SAVE_INTERVAL_SECONDS into SAVE_DIR (relative or absolute), in a per-day
subfolder: "<SAVE_DIR>/<YYYY-MM-DD>/<RTSP_NAME> YYYY-MM-DD HH-MM-SS.jpg".
After a screenshot, no frame is analysed for SAVE_INTERVAL_SECONDS
(no save is possible anyway). If the stream drops, the service
reconnects automatically (the first second of frames is discarded —
a decoder that joins mid-GOP cannot produce intact frames before the
first keyframe). Frames the decoder flags as corrupt (glitched H.264 data)
are discarded and counted — they never reach detection.
Every STAT_INTERVAL_SECONDS a status line (received, analysed, dropped,
corrupt, device, screenshots) proves it is alive.

Usage:
    python main.py                 run the service (Ctrl+C to stop)
    python main.py --selftest     verify the detection pipeline without a camera
    python main.py --limit 30     run for 30 seconds, then stop (for testing)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

try:                                # Windows console API (used by the hotkey)
    import msvcrt
except ImportError:                 # not Windows — the hotkey is unavailable
    msvcrt = None

try:
    import av  # PyAV — RTSP/H.264 decoding that reports corruption per frame
except ImportError:            # the service refuses to run without it (see run_service)
    av = None

import cv2
import torch
from dotenv import load_dotenv
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
TRIGGER_CLASS_ID = 0         # COCO id of "person" — the default trigger class
STALE_FRAME_SECONDS = 2.0    # frames older than this are skipped (stream hiccup)
CONNECT_WARMUP_SECONDS = 1.0 # after (re)connect, frames are discarded this long (mid-GOP join)
RATE_MEASURE_SECONDS = 2.0   # the true stream fps is measured over this long, after the warm-up
SELFTEST_IMAGE = "https://ultralytics.com/images/bus.jpg"

log = logging.getLogger("rtsp-smart")


# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #

def sanitize_env() -> None:
    """Remove UTF-8 BOM artifacts from env var names (e.g. .env saved by Notepad)."""
    for key in list(os.environ):
        if key.startswith("\ufeff"):
            os.environ[key.lstrip("\ufeff")] = os.environ[key]


def sanitize_name(name: str) -> str:
    """Make a configured name safe to use in file names (Windows and Linux)."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip().rstrip(". ")
    return cleaned or "RTSP"


def parse_active_window(raw: str) -> tuple[float, float, float, float]:
    """Parse ACTIVE_WINDOW = x1,y1,x2,y2 (percent of frame size, top-left origin).

    "0,0,100,100" (the default) means the full frame. Raises ValueError with
    a clear message if the value is malformed or the window is empty.
    """
    parts = [p for p in re.split(r"[,;\s]+", raw.strip()) if p]
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"ACTIVE_WINDOW must be 4 numbers (x1,y1,x2,y2 in %): {raw!r}") from None
    if len(vals) != 4 or not (0.0 <= vals[0] < vals[2] <= 100.0 and 0.0 <= vals[1] < vals[3] <= 100.0):
        raise ValueError("ACTIVE_WINDOW must be x1,y1,x2,y2 in % with 0 <= x1 < x2 <= 100 "
                         f"and 0 <= y1 < y2 <= 100 (from the top-left corner), got: {raw!r}")
    return vals[0], vals[1], vals[2], vals[3]


def load_config() -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    sanitize_env()

    return {
        # RTSP_STREAM is preferred; legacy RTSP_STRAM (typo) and RTSP_URL work too.
        "rtsp_url": os.getenv("RTSP_STREAM") or os.getenv("RTSP_STRAM") or os.getenv("RTSP_URL"),
        "stream_name": sanitize_name(os.getenv("RTSP_NAME", "RTSP")),
        "transport": os.getenv("RTSP_TRANSPORT", "tcp").lower(),
        "model": os.getenv("MODEL", "yolo26n.pt"),
        "device": os.getenv("DEVICE", "auto").lower(),
        "trigger_classes": [c for c in re.split(r"[,;\s]+", (os.getenv("TRIGGER_CLASS") or "person").lower()) if c],
        "conf_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.90")),
        "active_window": parse_active_window(os.getenv("ACTIVE_WINDOW") or "0,0,100,100"),
        "save_interval": max(float(os.getenv("SAVE_INTERVAL_SECONDS", "1.0")), 0.1),
        "detect_fps": float(os.getenv("DETECT_FPS", "5")),
        "stat_interval": max(float(os.getenv("STAT_INTERVAL_SECONDS", "60")), 1.0),
        "save_dir": Path(os.getenv("SAVE_DIR", "screenshots")),
        "draw_boxes": os.getenv("DRAW_BOXES", "false").lower() in ("1", "true", "yes"),
        "log_file": Path(os.getenv("LOG_FILE", "logs/service.log")),
    }


def resolve_path(path: Path) -> Path:
    """Resolve a possibly relative configured path against the project root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def setup_logging(log_file: Path) -> None:
    log_file = resolve_path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    log.addHandler(console)

    file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(fmt)
    log.addHandler(file_handler)


def mask_credentials(url: str) -> str:
    """Hide user:password in RTSP URLs before writing them to logs."""
    return re.sub(r"//[^/@]+@", "//***@", url)

# --------------------------------------------------------------------------- #
# Device selection (CPU / NVIDIA GPU)                                         #
# --------------------------------------------------------------------------- #

def nvidia_gpu_name() -> str | None:
    """Name of the installed NVIDIA GPU, or None if no driver/GPU is found."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def select_device(preference: str = "auto") -> str:
    """Resolve the inference device ('cpu' or 'cuda:N'), logging the choice.

    'auto' uses the NVIDIA GPU when the installed torch build supports it.
    If an NVIDIA GPU exists but torch is CPU-only, a warning explains how
    to enable GPU acceleration.
    """
    if preference not in ("auto", ""):
        if preference.startswith("cuda") and torch.cuda.is_available():
            return preference                      # explicit cuda / cuda:1 etc.
        if preference != "cpu":
            log.warning("DEVICE=%s is not available here — using CPU.", preference)
        return "cpu"

    if torch.cuda.is_available():
        log.info("NVIDIA GPU available (%s) — using CUDA.", torch.cuda.get_device_name(0))
        return "cuda:0"

    gpu = nvidia_gpu_name()
    if gpu:
        log.warning(
            "NVIDIA GPU found (%s) but this PyTorch build is CPU-only. Enable the GPU with:\n"
            "    python -m pip install -r requirements-gpu.txt\n"
            "then restart the service. Using CPU for now.", gpu,
        )
    else:
        log.info("No NVIDIA GPU in use — running on CPU.")
    return "cpu"



def rtsp_options(transport: str) -> dict:
    """FFmpeg options for opening the RTSP stream with PyAV: force the
    transport and a 5 s socket timeout so a dead camera cannot hang us."""
    return {"rtsp_transport": transport, "timeout": "5000000"}


def claimed_fps(stream) -> float:
    """The frame rate the stream metadata claims (a Fraction, or None).

    Never trusted for anything — only reported next to the measured rate,
    because cameras often misreport it (sometimes by half).
    """
    for rate in (stream.average_rate, stream.codec_context.framerate):
        try:
            fps = float(rate)
        except (TypeError, ZeroDivisionError):
            continue
        if fps > 0.0:
            return fps
    return 0.0


# --------------------------------------------------------------------------- #
# RTSP reader thread                                                           #
# --------------------------------------------------------------------------- #

class RTSPReader(threading.Thread):
    """Reads an RTSP stream in a background thread, keeping only the latest frame.

    Because detection always runs on the newest frame, internal FFmpeg
    buffering can never make latency accumulate. Reconnects automatically when
    the stream drops (capped exponential backoff). The first
    CONNECT_WARMUP_SECONDS after each (re)connect are discarded: the decoder
    joins mid-GOP and cannot produce intact frames before the first keyframe.
    The true frame rate is then measured over the next RATE_MEASURE_SECONDS and
    logged — the camera's metadata claim is often wrong (sometimes half the rate).

    Corrupt frames never reach detection: when the camera sends glitched
    H.264 data, FFmpeg conceals it into smeared frames — the reader (PyAV)
    sees the corruption flag, discards the frame and counts it in the status
    line. OpenCV hides this: it reports such frames as normal.
    """

    def __init__(self, url: str, transport: str = "tcp") -> None:
        super().__init__(daemon=True, name="rtsp-reader")
        self.url = url
        self.transport = transport
        self._lock = threading.Lock()
        self._frame = None
        self._frame_ts = 0.0
        self._frames_read = 0
        self._corrupt_skipped = 0
        self._stop = threading.Event()

    def run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            opened = False
            container = None
            try:
                container = av.open(self.url, options=rtsp_options(self.transport))
            except Exception as exc:
                log.warning("Cannot open stream (%s), retrying in %.0f s ...", exc, backoff)
            if container is not None:
                opened = True
                stored = 0
                try:
                    stored = self._read_stream(container)
                except Exception as exc:
                    log.warning("Stream read failed (%s), dropping it.", exc)
                finally:
                    container.close()
                if stored:
                    backoff = 1.0        # a productive connection: reconnect fast next time

            if self._stop.is_set():
                break
            if opened:
                log.warning("Stream dropped, reconnecting in %.0f s ...", backoff)
            if self._stop.wait(backoff):
                break
            backoff = min(backoff * 2.0, 15.0)

    def _read_stream(self, container) -> int:
        """Demux and decode the stream until it ends; returns the frames stored.

        Raises on a stream drop (network error, timeout, end of stream) so the
        caller can reconnect; damaged data never raises here — corrupt packets
        and frames are discarded and counted instead.
        """
        stream = container.streams.video[0]
        stream.codec_context.thread_type = "AUTO"    # multi-threaded decode keeps up
        stream.codec_context.thread_count = 0        # 0 = choose automatically
        log.info(
            "Stream connected (%dx%d)",
            stream.codec_context.width, stream.codec_context.height,
        )
        claimed = claimed_fps(stream)
        t_connect = time.time()
        warmup_until = t_connect + CONNECT_WARMUP_SECONDS
        rate_until = warmup_until + RATE_MEASURE_SECONDS
        rate_frames = 0
        rate_reported = False
        last_corrupt_log = 0.0
        stored = 0

        for packet in container.demux(stream):
            if self._stop.is_set():
                return stored
            try:
                frames = packet.decode()
            except av.error.FFmpegError:
                # The packet was too damaged to decode at all — discard and
                # count it; the next packet decodes normally.
                with self._lock:
                    self._corrupt_skipped += 1
                now = time.time()
                if now - last_corrupt_log >= 30.0:    # log at most once per 30 s
                    last_corrupt_log = now
                    log.warning("Undecodable video data discarded (corrupt packet)")
                continue
            for frame in frames or ():
                now = time.time()
                if frame.is_corrupt:
                    # The camera sent glitched H.264 data; FFmpeg concealed it
                    # into a smeared frame — that must not reach detection.
                    if now >= warmup_until:
                        with self._lock:
                            self._corrupt_skipped += 1
                        if now - last_corrupt_log >= 30.0:    # log at most once per 30 s
                            last_corrupt_log = now
                            log.warning("Corrupt frame discarded (damaged H.264 data)")
                    continue
                if now < warmup_until:
                    # The decoder joined mid-GOP: until the first keyframe
                    # arrives it can only guess, and these frames may be
                    # glitched (FFmpeg's "corrupted macroblock" lines).
                    continue
                if not rate_reported:
                    # Measure the true frame rate once, right after the
                    # warm-up (the connect-time burst of buffered frames
                    # has drained); the camera's metadata claim is often
                    # wrong (some encoders report half the real fps).
                    if now < rate_until:
                        rate_frames += 1
                    else:
                        rate_reported = True
                        measured = rate_frames / RATE_MEASURE_SECONDS
                        if claimed > 0.0:
                            log.info(
                                "Stream rate: ~%.1f fps measured over %.1f s "
                                "(camera metadata claimed %.1f fps)",
                                measured, RATE_MEASURE_SECONDS, claimed,
                            )
                        else:
                            log.info(
                                "Stream rate: ~%.1f fps measured over %.1f s",
                                measured, RATE_MEASURE_SECONDS,
                            )
                image = frame.to_ndarray(format="bgr24")    # convert outside the lock
                with self._lock:
                    self._frame = image
                    self._frame_ts = now
                    self._frames_read += 1
                stored += 1
        return stored

    def latest(self):
        """Return (frame, unix_timestamp) of the newest frame, or (None, 0.0)."""
        with self._lock:
            return self._frame, self._frame_ts

    def frames_read(self) -> int:
        """Total frames received from the stream since the service started."""
        with self._lock:
            return self._frames_read

    def corrupt_skipped(self) -> int:
        """Frames discarded as corrupt (damaged H.264 data) since the service started."""
        with self._lock:
            return self._corrupt_skipped

    def stop(self) -> None:
        self._stop.set()


# --------------------------------------------------------------------------- #
# Detection and saving                                                         #
# --------------------------------------------------------------------------- #

def resolve_class_ids(model: YOLO, names: list[str]) -> list[int]:
    """Translate configured trigger class names into model class ids.

    Unknown names are logged and skipped; if nothing remains, the default
    trigger class ("person") is used.
    """
    ids = []
    for name in names:
        found = [i for i, cname in model.names.items() if cname.lower() == name]
        if found:
            ids.append(found[0])
        else:
            log.warning("Unknown TRIGGER_CLASS '%s' ignored — model classes: %s",
                        name, ", ".join(model.names.values()))
    return list(dict.fromkeys(ids)) or [TRIGGER_CLASS_ID]


def crop_active_window(frame, roi: tuple[float, float, float, float]):
    """Crop the frame to the active window (percent coordinates, top-left origin).

    Returns (cropped, (x, y, w, h)) with the pixel offset and size of the
    window inside the full frame. The crop is a numpy view (no copy).

    Technical rule: the crop width is always divisible by 16, rounded to the
    bigger size — extend right when possible, otherwise shift the window
    left; only if the frame itself is narrower than the rounded width do we
    fall back to the largest multiple of 16 that fits.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi
    px1 = max(0, min(w - 1, int(w * x1 / 100.0 + 0.5)))
    py1 = max(0, min(h - 1, int(h * y1 / 100.0 + 0.5)))
    px2 = max(px1 + 1, min(w, int(w * x2 / 100.0 + 0.5)))
    py2 = max(py1 + 1, min(h, int(h * y2 / 100.0 + 0.5)))

    if (px2 - px1) % 16:
        desired = ((px2 - px1) + 15) // 16 * 16          # round to the bigger size
        if px1 + desired <= w:
            px2 = px1 + desired                           # extend to the right
        else:
            px1 = max(0, w - desired)                    # does not fit: shift left
            px2 = min(w, px1 + desired)
            if (px2 - px1) % 16:                         # frame narrower than the rounded width:
                px2 = px1 + (px2 - px1) // 16 * 16       # largest multiple of 16 that fits
    if px2 <= px1:                                       # degenerate frame (< 16 px wide)
        px1, px2 = 0, w

    return frame[py1:py2, px1:px2], (px1, py1, px2 - px1, py2 - py1)


def detect_objects(model: YOLO, frame, conf_threshold: float, trigger_ids: list[int],
                   device: str = "cpu"):
    """Run YOLO on one frame, restricted to the configured trigger classes.

    Returns (result, count, max_conf) where count is the number of trigger
    objects (any of the listed classes) detected at or above conf_threshold.
    """
    result = model(frame, classes=trigger_ids, conf=conf_threshold, device=device, verbose=False)[0]
    boxes = result.boxes
    count = 0 if boxes is None else len(boxes)
    max_conf = max(boxes.conf.tolist()) if count else 0.0
    return result, count, max_conf


def check_hotkey() -> bool:
    """Undocumented: True if 's' was pressed in the console (Windows only).

    Drains pending console keys; any 's'/'S' found triggers a manual
    screenshot. Silently unavailable on platforms without msvcrt.
    """
    if msvcrt is None:
        return False
    try:
        while msvcrt.kbhit():
            if msvcrt.getwch().lower() == "s":
                return True
    except Exception:
        pass
    return False


def save_screenshot(image, save_dir: Path, detections_log: Path, stream_name: str,
                    max_conf: float, count: int) -> bool:
    """Write one screenshot plus a JSONL record; returns True on success.

    Screenshots go into a per-day subfolder of save_dir and are named
    "<stream name> YYYY-MM-DD HH-MM-SS.jpg". Saving twice in the same second
    silently overwrites the earlier file (deterministic name, no duplicates).
    """
    now = datetime.now()
    day_dir = save_dir / now.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    name = f"{stream_name} {now.strftime('%Y-%m-%d %H-%M-%S.%f')[:-3]}.jpg"
    path = day_dir / name
    if not cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
        log.error("Failed to write screenshot %s", path)
        return False

    record = {
        "timestamp": now.isoformat(timespec="milliseconds"),
        "file": f"{day_dir.name}/{name}",
        "objects": count,
        "max_confidence": round(max_conf, 4),
    }
    with detections_log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info("Screenshot saved: %s (%d object(s), confidence %.1f%%)", path, count, max_conf * 100)
    return True


# --------------------------------------------------------------------------- #
# Service loop                                                                 #
# --------------------------------------------------------------------------- #

def run_service(cfg: dict, limit_seconds: float | None = None) -> int:
    if not cfg["rtsp_url"]:
        log.error("No RTSP URL configured — set RTSP_STREAM in .env")
        return 1

    if av is None:
        log.error("The PyAV package is required to decode the RTSP stream —"
                  " install it with:  python -m pip install av")
        return 1

    save_dir = resolve_path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    detections_log = save_dir / "detections.jsonl"

    log.info("Loading YOLO model '%s' (downloaded automatically on first run) ...", cfg["model"])
    model = YOLO(cfg["model"])
    device = select_device(cfg["device"])
    trigger_ids = resolve_class_ids(model, cfg["trigger_classes"])
    trigger_list = ",".join(model.names[i] for i in trigger_ids)
    log.info(
        "Watching %s | model=%s | device=%s | name=%s | trigger %s >= %.0f%% | "
        "window %.0f,%.0f->%.0f,%.0f%% | save interval %.1f s | screenshots -> %s",
        mask_credentials(cfg["rtsp_url"]), cfg["model"], device, cfg["stream_name"],
        trigger_list, cfg["conf_threshold"] * 100.0, *cfg["active_window"],
        cfg["save_interval"], save_dir,
    )

    reader = RTSPReader(cfg["rtsp_url"], cfg["transport"])
    reader.start()

    trigger_visible = False
    last_save = 0.0
    last_wait_log = 0.0
    frames_analysed = 0
    screenshots = 0
    frame_period = 1.0 / max(cfg["detect_fps"], 0.1)
    started = time.time()

    # Counters for the periodic one-line status (STAT_INTERVAL_SECONDS).
    last_stat = time.time()
    frames_read_prev = 0
    corrupt_prev = 0
    stat_analysed = 0
    stat_stale = 0
    stat_trigger = 0
    stat_shots = 0
    stat_infer_ms = 0.0
    window_logged = False

    try:
        while True:
            if limit_seconds is not None and time.time() - started >= limit_seconds:
                log.info("Time limit of %.0f s reached, stopping.", limit_seconds)
                break

            # One-line status every STAT_INTERVAL_SECONDS — proves the service is alive.
            now = time.time()
            if now - last_stat >= cfg["stat_interval"]:
                window = now - last_stat
                received = reader.frames_read() - frames_read_prev
                corrupt = reader.corrupt_skipped() - corrupt_prev
                # The loop samples only the newest frame at DETECT_FPS; frames
                # are also skipped while the save cooldown runs. Both count as
                # dropped — by design. "stale" counts checks skipped because the
                # newest frame was older than STALE_FRAME_SECONDS (a hiccup).
                dropped = max(0, received - stat_analysed)
                fps = stat_analysed / window
                avg_ms = stat_infer_ms / stat_analysed if stat_analysed else 0.0
                log.info(
                    "STAT %ds: stream %d fr | analysed %d (%.1f fps, %.0f ms/frame) | "
                    "dropped %d | corrupt %d fr | trigger %d fr",
                    int(window), received, stat_analysed, fps, avg_ms,
                    dropped, corrupt, stat_trigger,
                )
                last_stat = now
                frames_read_prev = reader.frames_read()
                corrupt_prev = reader.corrupt_skipped()
                stat_analysed = stat_stale = stat_trigger = stat_shots = 0
                stat_infer_ms = 0.0

            frame, frame_ts = reader.latest()
            if frame is None:
                if time.time() - last_wait_log >= 5.0:
                    log.info("Waiting for frames from the stream ...")
                    last_wait_log = time.time()
                time.sleep(0.25)
                continue
            if now - frame_ts > STALE_FRAME_SECONDS:
                # Stream hiccup: never analyse (or screenshot) an old frame.
                stat_stale += 1
                time.sleep(0.25)
                continue
            # Screenshot cooldown: nothing can be saved until SAVE_INTERVAL_SECONDS
            # has passed since the last screenshot, so skip the expensive
            # inference entirely — the skipped frames count as dropped. The
            # undocumented hotkey ('s' in the console) forces an analysis so a
            # manual screenshot stays immediate.
            manual_shot = check_hotkey()
            if not manual_shot and now - last_save < cfg["save_interval"]:
                time.sleep(frame_period)
                continue

            loop_start = time.perf_counter()
            crop, win = crop_active_window(frame, cfg["active_window"])
            if not window_logged:
                log.info("Active window: %dx%d px at (%d, %d) of %dx%d frame",
                         win[2], win[3], win[0], win[1], frame.shape[1], frame.shape[0])
                window_logged = True
            infer_start = time.perf_counter()
            result, count, max_conf = detect_objects(model, crop, cfg["conf_threshold"], trigger_ids, device)
            stat_infer_ms += (time.perf_counter() - infer_start) * 1000.0
            frames_analysed += 1
            stat_analysed += 1

            if count and not trigger_visible:
                log.info("Trigger object(s) detected (%d, best confidence %.1f%%)",
                         count, max_conf * 100.0)
            elif not count and trigger_visible:
                log.info("Trigger object(s) no longer visible")
            trigger_visible = count > 0
            if count:
                stat_trigger += 1

            # Normal save: trigger object present + save interval. The
            # undocumented hotkey ('s' in the console, checked above every
            # iteration) saves immediately, regardless of triggers; it never
            # affects the save pacing.
            save_now = bool(count) and time.time() - last_save >= cfg["save_interval"]
            if save_now or manual_shot:
                image = result.plot() if cfg["draw_boxes"] else crop
                if save_screenshot(image, save_dir, detections_log, cfg["stream_name"], max_conf, count):
                    screenshots += 1
                    stat_shots += 1
                if save_now:
                    last_save = time.time()

            # Pace the loop to keep CPU usage sane.
            elapsed = time.perf_counter() - loop_start
            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)
    except KeyboardInterrupt:
        log.info("Stopped by user (Ctrl+C).")
    finally:
        reader.stop()

    log.info("Done: %d frame(s) analysed, %d screenshot(s) saved.", frames_analysed, screenshots)
    return 0


# --------------------------------------------------------------------------- #
# Selftest (no camera required)                                                #
# --------------------------------------------------------------------------- #

def run_selftest(cfg: dict) -> int:
    """Validate the detection pipeline on a sample image containing people."""
    log.info("Loading YOLO model '%s' ...", cfg["model"])
    model = YOLO(cfg["model"])
    device = select_device(cfg["device"])
    trigger_ids = resolve_class_ids(model, cfg["trigger_classes"])
    log.info("Trigger classes: %s", ", ".join(model.names[i] for i in trigger_ids))

    target = PROJECT_ROOT / "selftest_input.jpg"
    log.info("Downloading test image %s ...", SELFTEST_IMAGE)
    try:
        with urllib.request.urlopen(SELFTEST_IMAGE, timeout=20) as resp:
            target.write_bytes(resp.read())
    except Exception as exc:
        log.error("Could not download the test image (offline?): %s", exc)
        return 1

    image = cv2.imread(str(target))
    if image is None:
        log.error("Downloaded test image could not be read.")
        return 1

    full_h, full_w = image.shape[:2]
    image, win = crop_active_window(image, cfg["active_window"])
    log.info("Active window: %dx%d px at (%d, %d) of %dx%d image",
             win[2], win[3], win[0], win[1], full_w, full_h)

    # Raw detections first — proves the model and the trigger classes work.
    raw = model(image, classes=trigger_ids, conf=0.25, device=device, verbose=False)[0]
    raw_confs = [] if not len(raw.boxes) else [round(c, 3) for c in raw.boxes.conf.tolist()]
    log.info("Selftest: %d object(s) detected at confidence >= 0.25 -> %s",
             len(raw.boxes), raw_confs)

    # Then check the configured threshold on the same image.
    result, count, max_conf = detect_objects(model, image, cfg["conf_threshold"], trigger_ids, device)
    log.info("Of those, %d object(s) pass your CONFIDENCE_THRESHOLD=%.2f.",
             count, cfg["conf_threshold"])
    log.info("(The sample image is a street scene — what is found depends on TRIGGER_CLASS and ACTIVE_WINDOW.)")

    out = PROJECT_ROOT / "selftest_output.jpg"
    cv2.imwrite(str(out), raw.plot())
    log.info("Annotated result saved to %s", out)
    return 0 if len(raw.boxes) else 2


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Watch an RTSP stream and save screenshots when a configured "
                    "object (person, dog, ...) is detected with enough confidence (YOLO)."
    )
    parser.add_argument("--selftest", action="store_true",
                        help="verify the detection pipeline on a sample image; no camera needed")
    parser.add_argument("--url", help="override the RTSP URL from .env")
    parser.add_argument("--limit", type=float, metavar="SECONDS",
                        help="stop automatically after this many seconds")
    args = parser.parse_args()

    cfg = load_config()
    if args.url:
        cfg["rtsp_url"] = args.url
    setup_logging(cfg["log_file"])

    if args.selftest:
        return run_selftest(cfg)
    return run_service(cfg, limit_seconds=args.limit)


if __name__ == "__main__":
    sys.exit(main())