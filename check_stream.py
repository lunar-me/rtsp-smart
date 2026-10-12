"""
Quick RTSP connectivity diagnostic.

Connects to the configured stream, drains the same 1 s connect warm-up the
service uses, then reads a burst of frames and prints the measured (not the
metadata — often wrong) frame rate, and saves the first frame — cropped to
the ACTIVE_WINDOW region, the same area the service analyses and saves —
as stream_check.jpg.

Usage:
    python check_stream.py
    python check_stream.py --url rtsp://user:pass@host:554/stream
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2

from main import (CONNECT_WARMUP_SECONDS, crop_active_window, load_config,
                  mask_credentials)

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="RTSP connectivity diagnostic.")
    parser.add_argument("--url", help="override the RTSP URL from .env")
    parser.add_argument("--frames", type=int, default=30,
                        help="how many frames to read (default 30)")
    args = parser.parse_args()

    cfg = load_config()
    url = args.url or cfg["rtsp_url"]
    if not url:
        print("No RTSP URL configured — set RTSP_STREAM in .env or pass --url.")
        return 1

    # The service itself decodes with PyAV; this diagnostic intentionally uses
    # the plain OpenCV path, so a camera problem can be told apart from a
    # service problem. These options mirror the service's RTSP settings.
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"rtsp_transport;{cfg['transport']}|timeout;5000000")
    print(f"Connecting to {mask_credentials(url)} (transport={cfg['transport']}) ...")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("FAILED: could not open the stream. Check the URL, credentials,")
        print("that the PC is on the camera's network, and that RTSP is enabled.")
        return 1
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # no-op on OpenCV's FFMPEG backend, harmless

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Connected: {width}x{height} (metadata claims {fps:.1f} fps — measured below)")

    # Warm-up, same as the service: the frames right after connect join the
    # H.264 stream mid-GOP and may arrive as a buffered burst — drain them so
    # the frame-rate measurement below reflects the real stream pace.
    warmup_until = time.time() + CONNECT_WARMUP_SECONDS
    while time.time() < warmup_until:
        ok, _ = cap.read()
        if not ok:
            print("FAILED: the stream dropped during the warm-up read.")
            cap.release()
            return 1

    read = 0
    t_first = t_last = 0.0
    for i in range(args.frames):
        ok, frame = cap.read()
        if not ok:
            break
        read += 1
        t_last = time.time()
        if read == 1:
            t_first = t_last
        if i == 0:
            fh, fw = frame.shape[:2]
            crop, win = crop_active_window(frame, cfg["active_window"])
            print(f"Active window: {win[2]}x{win[3]} px at ({win[0]}, {win[1]}) "
                  f"of {fw}x{fh} frame")
            out = PROJECT_ROOT / "stream_check.jpg"
            cv2.imwrite(str(out), crop)
            print(f"Saved active window to {out}")

    cap.release()
    if read >= 2 and t_last > t_first:
        measured = (read - 1) / (t_last - t_first)
        print(f"OK: read {read}/{args.frames} frame(s) — measured ~{measured:.1f} fps.")
    else:
        print(f"OK: read {read}/{args.frames} frame(s).")
    return 0 if read > 0 else 1


if __name__ == "__main__":
    sys.exit(main())