"""
Quick RTSP connectivity diagnostic.

Connects to the configured stream, reads a few frames, prints stream
parameters and saves the first frame — cropped to the ACTIVE_WINDOW region,
the same area the service analyses and saves — as stream_check.jpg.

Usage:
    python check_stream.py
    python check_stream.py --url rtsp://user:pass@host:554/stream
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from main import crop_active_window, load_config, mask_credentials, set_ffmpeg_options

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="RTSP connectivity diagnostic.")
    parser.add_argument("--url", help="override the RTSP URL from .env")
    parser.add_argument("--frames", type=int, default=5,
                        help="how many frames to read (default 5)")
    args = parser.parse_args()

    cfg = load_config()
    url = args.url or cfg["rtsp_url"]
    if not url:
        print("No RTSP URL configured — set RTSP_STREAM in .env or pass --url.")
        return 1

    set_ffmpeg_options(cfg["transport"])
    print(f"Connecting to {mask_credentials(url)} (transport={cfg['transport']}) ...")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("FAILED: could not open the stream. Check the URL, credentials,")
        print("that the PC is on the camera's network, and that RTSP is enabled.")
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Connected: {width}x{height} @ {fps:.1f} fps")

    read = 0
    for i in range(args.frames):
        ok, frame = cap.read()
        if not ok:
            break
        read += 1
        if i == 0:
            fh, fw = frame.shape[:2]
            crop, win = crop_active_window(frame, cfg["active_window"])
            print(f"Active window: {win[2]}x{win[3]} px at ({win[0]}, {win[1]}) "
                  f"of {fw}x{fh} frame")
            out = PROJECT_ROOT / "stream_check.jpg"
            cv2.imwrite(str(out), crop)
            print(f"Saved active window to {out}")

    cap.release()
    print(f"OK: read {read}/{args.frames} frame(s).")
    return 0 if read > 0 else 1


if __name__ == "__main__":
    sys.exit(main())