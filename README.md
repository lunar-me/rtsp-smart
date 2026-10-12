# rtsp-smart

Real-time object watcher (person, dog, …) for an RTSP (H.264) camera.

The service reads the stream, crops each frame to the `ACTIVE_WINDOW` region
(default: the full frame), runs YOLO26 (the latest Ultralytics model family)
on it and — while any object from `TRIGGER_CLASS` (default: person, dog) is
visible with **confidence above `CONFIDENCE_THRESHOLD`** — saves one
screenshot per second to a local folder. If the stream drops, it reconnects
automatically.

## Quick start

```bash
python -m pip install -r requirements.txt   # NVIDIA GPU: use requirements-gpu.txt instead

cp .env.example .env          # Windows: copy .env.example .env — then set your RTSP URL inside

python main.py --selftest     # optional: verify detection without the camera
python check_stream.py        # optional: verify the stream (saves the active window as stream_check.jpg)

python main.py                # start the service (Ctrl+C to stop)
```

> First run downloads the YOLO weights automatically (~5 MB, needs internet).
> The pip install is large because it includes PyTorch (CPU build by default;
> for NVIDIA GPU support use `requirements-gpu.txt` instead — see Troubleshooting).

## How it works

- A background thread keeps **only the most recent frame** from the camera, so
  internal buffering can never build up latency.
- Each analysed frame is first cropped to the `ACTIVE_WINDOW` region
  (x1,y1,x2,y2 in % of the frame, from the top-left corner;
  default `0,0,100,100` = full frame) — screenshots and `stream_check.jpg`
  show that region only. The crop width is always divisible by 16
  (rounded to the bigger size).
- The crop goes through YOLO, restricted to the `TRIGGER_CLASS` classes —
  whichever class first passes `CONFIDENCE_THRESHOLD` triggers the screenshot.
- While a trigger object is visible: one screenshot per `SAVE_INTERVAL_SECONDS`, saved as
  `<SAVE_DIR>/<YYYY-MM-DD>/<RTSP_NAME> YYYY-MM-DD HH-MM-SS.jpg` (same-second saves overwrite).
- After each screenshot, no frame is analysed for the rest of
  `SAVE_INTERVAL_SECONDS` — no inference runs while no screenshot could be
  saved anyway.
- If frames stop arriving (stream hiccup), stale frames are skipped —
  no screenshots are made from an old image.
- If the stream drops: automatic reconnect with exponential backoff (1 s -> 15 s); the first second of frames after every (re)connect is discarded — the decoder cannot produce intact frames before the first keyframe.
- A one-line status is logged every `STAT_INTERVAL_SECONDS` (default 60 s):
  frames received / analysed / dropped (superseded between analyses, or
  skipped while the screenshot cooldown runs — by design, the loop samples
  only the newest frame at `DETECT_FPS`), analysis speed, stale checks, trigger frames, screenshots, device.
- The NVIDIA GPU is used automatically when available (`DEVICE=auto`); the
  status line shows which device is actually in use.

## Output

| Path | Content |
|---|---|
| `screenshots/<YYYY-MM-DD>/<RTSP_NAME> YYYY-MM-DD HH-MM-SS.jpg` | saved screenshots (per-day folder) |
| `screenshots/detections.jsonl` | one JSON record per screenshot (timestamp, file, people count, confidence) |
| `logs/service.log` | rolling service log |

## Configuration (`.env`)

Copy [`.env.example`](.env.example) to `.env` as a starting point — it lists every key with its default.

| Key | Default | Meaning |
|---|---|---|
| `RTSP_STREAM` | – | RTSP URL (`RTSP_STRAM` / `RTSP_URL` also accepted) |
| `RTSP_TRANSPORT` | `tcp` | `tcp` = reliable; `udp` = lower latency but can drop frames |
| `RTSP_NAME` | `RTSP` | camera name, used in screenshot file names |
| `MODEL` | `yolo26n.pt` | `yolo26s.pt` / `yolo26m.pt` = more accurate, slower |
| `DEVICE` | `auto` | `auto` = NVIDIA GPU if available; `cpu`; `cuda:0` |
| `TRIGGER_CLASS` | `person` | comma-separated COCO classes to detect, e.g. `person,dog` |
| `CONFIDENCE_THRESHOLD` | `0.90` | minimum confidence of any trigger object |
| `ACTIVE_WINDOW` | `0,0,100,100` | analysed region x1,y1,x2,y2 in % of the frame (from the top-left); crop width rounded to a multiple of 16 |
| `SAVE_INTERVAL_SECONDS` | `1.0` | seconds between screenshots; analysis pauses for this long after each save |
| `DETECT_FPS` | `5` | frames analysed per second (lower = less CPU) |
| `STAT_INTERVAL_SECONDS` | `60` | seconds between one-line status log entries |
| `SAVE_DIR` | `screenshots` | screenshot folder — relative to the project or absolute (e.g. another disk) |
| `DRAW_BOXES` | `false` | draw YOLO boxes on saved screenshots |
| `LOG_FILE` | `logs/service.log` | log file |

## Run as a background service (optional)

Windows (NSSM):

```
nssm install RtspSmart "C:\Python\python.exe" "d:\lunar-me\rtsp-smart\main.py"
nssm set RtspSmart AppDirectory d:\lunar-me\rtsp-smart
nssm start RtspSmart
```

Linux (systemd):

```
[Unit]
Description=rtsp-smart object watcher
After=network.target

[Service]
WorkingDirectory=/opt/rtsp-smart
ExecStart=/usr/bin/python3 /opt/rtsp-smart/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Troubleshooting

- **"Cannot open stream"** — check the URL and credentials, confirm the PC is on
  the camera's network, try `RTSP_TRANSPORT=udp`, and run
  `python check_stream.py` for a detailed diagnosis.
- **Objects not detected** — check the `TRIGGER_CLASS` names against the model's
  COCO classes and that `ACTIVE_WINDOW` covers the area you expect, try a bigger
  model (`MODEL=yolo26s.pt`), or temporarily lower `CONFIDENCE_THRESHOLD` to see
  raw detections in the log.
- **`[h264 @ ...] corrupted macroblock` / `error while decoding MB` in the console** —
  the camera's H.264 stream glitched (typical with phone webcam apps; right after
  a (re)connect it is even expected, before the first keyframe). FFmpeg conceals it
  and keeps decoding, and those first frames are never analysed. If the lines
  spam non-stop while running, lower the camera app's resolution/bitrate or
  improve its Wi-Fi link.
- **Too much CPU** — lower `DETECT_FPS`.
- **Too many screenshots** — raise `SAVE_INTERVAL_SECONDS`.
- **GPU not used** — if the log says *"this PyTorch build is CPU-only"*, install
  the CUDA build of PyTorch and restart the service:

  ```
  python -m pip install -r requirements-gpu.txt
  ```

  (works on a fresh venv and also switches an existing CPU install; see
  <https://pytorch.org/get-started/locally/> for other CUDA versions);
  the status line then shows `device cuda:0`.