"""
Recolor blue objects to yellow in all videos under a dataset directory.

Uses ffmpeg pipes for decoding/encoding (handles AV1 and any other codec).

Strategy per frame:
  1. BGR → HSV
  2. Mask pixels where hue ∈ [blue_lo, blue_hi] (OpenCV 0-180 scale)
  3. Set masked hue to yellow_hue, keep S and V unchanged
     → shading / highlights on the box are preserved naturally

OpenCV HSV scale (0-180):
  Blue   ≈  95–130
  Yellow ≈  20– 35

Usage:
  # Preview before/after (saves side-by-side frames as preview_<n>.png)
  python scripts/recolor_blue_to_yellow.py \
      --input  dataset/G1_Dex3_ObjectPlacement_Dataset/videos \
      --preview dataset/.../episode_000000.mp4

  # Convert to a NEW directory (safe, keeps originals)
  python scripts/recolor_blue_to_yellow.py \
      --input  dataset/G1_Dex3_ObjectPlacement_Dataset/videos \
      --output dataset/G1_Dex3_ObjectPlacement_Dataset_yellow/videos \
      --workers 8

  # Convert in-place (overwrites originals)
  python scripts/recolor_blue_to_yellow.py \
      --input  dataset/G1_Dex3_ObjectPlacement_Dataset/videos \
      --workers 8
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------

def probe_video(path: Path) -> dict:
    """Return {width, height, fps, codec} for the first video stream."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_streams", "-of", "json",
        str(path),
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {
        "width": int(s["width"]),
        "height": int(s["height"]),
        "fps": float(num) / float(den),
        "codec": s["codec_name"],
    }


def open_decoder(path: Path, width: int, height: int) -> subprocess.Popen:
    """Open ffmpeg process that streams raw BGR24 frames to stdout."""
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-i", str(path),
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def open_encoder(path: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    """Open ffmpeg process that reads raw BGR24 frames from stdin and encodes to mp4/h264."""
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264",
        "-crf", "18",          # near-lossless; raise to 23 for smaller files
        "-preset", "fast",
        "-pix_fmt", "yuv420p", # broad player compatibility
        str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


# ---------------------------------------------------------------------------
# Core per-frame recoloring
# ---------------------------------------------------------------------------

def recolor_frame(frame_bgr: np.ndarray, blue_lo: int, blue_hi: int, yellow_hue: int) -> np.ndarray:
    """Recolor blue pixels to yellow, preserving saturation and value."""
    import cv2  # imported inside worker process
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    mask = (h >= blue_lo) & (h <= blue_hi) & (s >= 60) & (v >= 40)
    h[mask] = np.uint8(yellow_hue)

    hsv[:, :, 0] = h
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------------------
# Per-video worker (runs in a subprocess pool)
# ---------------------------------------------------------------------------

def process_video(
    src: Path,
    dst: Path,
    blue_lo: int,
    blue_hi: int,
    yellow_hue: int,
) -> tuple[str, bool, str]:
    try:
        info = probe_video(src)
        w, h, fps = info["width"], info["height"], info["fps"]
        frame_bytes = w * h * 3

        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4", dir=dst.parent)
        os.close(tmp_fd)

        decoder = open_decoder(src, w, h)
        encoder = open_encoder(Path(tmp_path), w, h, fps)

        frame_count = 0
        while True:
            raw = decoder.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)).copy()
            out = recolor_frame(frame, blue_lo, blue_hi, yellow_hue)
            encoder.stdin.write(out.tobytes())
            frame_count += 1

        decoder.stdout.close()
        decoder.wait()
        encoder.stdin.close()
        encoder.wait()

        if frame_count == 0:
            os.unlink(tmp_path)
            return str(src), False, "no frames decoded"

        shutil.move(tmp_path, dst)
        return str(src), True, f"{frame_count} frames"

    except Exception as exc:  # noqa: BLE001
        return str(src), False, str(exc)


# ---------------------------------------------------------------------------
# Preview (saves PNG side-by-side images, no display needed)
# ---------------------------------------------------------------------------

def preview_video(src: Path, blue_lo: int, blue_hi: int, yellow_hue: int, n_frames: int = 5):
    import cv2
    info = probe_video(src)
    w, h, fps = info["width"], info["height"], info["fps"]
    frame_bytes = w * h * 3

    decoder = open_decoder(src, w, h)
    out_dir = Path("preview_frames")
    out_dir.mkdir(exist_ok=True)

    for i in range(n_frames):
        raw = decoder.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)).copy()
        recolored = recolor_frame(frame, blue_lo, blue_hi, yellow_hue)
        side_by_side = np.hstack([frame, recolored])
        out_path = out_dir / f"preview_{i:02d}.png"
        cv2.imwrite(str(out_path), side_by_side)
        print(f"  saved {out_path}")

    decoder.stdout.close()
    decoder.wait()
    print(f"\nLeft = original, Right = recolored. Check {out_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recolor blue boxes to yellow in dataset videos.")
    p.add_argument("--input", required=True, type=Path, help="Root videos directory")
    p.add_argument("--output", type=Path, default=None,
                   help="Output directory (default: overwrite input in-place)")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2),
                   help="Parallel worker processes (default: half CPU count)")
    p.add_argument("--blue-lo", type=int, default=95,
                   help="Lower hue bound for blue (default 95)")
    p.add_argument("--blue-hi", type=int, default=130,
                   help="Upper hue bound for blue (default 130)")
    p.add_argument("--yellow-hue", type=int, default=28,
                   help="Target hue for yellow (default 28, OpenCV 0-180 scale)")
    p.add_argument("--preview", type=Path, default=None,
                   help="Save before/after PNG frames for this video then exit")
    p.add_argument("--preview-frames", type=int, default=5,
                   help="Number of frames to preview (default 5)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.preview:
        preview_video(args.preview, args.blue_lo, args.blue_hi, args.yellow_hue, args.preview_frames)
        return

    videos = sorted(args.input.rglob("*.mp4"))
    if not videos:
        print(f"No .mp4 files found under {args.input}", file=sys.stderr)
        sys.exit(1)

    in_place = args.output is None
    out_root = args.input if in_place else args.output

    print(f"Found {len(videos)} videos")
    print(f"Blue hue range : [{args.blue_lo}, {args.blue_hi}]")
    print(f"Yellow hue     : {args.yellow_hue}")
    print(f"Workers        : {args.workers}")
    print(f"Output root    : {out_root}  ({'in-place' if in_place else 'separate dir'})")
    print()

    tasks = [
        (src, out_root / src.relative_to(args.input), args.blue_lo, args.blue_hi, args.yellow_hue)
        for src in videos
    ]

    done, errors = 0, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_video, *t): t[0] for t in tasks}
        for fut in as_completed(futures):
            path, ok, msg = fut.result()
            done += 1
            status = "OK" if ok else "FAIL"
            print(f"[{done:4d}/{len(tasks)}] {status}  {Path(path).name}  {msg}")
            if not ok:
                errors.append((path, msg))

    print(f"\nDone. {len(tasks) - len(errors)}/{len(tasks)} succeeded.")
    if errors:
        print("Failures:")
        for path, msg in errors:
            print(f"  {path}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
