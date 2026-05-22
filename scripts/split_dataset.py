"""
Split a LeRobot v2 dataset into train/test subsets.

Each split gets its own directory with:
  - meta/        real copy, filtered to that split's episodes
  - data/        real directory with hard-linked parquet files (no disk copy)
  - videos/      real directory with hard-linked mp4 files (no disk copy)

Hard links share the same disk blocks as the originals — no extra storage used,
but each split only exposes the files that belong to it.

Usage:
    python scripts/split_dataset.py \
        --dataset dataset/G1_Dex3_ObjectPlacement_Dataset \
        --test-count 10          # hold out last 10 episodes as test

    # Or by ratio:
    python scripts/split_dataset.py \
        --dataset dataset/G1_Dex3_ObjectPlacement_Dataset \
        --test-ratio 0.1

Output:
    dataset/G1_Dex3_ObjectPlacement_Dataset_train/
    dataset/G1_Dex3_ObjectPlacement_Dataset_test/
"""

import argparse
import json
import os
import shutil
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def hardlink_tree(src_root: Path, dst_root: Path, episode_indices: set[int], chunk_size: int) -> int:
    """
    Recreate src_root under dst_root, hard-linking only files whose name
    matches episode_XXXXXX.{parquet,mp4} with an index in episode_indices.
    Non-episode files (e.g. chunk dirs themselves) are just mkdir'd.
    Returns the number of files linked.
    """
    n = 0
    for src_path in src_root.rglob("*"):
        rel = src_path.relative_to(src_root)
        dst_path = dst_root / rel

        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            continue

        # Parse episode index from filename: episode_000205.mp4 / .parquet
        stem = src_path.stem  # e.g. "episode_000205"
        if stem.startswith("episode_"):
            try:
                ep_idx = int(stem.split("_")[1])
            except (IndexError, ValueError):
                ep_idx = None

            if ep_idx is not None and ep_idx not in episode_indices:
                continue  # skip episodes not in this split

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        os.link(src_path, dst_path)
        n += 1
    return n


def create_split_dataset(
    src: Path,
    dst: Path,
    episodes: list[dict],
    split_label: str,
    chunk_size: int,  # passed through to info.json patch
) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    ep_indices = {e["episode_index"] for e in episodes}

    # --- meta/ ---------------------------------------------------------------
    src_meta = src / "meta"
    dst_meta = dst / "meta"
    dst_meta.mkdir()

    for f in src_meta.iterdir():
        shutil.copy2(f, dst_meta / f.name)

    write_jsonl(dst_meta / "episodes.jsonl", episodes)

    with open(dst_meta / "info.json") as f:
        info = json.load(f)

    sorted_indices = sorted(ep_indices)
    n_cams = sum(1 for k, v in info.get("features", {}).items() if v.get("dtype") == "video")
    info["splits"] = {split_label: f"{sorted_indices[0]}:{sorted_indices[-1] + 1}"}
    info["total_episodes"] = len(episodes)
    info["total_frames"] = sum(e["length"] for e in episodes)
    info["total_videos"] = len(episodes) * n_cams
    info["total_chunks"] = max(1, (sorted_indices[-1] // chunk_size) + 1 - (sorted_indices[0] // chunk_size))

    with open(dst_meta / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    # --- data/ and videos/ — hard-link only relevant episode files -----------
    total_links = 0
    for folder in ("data", "videos"):
        src_folder = src / folder
        if src_folder.exists():
            n = hardlink_tree(src_folder, dst / folder, ep_indices, chunk_size)
            total_links += n

    print(
        f"  {split_label:6s}: {len(episodes):4d} episodes, "
        f"{info['total_frames']:7d} frames, "
        f"{total_links} files hard-linked → {dst}"
    )


def main():
    p = argparse.ArgumentParser(description="Split LeRobot dataset into train/test.")
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--test-count", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    src = args.dataset.resolve()
    if not src.is_dir():
        raise SystemExit(f"Dataset not found: {src}")

    episodes = read_jsonl(src / "meta" / "episodes.jsonl")
    total = len(episodes)

    with open(src / "meta" / "info.json") as f:
        chunk_size = json.load(f).get("chunks_size", 1000)

    n_test = args.test_count if args.test_count is not None else max(1, round(total * args.test_ratio))
    n_train = total - n_test

    if n_train <= 0:
        raise SystemExit(f"Not enough episodes: total={total}, n_test={n_test}")

    train_eps = episodes[:n_train]
    test_eps  = episodes[n_train:]

    out_parent = args.output_dir or src.parent
    dst_train = out_parent / (src.name + "_train")
    dst_test  = out_parent / (src.name + "_test")

    print(f"Source : {src}  ({total} episodes)")
    print(f"Split  : {n_train} train / {n_test} test")
    print()

    create_split_dataset(src, dst_train, train_eps, "train", chunk_size)
    create_split_dataset(src, dst_test,  test_eps,  "test",  chunk_size)

    print()
    print("Done.")
    print(f"  Train : --dataset-path {dst_train}")
    print(f"  Test  : --dataset-path {dst_test}  --traj-ids {' '.join(str(i) for i in range(n_test))}")


if __name__ == "__main__":
    main()
