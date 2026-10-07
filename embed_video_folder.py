"""
Embed every video in a folder with RelSim and write the results to JSON.
Tuned for a single 24 GB GPU (e.g. SageMaker ml.g5.* / NVIDIA A10G).

Output (list, one entry per video):
    [
      {"filename": "clip_001.mov", "embedding": [0.0123, -0.0456, ...]},   # 384 floats, L2-normalized
      {"filename": "sub/clip_002.mov", "error": "..."},                     # only if that video failed
      ...
    ]

Usage:
    python embed_video_folder.py --input_dir /path/to/videos --output embeddings.json
    python embed_video_folder.py --input_dir videos --mode native --batch_size 2
    python embed_video_folder.py --input_dir videos --recursive --resume

How it is parallelized on one GPU:
  * CPU/GPU overlap (frames mode): a thread pool decodes upcoming videos while
    the GPU embeds the current batch, so the GPU rarely waits on decoding.
  * GPU batching: frames from several videos go through one forward pass
    (--frames_per_forward); native mode batches whole videos (--batch_size).
  * OOM back-off: on CUDA out-of-memory the batch is halved and retried, so
    one large video can't kill the run.
  * Results are checkpointed after every batch; --resume continues a run.

Needs a video decoder:  pip install decord   (or: pip install av)
"""
import argparse
import json
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

DEFAULT_CKPT = "thaoshibe/relsim-qwenvl25-lora"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Embed all videos in a folder with RelSim -> JSON")
    p.add_argument("--input_dir", required=True, help="Folder containing the videos")
    p.add_argument("--output", default="video_embeddings.json", help="Output JSON path")
    p.add_argument("--ext", nargs="+", default=[".mov"],
                   help="File extensions to include, case-insensitive (default: .mov)")
    p.add_argument("--recursive", action="store_true", help="Also search subfolders")
    p.add_argument("--checkpoint_dir", default=DEFAULT_CKPT, help="RelSim checkpoint (HF id or local path)")
    p.add_argument("--mode", choices=["frames", "native"], default="frames",
                   help="frames: mean of per-frame embeddings (default). native: whole clip as a Qwen video")
    p.add_argument("--nframes", type=int, default=8, help="Frames sampled per video (default: 8)")
    # frames mode
    p.add_argument("--max_image_size", type=int, default=448,
                   help="frames mode: cap each frame's longest edge (default 448, ~256 tokens/frame); 0 = no cap")
    p.add_argument("--frames_per_forward", type=int, default=32,
                   help="frames mode: frames per GPU forward pass (default 32)")
    p.add_argument("--decode_workers", type=int, default=4,
                   help="frames mode: CPU threads decoding videos ahead of the GPU (default 4)")
    # native mode
    p.add_argument("--fps", type=float, default=None, help="native mode: sample at this fps instead of --nframes")
    p.add_argument("--max_pixels", type=int, default=360 * 420,
                   help="native mode: per-frame pixel budget (default 151200 = 360*420); must be >= 100352")
    # shared
    p.add_argument("--batch_size", type=int, default=8,
                   help="Videos per step (default 8). Native mode: videos per forward pass; 2-4 suits an A10G")
    p.add_argument("--resume", action="store_true",
                   help="Skip videos already embedded in --output and append the rest")
    p.add_argument("--indent", type=int, default=None, help="Pretty-print JSON with this indent")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------- utils
def find_videos(input_dir, exts, recursive):
    exts = {e.lower() if e.startswith(".") else "." + e.lower() for e in exts}
    if recursive:
        found = [os.path.join(r, f) for r, _, fs in os.walk(input_dir) for f in fs]
    else:
        found = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
    return sorted(f for f in found
                  if os.path.isfile(f) and os.path.splitext(f)[1].lower() in exts
                  and not os.path.basename(f).startswith("._"))   # skip macOS resource-fork files


def save_json(results, path, indent=None):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=indent)
    os.replace(tmp, path)   # atomic: a crash never leaves a half-written file


def _is_oom(e):
    import torch
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def _free_gpu():
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _emb_entry(name, e):
    return {"filename": name, "embedding": [round(x, 6) for x in e.tolist()]}


def embed_with_backoff(fn, items, chunk):
    """Run fn on items in chunks of `chunk`; halve the chunk on CUDA OOM.

    Returns (list_of_embeddings_or_None, list_of_errors_or_None), aligned with items.
    """
    embs, errs = [None] * len(items), [None] * len(items)
    i = 0
    while i < len(items):
        part = items[i:i + chunk]
        try:
            out = fn(part)
            for j, e in enumerate(out):
                embs[i + j] = e.cpu()
            i += len(part)
        except Exception as e:  # noqa: BLE001
            _free_gpu()
            if _is_oom(e) and chunk > 1:
                chunk = max(1, chunk // 2)
                print(f"  ! CUDA OOM -> retrying with chunk={chunk}")
                continue
            if len(part) > 1:          # isolate the bad item(s)
                for j, item in enumerate(part):
                    sub_e, sub_err = embed_with_backoff(fn, [item], 1)
                    embs[i + j], errs[i + j] = sub_e[0], sub_err[0]
            else:
                errs[i] = f"{type(e).__name__}: {e}"
            i += len(part)
    return embs, errs


# ----------------------------------------------------------------------------- decoding
def decode_frames(sample_frames, path, nframes):
    """Decode `nframes` frames; on failure retry with fewer (clips shorter than nframes)."""
    n, last_err = nframes, None
    while n >= 2:
        try:
            return sample_frames(path, n), None
        except Exception as e:  # noqa: BLE001
            last_err, n = e, n // 2
    return None, f"{type(last_err).__name__}: {last_err}"


# ----------------------------------------------------------------------------- modes
def run_frames_mode(model, todo, rel, args, on_batch):
    """Decode on CPU threads ahead of the GPU; embed frames of many videos per forward."""
    max_size = args.max_image_size or None
    pending = deque()
    it = iter(todo)
    lookahead = args.batch_size * 3          # bounded prefetch keeps RAM in check

    def embed_videos(frame_lists):
        # each list already holds exactly the frames we want -> no re-sampling inside
        return model.embed_video(frame_lists, mode="frames", nframes=max(map(len, frame_lists)),
                                 max_image_size=max_size,
                                 micro_batch_size=args.frames_per_forward)

    with ThreadPoolExecutor(max_workers=args.decode_workers) as pool:
        def refill():
            while len(pending) < lookahead:
                p = next(it, None)
                if p is None:
                    return
                pending.append((p, pool.submit(decode_frames, model._sample_frames, p, args.nframes)))

        refill()
        while pending:
            batch = [pending.popleft() for _ in range(min(args.batch_size, len(pending)))]
            refill()                          # keep decoding while the GPU works
            entries, ok_names, ok_frames = [], [], []
            for p, fut in batch:
                frames, err = fut.result()
                if frames is None:
                    entries.append((rel(p), None, err))
                else:
                    ok_names.append(rel(p))
                    ok_frames.append(frames)
            if ok_frames:
                # group videos so one forward covers ~frames_per_forward frames
                vids_per_call = max(1, args.frames_per_forward // max(1, args.nframes))
                embs, errs = embed_with_backoff(embed_videos, ok_frames, vids_per_call)
                entries += [(n, e, er) for n, e, er in zip(ok_names, embs, errs)]
            on_batch(entries)


def run_native_mode(model, todo, rel, args, on_batch):
    kw = dict(mode="native", nframes=args.nframes, fps=args.fps, max_pixels=args.max_pixels)

    def embed_paths(paths):
        return model.embed_video(paths, **kw)

    for i in range(0, len(todo), args.batch_size):
        batch = todo[i:i + args.batch_size]
        embs, errs = embed_with_backoff(embed_paths, batch, args.batch_size)
        entries = []
        for p, e, err in zip(batch, embs, errs):
            if e is None and not args.fps:    # short clip? retry with fewer frames
                n = args.nframes // 2
                while e is None and n >= 2:
                    sub, sub_err = embed_with_backoff(
                        lambda ps: model.embed_video(ps, **{**kw, "nframes": n}), [p], 1)
                    e, err, n = sub[0], sub_err[0] or err, n // 2
            entries.append((rel(p), e, err))
        on_batch(entries)


# ----------------------------------------------------------------------------- main
def run(args, model=None):
    if not os.path.isdir(args.input_dir):
        sys.exit(f"--input_dir not found: {args.input_dir}")
    videos = find_videos(args.input_dir, args.ext, args.recursive)
    if not videos:
        sys.exit(f"No {'/'.join(args.ext)} files found in {args.input_dir}")

    results = []
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            results = [r for r in json.load(f) if "embedding" in r]   # retry earlier failures
    done = {r["filename"] for r in results}
    rel = lambda p: os.path.relpath(p, args.input_dir)  # noqa: E731
    todo = [v for v in videos if rel(v) not in done]
    print(f"{len(videos)} videos found, {len(done)} already done, {len(todo)} to embed "
          f"(mode={args.mode}, nframes={args.nframes})")
    if not todo:
        return results

    if model is None:
        from relsim.relsim_score import relsim
        model, _ = relsim(pretrained=True, checkpoint_dir=args.checkpoint_dir)

    t0, n_done = time.time(), 0

    def on_batch(entries):
        nonlocal n_done
        for name, e, err in entries:
            if e is not None:
                results.append(_emb_entry(name, e))
            else:
                print(f"  ! failed: {name} ({err})")
                results.append({"filename": name, "error": err})
        save_json(results, args.output, args.indent)   # checkpoint after every batch
        n_done += len(entries)
        dt = time.time() - t0
        print(f"  [{n_done}/{len(todo)}] {dt:.1f}s  ({n_done / dt:.2f} videos/s)")

    if args.mode == "frames":
        run_frames_mode(model, todo, rel, args, on_batch)
    else:
        run_native_mode(model, todo, rel, args, on_batch)

    ok = sum("embedding" in r for r in results)
    print(f"Done: {ok} embedded, {len(results) - ok} failed -> {args.output}")
    return results


if __name__ == "__main__":
    run(parse_args())
