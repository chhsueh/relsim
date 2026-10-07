# RelSim for Video (experimental)

`relsim` was trained on **still images**, but its backbone, Qwen2.5-VL-7B, natively understands video. `model.embed_video` lets you compute **relational visual similarity between videos**, and between videos and images, with the released checkpoint. No retraining is needed.

> **Status: zero-shot.** The released LoRA and projection head never saw video during training. The code path is tested, and image behavior is unchanged bit for bit, but the quality of video scores has not been benchmarked yet. Treat the results as a strong baseline, not as a validated metric.

---

## Table of contents

1. [Install](#install)
2. [Quick start](#quick-start)
3. [Two modes: `frames` vs `native`](#two-modes)
4. [API reference](#api-reference)
5. [Recipes](#recipes)
6. [Memory and speed](#memory-and-speed)
7. [How it works](#how-it-works)
8. [Limitations](#limitations)
9. [Testing](#testing)
10. [Roadmap: training a video RelSim](#roadmap)

---

## Install <a name="install"></a>

Install the package as in the main [README](README.md), then add a video decoder. `qwen_vl_utils` picks one automatically: `decord` first, then `torchcodec`, then `torchvision` (which needs `av`).

```bash
pip install relsim
pip install decord          # recommended
# or: pip install av        # used through torchvision
```

You don't need a decoder if you pass videos as lists of PIL frames.

---

## Quick start <a name="quick-start"></a>

```python
from relsim.relsim_score import relsim

model, preprocess = relsim(pretrained=True, checkpoint_dir="thaoshibe/relsim-qwenvl25-lora")

emb = model.embed_video(["clip_a.mp4", "clip_b.mp4"])   # [2, 384], L2-normalized
score = (emb[0] @ emb[1]).item()                        # cosine similarity
print(f"relational similarity: {score:.3f}")
```

Video embeddings live in **the same 384-d space** as `model.embed` image embeddings, so a dot product works across video↔video, image↔video and image↔image.

---

## Two modes: `frames` vs `native` <a name="two-modes"></a>

|                      | `mode="frames"` (default)                       | `mode="native"`                                   |
|----------------------|-------------------------------------------------|---------------------------------------------------|
| What it does         | Embeds `nframes` sampled frames as images, then mean-pools them | Feeds the whole clip to Qwen2.5-VL as a video and reads the `<\|query\|>` token |
| Training distribution | ✅ Same as training (images)                    | ⚠️ Out of distribution (the model never saw video tokens) |
| Sees motion / order  | ❌ No (mean-pooling ignores frame order)   | ✅ Yes (temporal patches + M-RoPE time positions)  |
| Cost                 | `nframes` image forward passes                  | One forward pass. Tokens grow with frames × resolution |
| Use it for           | Scene-level relations: *"a {Animal} peeking from behind a {Barrier}"* | Event-level relations: *"a {Agent} tries, fails, then succeeds at {Task}"* |

**Rule of thumb:** start with `frames`. Then compare `native` on a few pairs where you know the right answer. Pick `native` only if it ranks those pairs better.

```python
pairs = [("a.mp4", "b.mp4"), ("a.mp4", "c.mp4")]
for mode in ["frames", "native"]:
    for x, y in pairs:
        e = model.embed_video([x, y], mode=mode, nframes=8)
        print(mode, x, y, round((e[0] @ e[1]).item(), 3))
```

---

## API reference <a name="api-reference"></a>

```python
model.embed_video(
    videos,                 # see "Inputs" below
    mode="frames",          # "frames" | "native"
    nframes=8,              # frames sampled per video
    fps=None,               # native only: sample at this rate instead of nframes (path inputs)
    max_pixels=None,        # native only: per-frame pixel budget, e.g. 360*420
    max_image_size=None,    # frames only: cap each frame's longest edge (as in embed)
    micro_batch_size=None,  # frames: frames per forward; native: videos per forward
) -> torch.Tensor           # [N, 384], L2-normalized, float32
```

**Inputs.** `videos` can be any of the following:

| Input                                   | Interpreted as      |
|-----------------------------------------|---------------------|
| `"clip.mp4"` (path or URL)              | 1 video             |
| `[PIL.Image, PIL.Image, ...]`           | 1 video (its frames) |
| `["a.mp4", "b.mp4", [frames...], ...]`  | N videos (you can mix types) |

**Frame sampling.**
- Path inputs: `nframes` frames are sampled uniformly over the whole clip. In `native` mode you can pass `fps` instead.
- Frame-list inputs: frames are sampled uniformly from the list (all of them if the list has ≤ `nframes`).
- `native` mode rounds the frame count to an even number, because Qwen packs frames in pairs.

**Unchanged.** `embed`, `model(img1, img2)` and `preprocess` work exactly as before.

---

## Recipes <a name="recipes"></a>

### Video ↔ video similarity

```python
e = model.embed_video(["query.mp4", "candidate.mp4"], nframes=8)
print((e[0] @ e[1]).item())
```

### Video retrieval (top-k)

```python
import torch

db_paths = ["v1.mp4", "v2.mp4", "v3.mp4", ...]
db = model.embed_video(db_paths, micro_batch_size=32)           # [M, 384]; cache this
torch.save({"paths": db_paths, "emb": db}, "video_index.pt")

q = model.embed_video("query.mp4")                              # [1, 384]
scores = (q @ db.T)[0]
for i in scores.topk(5).indices.tolist():
    print(f"{scores[i]:.3f}  {db_paths[i]}")
```

### Image → video retrieval (cross-modal)

Images and videos share one embedding space, so you can query a video index with an image, or the other way round:

```python
from PIL import Image

img = model.embed(preprocess(Image.open("cat_peeking.jpg")))    # [1, 384]
scores = (img @ db.T)[0]
```

### In-memory frames (decoded elsewhere, webcam, generated video)

```python
frames = [Image.fromarray(f) for f in my_numpy_frames]          # list of PIL images
e = model.embed_video(frames)                                   # treated as one video
e2 = model.embed_video([frames_a, frames_b], mode="native")     # two videos
```

### Long videos: embed per segment

Neither mode is meant for minutes-long clips in a single embedding. Split the video into shots or fixed windows, embed each one, then compare windows or pool them:

```python
segments = ["clip_000.mp4", "clip_001.mp4", "clip_002.mp4"]     # e.g. from ffmpeg -f segment
seg_emb = model.embed_video(segments, micro_batch_size=16)      # [S, 384]
video_emb = torch.nn.functional.normalize(seg_emb.mean(0, keepdim=True), dim=-1)
```

### Per-frame relational timeline

To see how the relational content changes over time, embed the frames individually:

```python
frames = model._sample_frames("clip.mp4", 16)                   # 16 uniformly sampled PIL frames
per_frame = model.embed(frames)                                 # [16, 384]
ref = model.embed(preprocess(Image.open("reference.jpg")))
timeline = (per_frame @ ref.T).squeeze(-1)                      # similarity of each frame to the reference
```

---

## Memory and speed <a name="memory-and-speed"></a>

| Knob                        | Effect                                                         |
|-----------------------------|----------------------------------------------------------------|
| `nframes`                   | Linear in cost for both modes. 8 is a good default; 4 for quick scans |
| `max_image_size` (frames)   | Caps frame resolution, e.g. `448`                              |
| `max_pixels` (native)       | Caps per-frame tokens, e.g. `360*420`. **Must be ≥ `128*28*28` (≈100k)**, which is `qwen_vl_utils`' video minimum; smaller values raise an error |
| `micro_batch_size`          | Splits the work into chunks to avoid OOM; results are identical |

Rough native-mode token count per video: `(nframes / 2) × (H/28) × (W/28)` after resizing. Keep it to a few thousand tokens per video on a 48 GB GPU.

---

## How it works <a name="how-it-works"></a>

RelSim appends a learned `<|query|>` token after the visual input. The token's final hidden state goes through a linear head to 384-d. That head was trained contrastively against all-MiniLM-L6-v2 embeddings of **anonymous captions**, such as *"A curious {Animal} peeking from behind a {Barrier}"*.

- **`frames` mode** reuses that exact image pipeline once per frame and averages the results. A video's embedding is then roughly "the average relational caption across its frames".
- **`native` mode** replaces the image with a Qwen2.5-VL video input (`pixel_values_videos`, `video_grid_thw`, `second_per_grid_ts`). `QwenWithQueryToken.forward` passes these through only when they are present, so the image path is untouched.
- The LoRA only targets the language model's `q_proj` / `v_proj`. Qwen's vision encoder is stock, so it encodes video frames exactly as base Qwen2.5-VL does. The only out-of-distribution part is how the LoRA'd language model reads a video token sequence.

---

## Limitations <a name="limitations"></a>

- **Zero-shot.** There are no video training data, no video benchmark, and no calibration. Absolute scores are not comparable to image scores; use them for ranking.
- **`frames` mode ignores time.** "Pouring water into a glass" and "a glass emptying" can look the same.
- **`native` mode may be noisy** because it is out of distribution. Check it on your data before trusting it.
- **No audio.**
- **Long clips** need segmenting (see [Recipes](#recipes)).

---

## Testing <a name="testing"></a>

```bash
python -m pytest tests/ -v                       # mocked, CPU, no checkpoint
python -m pytest tests/test_embed_video.py -v    # video tests only
```

`tests/test_embed_video.py` covers input normalization, frame sampling, pooling, batching and micro-batching, video tensor plumbing in native mode, and a guard that image calls never receive video arguments. It uses fake models like `tests/test_embed_batch.py`, so it runs in seconds.

---

## Roadmap: training a video RelSim <a name="roadmap"></a>

The training objective carries over to video directly. Only the data changes:

1. **Collect clips** and write **anonymous video captions** that describe relations over time, e.g. *"A {Agent} repeatedly fails at {Task} before succeeding"*. Options: prompt Qwen2.5-VL or a GPT model on the clips, or run the [anonymous caption model](anonymous_caption/) on keyframes and merge the results.
2. **Start from the image LoRA** (`thaoshibe/relsim-qwenvl25-lora`) and swap the image inputs in `relsim/train_relsim.py` for video inputs (the same keys `embed_video(mode="native")` uses).
3. **Keep the MiniLM text targets.** Video and image embeddings then stay in one shared space, so image↔video retrieval keeps working.
4. **Mix in image batches** so image performance doesn't regress.

Contributions are welcome.
