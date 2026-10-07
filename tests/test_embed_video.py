"""
Tests for ``relsim.embed_video`` (frames + native modes).

Like ``test_embed_batch.py``, these use lightweight fakes so they run on CPU
without the 7B checkpoint. They check input normalization, frame sampling,
pooling, batching and that native mode forwards the video tensors.

    python -m pytest tests/test_embed_video.py -v
"""
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

import relsim.relsim_score as rs  # noqa: E402
from relsim.relsim_score import _RelSimClass  # noqa: E402

from test_embed_batch import (  # noqa: E402
    EMB_DIM, _FakeBaseModel, _FakeProcessor, _img_key, _rgb,
)


class _FakeVideoProcessor(_FakeProcessor):
    """Adds a video path: each video -> one row keyed by its first frame."""

    def __init__(self):
        super().__init__()
        self.video_calls = []

    def __call__(self, text=None, images=None, videos=None, padding=True,
                 return_tensors="pt", **kwargs):
        if videos is None:
            return super().__call__(text=text, images=images, padding=padding,
                                    return_tensors=return_tensors)
        self.video_calls.append({"n": len(videos), "lens": [len(v) for v in videos], **kwargs})
        b, seq_len = len(videos), 5
        input_ids = torch.ones((b, seq_len), dtype=torch.long)
        for i, frames in enumerate(videos):
            input_ids[i, 0] = _img_key(frames[0]) + len(frames)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones((b, seq_len), dtype=torch.long),
            "pixel_values_videos": torch.zeros((b, 3)),
            "video_grid_thw": torch.ones((b, 3), dtype=torch.long),
            "second_per_grid_ts": [1.0] * b,
        }


class _RecordingBaseModel(_FakeBaseModel):
    def __init__(self):
        super().__init__()
        self.kwargs_seen = []

    def __call__(self, **kwargs):
        self.kwargs_seen.append(set(kwargs))
        return super().__call__(**kwargs)


def _fake_process_vision_info(messages, return_video_kwargs=False):
    ele = messages[0]["content"][0]
    if "image" in ele:
        return [ele["image"]], None
    video = ele["video"]
    assert isinstance(video, list), "path inputs are patched via fetch_video"
    return None, [video], {"fps": [2.0], "do_sample_frames": False}


def _video(color_base, n=10):
    return [_rgb((color_base, i * 20, 0)) for i in range(n)]


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(rs, "process_vision_info", _fake_process_vision_info)
    return _RelSimClass(_RecordingBaseModel(), _FakeVideoProcessor())


# --------------------------------------------------------------------------- #
# input normalization / frame sampling
# --------------------------------------------------------------------------- #
def test_as_video_list_variants():
    frames = _video(1, 3)
    assert _RelSimClass._as_video_list("a.mp4") == ["a.mp4"]
    assert _RelSimClass._as_video_list(frames) == [frames]
    assert _RelSimClass._as_video_list(["a.mp4", frames]) == ["a.mp4", frames]
    with pytest.raises(TypeError):
        _RelSimClass._as_video_list(123)


def test_sample_frames_uniform_and_capped():
    frames = _video(1, 10)
    out = _RelSimClass._sample_frames(frames, 4)
    assert len(out) == 4
    assert out[0] is not None and _img_key(out[0]) == _img_key(frames[0])
    assert _img_key(out[-1]) == _img_key(frames[-1])
    assert len(_RelSimClass._sample_frames(frames[:2], 8)) == 2


def test_sample_frames_from_path_uses_fetch_video(monkeypatch):
    monkeypatch.setattr(rs, "fetch_video",
                        lambda ele: torch.full((ele["nframes"], 3, 4, 4), 128.0))
    out = _RelSimClass._sample_frames("clip.mp4", 6)
    assert len(out) == 6 and all(f.mode == "RGB" and f.size == (4, 4) for f in out)


def test_sample_frames_bad_input_raises():
    with pytest.raises(TypeError):
        _RelSimClass._sample_frames([], 4)


# --------------------------------------------------------------------------- #
# frames mode
# --------------------------------------------------------------------------- #
def test_frames_mode_shape_norm_and_pooling(model):
    v1, v2 = _video(1), _video(2)
    out = model.embed_video([v1, v2], mode="frames", nframes=4)
    assert out.shape == (2, EMB_DIM)
    assert torch.allclose(out.norm(dim=-1), torch.ones(2), atol=1e-5)
    # equals normalized mean of the per-frame image embeddings
    per = model.embed(_RelSimClass._sample_frames(v1, 4))
    expected = torch.nn.functional.normalize(per.mean(0), dim=-1)
    assert torch.allclose(out[0], expected, atol=1e-5)


def test_frames_mode_single_video_is_one_batched_forward(model):
    model.embed_video(_video(1), mode="frames", nframes=4)
    assert model.base_model.batch_sizes == [4]


def test_frames_mode_micro_batch_matches(model):
    vids = [_video(1), _video(2), _video(3)]
    full = model.embed_video(vids, mode="frames", nframes=4)
    chunked = model.embed_video(vids, mode="frames", nframes=4, micro_batch_size=5)
    assert torch.allclose(full, chunked, atol=1e-6)


# --------------------------------------------------------------------------- #
# native mode
# --------------------------------------------------------------------------- #
def test_native_mode_forwards_video_tensors(model):
    out = model.embed_video([_video(1), _video(2)], mode="native", nframes=4)
    assert out.shape == (2, EMB_DIM)
    assert torch.allclose(out.norm(dim=-1), torch.ones(2), atol=1e-5)
    seen = model.base_model.kwargs_seen[-1]
    assert {"pixel_values_videos", "video_grid_thw", "second_per_grid_ts"} <= seen
    call = model.processor.video_calls[-1]
    assert call["n"] == 2 and call["fps"] == [2.0, 2.0] and call["do_sample_frames"] is False


def test_native_mode_subsamples_frame_lists(model):
    model.embed_video(_video(1, 20), mode="native", nframes=4)
    assert model.processor.video_calls[-1]["lens"] == [4]


def test_native_mode_micro_batch(model):
    vids = [_video(1), _video(2), _video(3)]
    full = model.embed_video(vids, mode="native")
    model.base_model.batch_sizes = []
    chunked = model.embed_video(vids, mode="native", micro_batch_size=2)
    assert model.base_model.batch_sizes == [2, 1]
    assert torch.allclose(full, chunked, atol=1e-6)


def test_bad_mode_raises(model):
    with pytest.raises(ValueError):
        model.embed_video(_video(1), mode="nope")


def test_image_embed_unchanged(model):
    """Image path must not receive any video kwargs."""
    model.embed(_rgb("red"))
    assert "pixel_values_videos" not in model.base_model.kwargs_seen[-1]
