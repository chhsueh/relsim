import torch
import torch.nn.functional as F
import os
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info, fetch_video
from PIL import Image
from huggingface_hub import hf_hub_download

# Import model from training script
import sys
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .relsim_model import QwenWithQueryToken

class relsim:
    """RelSim perceptual similarity model"""
    
    def __init__(self, base_model, processor):
        self.base_model = base_model
        self.processor = processor
        self.device = next(base_model.parameters()).device
    
    def embed(self, images, max_image_size=None, micro_batch_size=None):
        """
        Extract RelSim embedding(s) for one image or a list of images.

        Behaves like CLIP's ``get_image_features``: pass a single PIL Image or a
        list of PIL Images and get back a normalized embedding tensor computed in
        a single (batched) forward pass.

        Args:
            images: A PIL Image, or a list of PIL Images.
            max_image_size: If set, cap each image's longest edge to this many
                pixels (LANCZOS) before inference to avoid OOM. Default: None.
            micro_batch_size: If set, split a long list into chunks of this many
                images per forward pass and concatenate the results. Default:
                None (all images in one forward pass, like CLIP).

        Returns:
            Normalized embedding tensor of shape ``[N, 384]``. ``N == 1`` for a
            single-image input.

        Note:
            RelSim is image-only; there is no text-embedding counterpart to
            CLIP's ``get_text_features``. The ``<|query|>`` token is fixed.
        """
        # Accept a single image or a list; normalize to a list internally.
        if isinstance(images, Image.Image):
            images = [images]

        # Ensure RGB (and optionally cap resolution) per image.
        prepared = []
        for img in images:
            if not isinstance(img, Image.Image):
                raise TypeError("Each image must be a PIL.Image.Image")
            img = img.convert("RGB")
            if max_image_size and (img.width > max_image_size or img.height > max_image_size):
                img = img.copy()
                img.thumbnail((max_image_size, max_image_size), Image.LANCZOS)
            prepared.append(img)

        # One forward pass, or chunked forwards concatenated.
        if micro_batch_size and micro_batch_size > 0:
            outputs = [
                self._embed_batch(prepared[i:i + micro_batch_size])
                for i in range(0, len(prepared), micro_batch_size)
            ]
            return torch.cat(outputs, dim=0)
        return self._embed_batch(prepared)

    def _embed_batch(self, images):
        """Run one batched forward pass over a list of RGB PIL images -> [N, 384]."""
        texts = []
        all_image_inputs = []
        for img in images:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "<|query|>"},
                    ],
                }
            ]
            texts.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))
            image_inputs, _ = process_vision_info(messages)
            if image_inputs:
                all_image_inputs.extend(image_inputs)

        inputs = self.processor(
            text=texts,
            images=all_image_inputs if all_image_inputs else None,
            padding=True,
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            embeddings = self.base_model(**inputs)
            embeddings = embeddings.float()
            embeddings = F.normalize(embeddings, dim=-1)

        return embeddings

    # ------------------------------------------------------------------ video
    @staticmethod
    def _as_video_list(videos):
        """Normalize input to a list of videos.

        A single video is a path/URL string or a list of PIL frames; anything
        else is treated as a list of such videos.
        """
        if isinstance(videos, str):
            return [videos]
        if isinstance(videos, (list, tuple)) and videos and all(
            isinstance(f, Image.Image) for f in videos
        ):
            return [list(videos)]
        if isinstance(videos, (list, tuple)):
            return list(videos)
        raise TypeError("videos must be a path, a list of PIL frames, or a list of those")

    @staticmethod
    def _sample_frames(video, nframes):
        """Return ``nframes`` uniformly sampled RGB PIL frames from a video."""
        if isinstance(video, str):
            frames = fetch_video({"video": video, "nframes": nframes})  # [T, C, H, W]
            return [
                Image.fromarray(f.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy())
                for f in frames
            ]
        if not video or not all(isinstance(f, Image.Image) for f in video):
            raise TypeError("Each video must be a path or a non-empty list of PIL Images")
        idx = torch.linspace(0, len(video) - 1, min(nframes, len(video))).round().long().tolist()
        return [video[i].convert("RGB") for i in idx]

    def embed_video(self, videos, mode="frames", nframes=8, fps=None,
                    max_pixels=None, max_image_size=None, micro_batch_size=None):
        """
        Extract RelSim embedding(s) for one video or a list of videos -> ``[N, 384]``.

        Args:
            videos: A video path/URL, a list of PIL frames (one video), or a list
                of either.
            mode: ``"frames"`` (default) embeds ``nframes`` uniformly sampled
                frames as images and mean-pools them. This stays within the
                image distribution the model was trained on, but ignores
                temporal order. ``"native"`` feeds the clip to Qwen2.5-VL as a
                video and reads the ``<|query|>`` token, so it can see motion.
                It is zero-shot: the released LoRA never saw video tokens.
            nframes: Frames sampled per video (rounded to an even number in
                native mode, as Qwen requires).
            fps: Native mode only. Sample at this rate instead of ``nframes``.
            max_pixels: Native mode only. Per-frame pixel budget, e.g.
                ``360 * 420``. Lower it for long clips to avoid OOM, but keep
                it at or above qwen_vl_utils' video minimum (128 * 28 * 28).
            max_image_size: Frames mode only. Same as in ``embed``.
            micro_batch_size: Frames mode: frames per forward pass. Native
                mode: videos per forward pass.

        Returns:
            L2-normalized tensor of shape ``[N, 384]``, comparable with image
            embeddings from ``embed`` by dot product.
        """
        vids = self._as_video_list(videos)
        if mode == "frames":
            frame_lists = [self._sample_frames(v, nframes) for v in vids]
            flat = [f for frames in frame_lists for f in frames]
            per_frame = self.embed(flat, max_image_size=max_image_size,
                                   micro_batch_size=micro_batch_size)
            pooled, start = [], 0
            for frames in frame_lists:
                pooled.append(per_frame[start:start + len(frames)].mean(dim=0))
                start += len(frames)
            return F.normalize(torch.stack(pooled, dim=0), dim=-1)
        if mode == "native":
            step = micro_batch_size if micro_batch_size and micro_batch_size > 0 else len(vids)
            return torch.cat([
                self._embed_video_batch(vids[i:i + step], nframes, fps, max_pixels)
                for i in range(0, len(vids), step)
            ], dim=0)
        raise ValueError(f"mode must be 'frames' or 'native', got {mode!r}")

    def _embed_video_batch(self, videos, nframes, fps, max_pixels):
        """One batched forward pass over a list of videos (native mode) -> [N, 384]."""
        texts, video_inputs, sample_fps = [], [], []
        for v in videos:
            ele = {"type": "video", "video": v}
            if isinstance(v, str):
                ele.update({"fps": fps} if fps else {"nframes": nframes})
            else:
                ele["video"] = self._sample_frames(v, nframes)
            if max_pixels:
                ele["max_pixels"] = max_pixels
            messages = [{"role": "user", "content": [ele, {"type": "text", "text": "<|query|>"}]}]
            texts.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))
            _, vid, vkw = process_vision_info(messages, return_video_kwargs=True)
            video_inputs.extend(vid)
            sample_fps.extend(vkw.get("fps", []))

        inputs = self.processor(
            text=texts,
            videos=video_inputs,
            fps=sample_fps,
            do_sample_frames=False,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            embeddings = self.base_model(**inputs)
            embeddings = F.normalize(embeddings.float(), dim=-1)
        return embeddings

    def __call__(self, img1, img2):
        """
        Compute perceptual similarity between two preprocessed images.
        
        Args:
            img1: First preprocessed PIL Image
            img2: Second preprocessed PIL Image
            
        Returns:
            Similarity score (higher = more similar)
        """
        # Embed both images in a single batched forward pass.
        embeddings = self.embed([img1, img2])

        # Cosine similarity (embeddings are normalized, so it's dot product)
        similarity = (embeddings[0] * embeddings[1]).sum().item()

        return similarity
    
    def eval(self):
        """Set model to evaluation mode"""
        self.base_model.eval()
        return self


def preprocess_image(img):
    """
    Preprocess function for images.

    Accepts a single PIL Image or a list of PIL Images, mirroring the batched
    ``embed`` API (list in -> list out).

    Args:
        img: A PIL Image, or a list of PIL Images

    Returns:
        An RGB PIL Image, or a list of RGB PIL Images (matching the input type)
    """
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    elif isinstance(img, (list, tuple)):
        return [preprocess_image(x) for x in img]
    else:
        raise TypeError("Input must be a PIL Image or a list of PIL Images")


# Store reference to class before function shadows the name
_RelSimClass = relsim


def relsim(pretrained=True, checkpoint_dir=None, cache_dir=None):
    """
    Load RelSim model and preprocessing function.
    
    Args:
        pretrained: If True, load pretrained model (requires checkpoint_dir)
        checkpoint_dir: Path to trained model checkpoint OR HuggingFace model ID
                       Examples: "path/to/checkpoint" or "thaoshibe/relsim-qwenvl25-lora"
        cache_dir: Cache directory for model files
        
    Returns:
        model: relsim instance
        preprocess: Preprocessing function
        
    Example:
        >>> from relsim import relsim
        >>> from PIL import Image
        >>> 
        >>> # Load from HuggingFace
        >>> model, preprocess = relsim(pretrained=True, checkpoint_dir="thaoshibe/relsim-qwenvl25-lora")
        >>> 
        >>> # Or load from local path
        >>> model, preprocess = relsim(pretrained=True, checkpoint_dir="path/to/checkpoint")
        >>> 
        >>> # Feature extraction (single image) -> [1, 384]
        >>> img1 = preprocess(Image.open("img1.jpg"))
        >>> embedding = model.embed(img1)
        >>>
        >>> # Feature extraction (batch, like CLIP) -> [N, 384]
        >>> imgs = preprocess([Image.open(p) for p in ["img1.jpg", "img2.jpg"]])
        >>> embeddings = model.embed(imgs)
        >>>
        >>> # Perceptual similarity
        >>> img1 = preprocess(Image.open("img1.jpg"))
        >>> img2 = preprocess(Image.open("img2.jpg"))
        >>> distance = model(img1, img2)
    """
    if not pretrained:
        raise ValueError("Only pretrained=True is supported")
    
    if checkpoint_dir is None:
        raise ValueError("checkpoint_dir must be provided")
    
    print(f"Loading checkpoint from: {checkpoint_dir}")
    
    # Load processor (includes query token from training)
    # Works with both local paths and HuggingFace model IDs
    processor = AutoProcessor.from_pretrained(
        checkpoint_dir, 
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    
    # Load base model
    base_model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    print(f"Loading base model: {base_model_name}")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir
    )
    
    # Resize token embeddings to match the processor (with added query token)
    print(f"Resizing token embeddings: {len(processor.tokenizer)} tokens")
    base_model.resize_token_embeddings(len(processor.tokenizer))
    
    # Load LoRA weights
    print("Loading LoRA weights...")
    # Load adapter weights to CPU first to avoid device issues
    # Works with both local paths and HuggingFace model IDs
    base_model = PeftModel.from_pretrained(
        base_model, 
        checkpoint_dir,
        torch_device="cpu"
    )
    
    # Wrap with query token
    wrapped_model = QwenWithQueryToken(base_model, processor, hidden_size=3584)
    
    # Load projection head
    print("Loading projection head...")
    # For HuggingFace models, this will be downloaded to cache
    # For local paths, this will be loaded directly
    
    # Check if it's a HuggingFace model ID (contains '/') or local path
    if '/' in checkpoint_dir and not os.path.exists(checkpoint_dir):
        # It's a HuggingFace model ID
        projection_path = hf_hub_download(
            repo_id=checkpoint_dir,
            filename="projection_and_config.pt",
            cache_dir=cache_dir
        )
    else:
        # It's a local path
        projection_path = os.path.join(checkpoint_dir, "projection_and_config.pt")
    
    checkpoint = torch.load(projection_path, map_location="cpu")
    wrapped_model.projection.load_state_dict(checkpoint['projection'])
    
    # Move projection to correct device and dtype
    try:
        param = next(base_model.parameters())
        wrapped_model.projection = wrapped_model.projection.to(device=param.device, dtype=param.dtype)
        print(f"✅ Projection head loaded: device={param.device}, dtype={param.dtype}")
    except StopIteration:
        pass
    
    wrapped_model.eval()
    print("✅ Model loaded successfully!")
    
    # Create API model
    model = _RelSimClass(wrapped_model, processor)
    
    return model, preprocess_image


# For backward compatibility - CLI usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute similarity score between two images")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--image1",
        type=str,
        required=True,
        help="Path to first image"
    )
    parser.add_argument(
        "--image2",
        type=str,
        required=True,
        help="Path to second image"
    )
    args = parser.parse_args()
    
    print(f"Image 1: {args.image1}")
    print(f"Image 2: {args.image2}")
    print("\n" + "="*60)
    print("Loading model...")
    print("="*60)
    
    # Load model using new API
    model, preprocess = relsim(pretrained=True, checkpoint_dir=args.checkpoint_dir)
    
    # Load and preprocess images
    print("\n" + "="*60)
    print("Loading images...")
    print("="*60)
    
    image1 = preprocess(Image.open(args.image1))
    image2 = preprocess(Image.open(args.image2))
    print("✅ Images loaded")
    
    # Compute similarity
    print("\n" + "="*60)
    print("Computing similarity...")
    print("="*60)
    
    similarity = model(image1, image2)
    
    print(f"✅ Similarity score: {similarity:.4f}")
    
    print("\n" + "="*60)
    print("🎉 Done!")
    print("="*60)