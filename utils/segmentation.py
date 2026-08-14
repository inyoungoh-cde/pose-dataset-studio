"""
utils/segmentation.py
---------------------

Promptable segmentation backends shared by `3a_segment_object_masks.py` and the
browser picker in `utils/picker.py`.

Two models, both loaded from the Hugging Face hub on first use and cached in
`~/.cache/huggingface` afterwards:

* **Grounding DINO** (`IDEA-Research/grounding-dino-base`) turns a free-text
  phrase into boxes. It is open-vocabulary, so the object does not have to
  belong to a fixed class list -- "the white product box", "a black DSLR
  camera" and so on all work. The `transformers` implementation is used rather
  than the original repository because it is pure PyTorch: nothing has to be
  compiled against the local CUDA toolkit.
* **SAM 2.1** (`facebook/sam2.1-hiera-large`) turns a box or a few clicks into
  a pixel-accurate mask, and -- this is the part that matters here -- carries
  that mask across the rest of the sequence with its own memory bank, so only
  one frame ever has to be prompted.

Nothing in this module opens a window; every entry point takes and returns
arrays. That is what lets the same code run on a headless server and on a
laptop.
"""

import numpy as np
import cv2

DEFAULT_DETECTOR = "IDEA-Research/grounding-dino-base"
DEFAULT_SAM2 = "facebook/sam2.1-hiera-large"

# Alternatives, smaller/faster, same API: 'IDEA-Research/grounding-dino-tiny',
# 'facebook/sam2.1-hiera-base-plus', 'facebook/sam2.1-hiera-small'.


def pick_device(spec="auto"):
    """Resolve --device. 'auto' prefers CUDA, then Apple MPS, then CPU."""
    import torch
    if spec != "auto":
        return spec
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def enable_fast_math(device):
    """TF32 matmuls on Ampere+; a few times faster and irrelevant to mask quality."""
    import torch
    if device.startswith("cuda") and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


class Autocast(object):
    """bfloat16 autocast on CUDA, a no-op everywhere else."""

    def __init__(self, device):
        import torch
        self.ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if device.startswith("cuda") else None)

    def __enter__(self):
        if self.ctx is not None:
            self.ctx.__enter__()
        return self

    def __exit__(self, *exc):
        if self.ctx is not None:
            return self.ctx.__exit__(*exc)
        return False


# --------------------------------------------------------------------------
# Grounding DINO -- text -> boxes
# --------------------------------------------------------------------------

class TextDetector(object):
    def __init__(self, model_id=DEFAULT_DETECTOR, device="cuda"):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.model.eval()

    def detect(self, rgb, prompt, box_threshold=0.30, text_threshold=0.25):
        """Boxes for `prompt` in an RGB uint8 image, best score first.

        Returns a list of {'box': (x0,y0,x1,y1) float pixels, 'score': float,
        'label': str}.
        """
        import torch
        text = prompt.strip()
        if not text.endswith("."):
            # Grounding DINO expects phrases terminated by '.', otherwise the
            # last phrase is silently dropped.
            text += "."
        inputs = self.processor(images=rgb, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=box_threshold,
            text_threshold=text_threshold, target_sizes=[rgb.shape[:2]])[0]
        boxes = res["boxes"].detach().cpu().numpy()
        scores = res["scores"].detach().cpu().numpy()
        # 'text_labels' since transformers 4.51; 'labels' before that returned the
        # same strings (and warns when merely looked up on newer versions).
        labels = res["text_labels"] if "text_labels" in res else res.get("labels")
        if labels is None:
            labels = [""] * len(scores)
        labels = [str(l) for l in labels]
        order = np.argsort(-scores)
        return [{"box": tuple(float(v) for v in boxes[i]),
                 "score": float(scores[i]),
                 "label": labels[i]} for i in order]


# --------------------------------------------------------------------------
# SAM 2 -- box/clicks -> mask, on a single image
# --------------------------------------------------------------------------

class ImageSegmenter(object):
    def __init__(self, model_id=DEFAULT_SAM2, device="cuda"):
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.device = device
        self.predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)
        self._image_id = None

    def set_image(self, rgb, image_id=None):
        """Encode an image once; repeated prompts on it are then nearly free."""
        if image_id is None or image_id != self._image_id:
            with Autocast(self.device):
                self.predictor.set_image(rgb)
            self._image_id = image_id
        return self

    def segment(self, box=None, points=None, labels=None):
        """Mask for the given prompt.

        box    : (x0, y0, x1, y1) in pixels, or None
        points : (n, 2) click coordinates in pixels, or None
        labels : (n,) 1 = foreground click, 0 = background click

        Returns (mask bool (h, w), score float).
        """
        kw = {}
        if box is not None:
            kw["box"] = np.asarray(box, dtype=np.float32)[None, :]
        if points is not None and len(points):
            kw["point_coords"] = np.asarray(points, dtype=np.float32)
            kw["point_labels"] = np.asarray(labels, dtype=np.int32)
        if not kw:
            raise ValueError("segment() needs a box or at least one point")
        # multimask only helps a bare single click, where the intended scale
        # (part / object / group) is ambiguous; take SAM's best of the three.
        multi = "box" not in kw and len(kw.get("point_coords", [])) == 1
        with Autocast(self.device):
            masks, scores, _ = self.predictor.predict(multimask_output=multi, **kw)
        best = int(np.argmax(scores))
        return masks[best].astype(bool), float(scores[best])


# --------------------------------------------------------------------------
# SAM 2 -- mask propagation across the sequence
# --------------------------------------------------------------------------

def load_video_predictor(model_id=DEFAULT_SAM2, device="cuda"):
    from sam2.build_sam import build_sam2_video_predictor_hf
    return build_sam2_video_predictor_hf(model_id, device=device)


# --------------------------------------------------------------------------
# Mask post-processing
# --------------------------------------------------------------------------

def largest_component(mask):
    """Keep only the biggest connected blob -- drops speckle a prompt picked up."""
    mask = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 2:
        return mask.astype(bool)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == biggest


def marker_mask(bgr, dilate_px=3):
    """Pixels covered by ArUco markers, so they can be cut out of the object mask.

    The object stands on the marker board, and a mask that leaks a few
    millimetres onto the board drags the board's points into the object cloud,
    which then biases the oriented bounding box that step 4 fits. Markers are
    the one region we can positively identify as *not* the object.
    """
    from utils.markers import detect_markers
    corners, ids, _ = detect_markers(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    m = np.zeros(bgr.shape[:2], dtype=np.uint8)
    if ids is None or not len(ids):
        return m.astype(bool)
    for c in corners:
        cv2.fillConvexPoly(m, np.round(c[0]).astype(np.int32), 1)
    if dilate_px > 0:
        k = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
        m = cv2.dilate(m, k)
    return m.astype(bool)


def erode(mask, px):
    """Shrink a mask by `px` pixels (silhouette pixels mix object and background depth)."""
    if px <= 0:
        return mask
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return cv2.erode(mask.astype(np.uint8), k).astype(bool)


def bbox_of(mask):
    """(x0, y0, x1, y1) of a boolean mask, or None if it is empty."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def iou(a, b):
    """IoU of two (x0, y0, x1, y1) boxes."""
    if a is None or b is None:
        return 0.0
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def overlay(bgr, mask, color=(0, 255, 0), alpha=0.45, box=None, points=None,
            labels=None, text=None):
    """Mask tint + outline over a BGR frame, for the previews written to disk."""
    out = bgr.copy()
    if mask is not None and mask.any():
        sel = mask.astype(bool)
        out[sel] = (alpha * np.array(color, dtype=np.float32)
                    + (1 - alpha) * out[sel]).astype(np.uint8)
        cnts = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        cv2.drawContours(out, cnts, -1, color, 2)
    if box is not None:
        p0 = (int(round(box[0])), int(round(box[1])))
        p1 = (int(round(box[2])), int(round(box[3])))
        cv2.rectangle(out, p0, p1, (0, 200, 255), 2)
    if points is not None:
        for (x, y), l in zip(points, labels if labels is not None else [1] * len(points)):
            cv2.circle(out, (int(round(x)), int(round(y))), 7,
                       (0, 255, 0) if l else (0, 0, 255), -1)
            cv2.circle(out, (int(round(x)), int(round(y))), 7, (255, 255, 255), 2)
    if text:
        cv2.putText(out, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out
