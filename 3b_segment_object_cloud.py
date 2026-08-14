"""
3b_segment_object_cloud.py
--------------------------

Turn the per-frame masks from `3a_segment_object_masks.py` into `object.ply` --
the file the manual CloudCompare step used to produce.

    python 3b_segment_object_cloud.py box                  # crop registeredScene.ply
    python 3b_segment_object_cloud.py box --source depth   # fuse the masked depth instead
    python 3b_segment_object_cloud.py box --min-observations 5 --min-ratio 0.8
    python 3b_segment_object_cloud.py box --gpu 0          # same output, on CUDA:0

Why this is safer than cutting the cloud by hand: the decision "is this point
part of the object?" is not taken once, in one view, by eye. Every frame votes
on every point, and a point is kept only if enough frames agree. A mask that
is wrong in a few frames is outvoted; a hand-drawn cut is not.

What a vote means, per frame and per point (poses come from step 2, so the
projection is exact):

  * project the point into the frame with that frame's pose;
  * compare its depth with the measured depth at that pixel. Further away than
    the measurement means something is in front of it -- the point is occluded
    in this view and the frame simply abstains, rather than voting it away;
  * only if the point lies *on* the measured surface does the frame vote, and
    then the mask decides: inside = object, outside = background.

Two sources are available:

  scene  (default) crop `registeredScene.ply` from step 3 -- exactly the points
         you would have cut by hand, minus the hand.
  depth  skip step 3 and fuse the masked depth of every frame directly. Denser
         on the object, because step 3's scene-wide vote filter and marker-board
         crop are tuned for the whole scene rather than for one object.

Outputs (inside the sequence folder):

    object.ply                     what step 4 consumes
    seg_preview/object_%06d.jpg    the result reprojected onto real frames,
                                   with its oriented bounding box drawn

`--gpu N` runs the same work on CUDA device N -- same inputs, same outputs,
same voting semantics, only faster. What it changes:

1. **The vote loop** (`--source scene`) projects every scene point into every
   selected frame with numpy on one core. On the GPU the scene is a float32
   tensor on the device, each frame is one matmul + projection + two flattened
   gathers (depth and mask), and the votes are int32 counters that never leave
   the device until the end. The rules are the CPU path's, bit for bit where
   dtypes allow:

       z > 1e-6, 0 <= u < w, 0 <= v < h    else the frame ignores the point
       measured = depth[v, u] * depth_scale, 0 -> ignore
       |z - measured| > --depth-tol        -> abstain (occluded in this view)
       else objmask[v, u] decides           object vote / background vote

   Pixel coords use the same truncation the CPU code's `.astype(np.int32)`
   performs (u, v are >= 0 there, so truncation == floor). The only permitted
   difference: the CPU compares depths in float64, the GPU in float32, so
   points sitting exactly on a threshold boundary may flip -- a vanishing
   fraction of the cloud.

2. **Frame loading** happens once, up front, on a thread pool.

3. **`--source depth`** uses `3_register_scene.py`'s `load_frame` +
   `gpu_post_process` (the CUDA vote-merge) instead of its CPU `post_process`,
   with the same auto merge radius rule (one depth pixel at the object's
   median range).

Everything after the vote -- min-observations/min-ratio keep rule, voxel
downsample, statistical outlier removal, largest-blob filter, OBB, PLY writer,
reprojection previews -- is shared by both paths.

If large scenes exceed GPU memory the point cloud is voted in chunks sized
from `torch.cuda.mem_get_info` (override with --chunk). If `--gpu` is given
and no CUDA device is available the script says so and exits; it never falls
back silently.
"""

import concurrent.futures as futures
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm

from config.registrationParameters import *
from utils.cli import build_parser, resolve_dataset
from utils.ply import Ply

HERE = os.path.dirname(os.path.abspath(__file__))


def load_step3():
    """The vote-based merge from `3_register_scene.py` (CPU and GPU flavours).

    Loaded by path because the pipeline scripts are named `3_...`, which is not
    a valid module name. Importing it is still better than pasting a second
    copy of the merge here: two copies would drift apart, and the whole point of
    `--source depth` is that its cloud is built the same way step 3 builds its
    scene.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("step3_register_scene",
                                                  os.path.join(HERE, "3_register_scene.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mask_frames(mask_dir):
    ids = []
    for f in glob.glob1(mask_dir, "*.png"):
        stem = os.path.splitext(f)[0]
        if stem.isdigit():
            ids.append(int(stem))
    return sorted(ids)


def read_mask(mask_dir, fid, erode_px):
    m = cv2.imread(os.path.join(mask_dir, "%06d.png" % fid), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    m = m > 127
    if erode_px > 0 and m.any():
        k = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
        m = cv2.erode(m.astype(np.uint8), k).astype(bool)
    return m


def read_depth(path, fid):
    d = cv2.imread(os.path.join(path, "depth", "%06d.png" % fid), cv2.IMREAD_UNCHANGED)
    return None if d is None else d.astype(np.float32)


def project(points_world, T_cam_to_world, K):
    """World points -> (u, v, z) in the camera of that frame."""
    Tinv = np.linalg.inv(T_cam_to_world)
    cam = points_world.dot(Tinv[:3, :3].T) + Tinv[:3, 3]
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * cam[:, 0] / z + K[0, 2]
        v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    return u, v, z


# The 12 edges of a box as index pairs into open3d's get_box_points() order.
OBB_EDGES = [(0, 1), (0, 2), (0, 3), (1, 6), (1, 7), (2, 5),
             (2, 7), (3, 5), (3, 6), (4, 5), (4, 6), (4, 7)]


def draw_result(bgr, points_world, T, K, obb):
    """Kept points + the oriented bounding box, drawn onto one real frame."""
    out = bgr.copy()
    h, w = out.shape[:2]
    u, v, z = project(points_world, T, K)
    ok = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    ui, vi = u[ok].astype(np.int32), v[ok].astype(np.int32)
    tint = out[vi, ui].astype(np.float32) * 0.35 + np.array([0, 255, 0], np.float32) * 0.65
    out[vi, ui] = tint.astype(np.uint8)
    corners = np.asarray(obb.get_box_points())
    cu, cv_, cz = project(corners, T, K)
    if np.all(cz > 0):
        pts = [(int(round(a)), int(round(b))) for a, b in zip(cu, cv_)]
        for i, j in OBB_EDGES:
            cv2.line(out, pts[i], pts[j], (0, 200, 255), 2, cv2.LINE_AA)
    return out


def require_cuda(source, gpu, dataset):
    """--gpu N: verify device N exists, or exit with a clear message.

    The scene vote needs torch CUDA, the depth merge Open3D's CUDA build.
    """
    hint = ("Drop --gpu to run the CPU path:\n"
            "  python 3b_segment_object_cloud.py %s" % dataset)
    if source == "scene":
        import torch
        who, ok = "torch", torch.cuda.is_available()
        count = torch.cuda.device_count() if ok else 0
    else:
        import open3d.core as o3c
        who, ok = "Open3D", o3c.cuda.is_available()
        count = o3c.cuda.device_count() if ok else 0
    if not ok:
        sys.exit("--gpu %d was given, but %s sees no CUDA device on this machine.\n%s"
                 % (gpu, who, hint))
    if not 0 <= gpu < count:
        sys.exit("--gpu %d: no such CUDA device -- %s sees %d (0..%d).\n%s"
                 % (gpu, who, count, count - 1, hint))


def main():
    parser = build_parser("Mask-guided 3D crop: objmask/ + poses -> object.ply")
    parser.add_argument("--source", choices=["auto", "scene", "depth"], default="auto",
                        help="'scene' crops registeredScene.ply from step 3; 'depth' fuses the "
                             "masked depth directly and does not need step 3 at all; 'auto' "
                             "uses the scene if it is there")
    parser.add_argument("--input-scene", default="registeredScene.ply",
                        help="the merged scene to crop (--source scene)")
    parser.add_argument("--output", default="object.ply",
                        help="written inside the sequence folder; step 4 reads object.ply")
    parser.add_argument("--mask-dir", default="objmask",
                        help="per-frame masks from 3a_segment_object_masks.py")
    parser.add_argument("--interval", type=int, default=RECONSTRUCTION_INTERVAL,
                        help="use every Nth frame. More frames = more votes and a denser cloud, "
                             "at a linear cost in time")
    parser.add_argument("--label-interval", type=int, default=LABEL_INTERVAL,
                        help="frame interval used in 2_compute_gt_poses.py (frame i uses pose "
                             "i / this)")
    parser.add_argument("--depth-tol", type=float, default=0.03,
                        help="how far (m) a point may sit from the measured depth at its pixel "
                             "and still count as lying on that surface. Below this a frame "
                             "votes; beyond it the point is behind the measurement, i.e. "
                             "occluded, and the frame abstains")
    parser.add_argument("--erode", type=int, default=2,
                        help="shrink each mask by this many pixels before using it. Silhouette "
                             "pixels mix object and background depth, and the background is "
                             "metres behind the object")
    parser.add_argument("--min-observations", type=int, default=3,
                        help="keep a point only if at least this many frames saw it on the "
                             "surface and inside the mask")
    parser.add_argument("--min-ratio", type=float, default=0.6,
                        help="keep a point only if this fraction of the frames that voted on it "
                             "voted 'object'")
    parser.add_argument("--voxel", type=float, default=0.002,
                        help="final downsample spacing (m); 0 disables it. 2 mm is what the "
                             "manual CloudCompare step used")
    parser.add_argument("--no-cluster-filter", dest="cluster_filter", action="store_false",
                        help="keep every blob instead of only the largest connected one")
    parser.add_argument("--cluster-eps", type=float, default=0.01,
                        help="neighbour distance (m) for the connected-blob filter")
    parser.add_argument("--min-cluster-points", type=int, default=50,
                        help="ignore blobs smaller than this when picking the largest one")
    parser.add_argument("--no-outlier-removal", dest="outlier_removal", action="store_false",
                        help="skip the statistical outlier removal")
    parser.add_argument("--min-votes", type=int, default=2,
                        help="--source depth only: confirming observations required by the "
                             "merge, as in 3_register_scene.py")
    parser.add_argument("--preview-frames", type=int, default=4,
                        help="how many frames to reproject the result onto (0 = none)")
    parser.add_argument("--gpu", nargs="?", const=0, default=None, type=int, metavar="N",
                        help="run the vote / merge on CUDA device N (a bare --gpu means "
                             "device 0). Omit it and the CPU path runs, unchanged")
    parser.add_argument("--chunk", type=int, default=0,
                        help="points per vote chunk; 0 = auto from free GPU memory (--gpu only)")
    parser.add_argument("--io-threads", type=int, default=8,
                        help="parallel frame decoders (--gpu only)")
    args = parser.parse_args()

    dataset, path = resolve_dataset(
        args, require=("intrinsics.json", "transforms.npy", "JPEGImages", "depth", args.mask_dir))

    use_gpu = args.gpu is not None
    device = "CUDA:%d" % args.gpu if use_gpu else None
    t_start = time.perf_counter()
    timings = {"load": 0.0, "vote": 0.0}

    mask_dir = os.path.join(path, args.mask_dir)
    prev_dir = os.path.join(path, "seg_preview")
    if not os.path.isdir(prev_dir):
        os.makedirs(prev_dir)

    with open(os.path.join(path, "intrinsics.json")) as f:
        intr = json.load(f)
    K = np.array([[float(intr["fx"]), 0, float(intr["ppx"])],
                  [0, float(intr["fy"]), float(intr["ppy"])],
                  [0, 0, 1]], dtype=np.float64)
    scale = float(intr["depth_scale"])
    width, height = int(intr["width"]), int(intr["height"])

    Ts = np.load(os.path.join(path, "transforms.npy"))
    label_interval = args.label_interval

    have_masks = mask_frames(mask_dir)
    if not have_masks:
        sys.exit("No masks in %s -- run 3a_segment_object_masks.py first." % mask_dir)
    meta_file = os.path.join(mask_dir, "seg_meta.json")
    if os.path.exists(meta_file):
        with open(meta_file) as f:
            meta = json.load(f)
        empty = set(meta.get("empty_frames", []))
        if empty:
            print("Skipping %d frame(s) 3a reported as having no mask." % len(empty))
        have_masks = [f for f in have_masks if f not in empty]
        if meta.get("prompt", {}).get("text"):
            print("Masks were produced from the prompt %r" % meta["prompt"]["text"])

    # --interval counts frames of the *sequence*, but 3a may have masked only
    # every Nth of them, so it cannot be applied to the mask list directly.
    mask_stride = meta_stride(mask_dir)
    step = max(1, args.interval // mask_stride)
    frames = [f for f in have_masks[::step]
              if f % label_interval == 0 and f // label_interval < len(Ts)]
    if not frames:
        sys.exit("No frame has both a mask and a pose. Check --interval / --label-interval.")
    effective = step * mask_stride
    print("Using %d of the %d masked frames (every %dth frame of the sequence%s)"
          % (len(frames), len(have_masks), effective,
             "" if effective == args.interval else
             "; --interval %d was rounded up to the %d-frame mask stride"
             % (args.interval, mask_stride)))

    source = args.source
    scene_file = os.path.join(path, args.input_scene)
    if source == "auto":
        source = "scene" if os.path.exists(scene_file) else "depth"
        print("Source: %s (auto)" % source)
    if source == "scene" and not os.path.exists(scene_file):
        sys.exit("%s not found -- run 3_register_scene.py, or use --source depth." % scene_file)

    if use_gpu:
        require_cuda(source, args.gpu, args.dataset or dataset)
        print("Device: %s" % device)

    if source == "scene":
        crop = gpu_crop_scene if use_gpu else crop_scene
        points, colors = crop(scene_file, path, mask_dir, frames, Ts, K, scale,
                              width, height, label_interval, args, timings)
    else:
        fuse = gpu_fuse_depth if use_gpu else fuse_depth
        points, colors = fuse(path, mask_dir, frames, Ts, K, scale, args, timings)

    if not len(points):
        sys.exit("Nothing survived. Look at seg_preview/ first: if the masks are right, relax "
                 "--min-observations / --min-ratio, or raise --depth-tol (the poses and the "
                 "depth may disagree by more than %.0f mm)." % (args.depth_tol * 1000))

    t0 = time.perf_counter()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))

    if args.voxel > 0:
        before = len(pcd.points)
        pcd = pcd.voxel_down_sample(args.voxel)
        print("Downsampled to %.0f mm spacing: %d -> %d points"
              % (args.voxel * 1000, before, len(pcd.points)))
    if args.outlier_removal and len(pcd.points) > 30:
        before = len(pcd.points)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print("Statistical outlier removal: %d -> %d points" % (before, len(pcd.points)))
    if args.cluster_filter and len(pcd.points) > args.min_cluster_points:
        labels = np.array(pcd.cluster_dbscan(eps=args.cluster_eps,
                                             min_points=10, print_progress=False))
        if (labels >= 0).any():
            sizes = np.bincount(labels[labels >= 0])
            biggest = int(np.argmax(sizes))
            keep = labels == biggest
            dropped = int((~keep).sum())
            if dropped:
                others = int((sizes >= args.min_cluster_points).sum()) - 1
                print("Connected-blob filter: kept the largest blob, dropped %d point(s) "
                      "(%d other blob(s) above --min-cluster-points)" % (dropped, max(0, others)))
            pcd = pcd.select_by_index(np.nonzero(keep)[0])

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    if not len(points):
        sys.exit("Every point was filtered out; loosen --cluster-eps / --min-observations.")

    obb = pcd.get_oriented_bounding_box()
    t_filter = time.perf_counter() - t0
    ext = np.sort(np.asarray(obb.extent))[::-1] * 100.0
    print("\nObject: %d points, oriented bounding box %.1f x %.1f x %.1f cm"
          % (len(points), ext[0], ext[1], ext[2]))
    print("  Check that against the physical object with a ruler -- it is the single number "
          "that\n  tells you whether the segmentation kept a piece of the table or lost part of "
          "the object.")

    out_file = os.path.join(path, args.output)
    Ply(points, np.round(np.clip(colors, 0, 1) * 255.0)).write(out_file)
    print("Saved %s" % out_file)

    if args.preview_frames > 0:
        sel = [frames[int(round(i))] for i in
               np.linspace(0, len(frames) - 1, min(args.preview_frames, len(frames)))]
        for fid in sel:
            bgr = cv2.imread(os.path.join(path, "JPEGImages", "%06d.jpg" % fid))
            if bgr is None:
                continue
            T = Ts[fid // label_interval]
            cv2.imwrite(os.path.join(prev_dir, "object_%06d.jpg" % fid),
                        draw_result(bgr, points, T, K, obb))
        print("Reprojection previews: %s/object_*.jpg" % prev_dir)
        print("  The green points must sit on the object and the orange box must enclose it. "
              "If the\n  box is inflated, something else survived the vote -- raise "
              "--min-ratio or --erode.")

    if use_gpu:
        total = time.perf_counter() - t_start
        print("[timing] mask/depth load %.1fs | vote %.1fs | filter %.1fs | total %.1fs | "
              "%.3f s/frame over %d frames"
              % (timings["load"], timings["vote"], t_filter, total,
                 (timings["load"] + timings["vote"]) / max(1, len(frames)), len(frames)))
    # step 4 has no --gpu: its fast path is CPU vectorisation, always on
    print("Next: python 4_create_label_files.py %s" % (args.dataset or dataset))


def meta_stride(mask_dir):
    """The stride 3a used, so --interval stays a stride of the *sequence*."""
    meta_file = os.path.join(mask_dir, "seg_meta.json")
    if os.path.exists(meta_file):
        try:
            with open(meta_file) as f:
                return max(1, int(json.load(f).get("stride", 1)))
        except (ValueError, TypeError):
            pass
    return 1


def crop_scene(scene_file, path, mask_dir, frames, Ts, K, scale, width, height,
               label_interval, args, timings):
    """Keep the points of the merged scene that the frames agree are the object."""
    print("Loading %s ..." % scene_file)
    pcd = o3d.io.read_point_cloud(scene_file)
    P = np.asarray(pcd.points)
    C = np.asarray(pcd.colors)
    if not len(P):
        sys.exit("%s is empty." % scene_file)
    print("Scene: %d points; voting with %d frames ..." % (len(P), len(frames)))

    t0 = time.perf_counter()
    obj = np.zeros(len(P), np.int32)
    bg = np.zeros(len(P), np.int32)
    for fid in tqdm(frames, desc="vote"):
        mask = read_mask(mask_dir, fid, args.erode)
        depth = read_depth(path, fid)
        if mask is None or depth is None:
            continue
        u, v, z = project(P, Ts[fid // label_interval], K)
        vis = (z > 1e-6) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        idx = np.nonzero(vis)[0]
        if not len(idx):
            continue
        ui = u[idx].astype(np.int32)
        vi = v[idx].astype(np.int32)
        measured = depth[vi, ui] * scale
        on_surface = (measured > 0) & (np.abs(z[idx] - measured) <= args.depth_tol)
        inside = mask[vi, ui]
        obj[idx[on_surface & inside]] += 1
        bg[idx[on_surface & ~inside]] += 1
    timings["vote"] = time.perf_counter() - t0

    return keep_voted(P, C, obj, bg, args)


def keep_voted(P, C, obj, bg, args):
    """The min-observations / min-ratio keep rule, shared by both vote loops."""
    voted = obj + bg
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(voted > 0, obj / np.maximum(voted, 1), 0.0)
    keep = (obj >= args.min_observations) & (ratio >= args.min_ratio)
    print("Kept %d of %d scene points (%.2f %%): >= %d object votes and >= %.0f %% agreement"
          % (keep.sum(), len(P), 100.0 * keep.sum() / len(P),
             args.min_observations, 100.0 * args.min_ratio))
    if not keep.any():
        print("  (%d point(s) were never seen on the surface by any frame -- if that is most of "
              "the\n   scene, the poses and the depth disagree: raise --depth-tol)"
              % int((voted == 0).sum()))
    return P[keep], (C[keep] if len(C) else np.zeros((int(keep.sum()), 3)))


def fuse_depth(path, mask_dir, frames, Ts, K, scale, args, timings):
    """Build the object cloud from the masked depth alone (step 3 not needed)."""
    from utils.camera import convert_depth_frame_to_pointcloud
    step3 = load_step3()

    intr = {"fx": K[0, 0], "fy": K[1, 1], "ppx": K[0, 2], "ppy": K[1, 2],
            "depth_scale": scale}
    t0 = time.perf_counter()
    segments = []
    for fid in tqdm(frames, desc="unproject"):
        mask = read_mask(mask_dir, fid, args.erode)
        depth = cv2.imread(os.path.join(path, "depth", "%06d.png" % fid), cv2.IMREAD_UNCHANGED)
        bgr = cv2.imread(os.path.join(path, "JPEGImages", "%06d.jpg" % fid))
        if mask is None or depth is None or bgr is None or not mask.any():
            continue
        sel = mask & (depth > 0)
        if not sel.any():
            continue
        xyz = convert_depth_frame_to_pointcloud(depth.astype(np.float64), intr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        seg_pcd = o3d.geometry.PointCloud()
        seg_pcd.points = o3d.utility.Vector3dVector(xyz[sel])
        seg_pcd.colors = o3d.utility.Vector3dVector(rgb[sel].astype(np.float64) / 255.0)
        seg_pcd.transform(Ts[fid // args.label_interval])
        segments.append(seg_pcd)
    timings["load"] = time.perf_counter() - t0

    if not segments:
        sys.exit("No frame produced any masked depth. Dark or glossy surfaces return no depth "
                 "at all -- check depth/ before blaming the masks.")

    # One depth pixel at the object's range: the same rule step 3 uses, but
    # measured on the object rather than on the marker board, because that is
    # what is being merged here.
    centre = np.median(np.vstack([np.asarray(s.points) for s in segments]), axis=0)
    voxel_radius = float(np.linalg.norm(centre)) / float(K[0, 0])
    print("Merging %d masked segments (merge radius %.4f m = one depth pixel at %.3f m) ..."
          % (len(segments), voxel_radius, float(np.linalg.norm(centre))))
    t0 = time.perf_counter()
    points, colors, vote = step3.post_process(segments, voxel_radius, voxel_radius * 2.5)
    timings["vote"] = time.perf_counter() - t0
    keep = vote >= args.min_votes
    print("Kept %d of %d fused points (%.2f %%) with >= %d confirming observations"
          % (keep.sum(), len(points), 100.0 * keep.sum() / max(1, len(points)), args.min_votes))
    return points[keep], colors[keep]


# ---------------------------------------------------------------------------
# --gpu: the same two paths on CUDA
# ---------------------------------------------------------------------------

def auto_chunk(n_points, device, override):
    """Points per vote chunk: from free GPU memory unless --chunk is given."""
    import torch
    if override and override > 0:
        return min(n_points, override)
    free, _ = torch.cuda.mem_get_info(device)
    # ~128 B of transients per point per chunk (cam/z/u/v/flat/bools) with a
    # 2x safety margin already inside the constant.
    return max(1, min(n_points, int(free * 0.5 // 128)))


def gpu_vote(P, frames, loaded, Ts, K, scale, width, height, label_interval,
             depth_tol, device, chunk_override):
    """The CPU vote loop from crop_scene(), on torch CUDA.

    P       : (n,3) float64 world points (kept as float32 on the device)
    loaded  : {fid: (mask bool (h,w) or None, depth float32 (h,w) or None)}
    Returns (obj, bg) int32 numpy vote counters, one pair per point.
    """
    import torch
    dev = torch.device(device)
    n = len(P)
    pts = torch.from_numpy(np.ascontiguousarray(P, dtype=np.float32)).to(dev)
    obj = torch.zeros(n, dtype=torch.int32, device=dev)
    bg = torch.zeros(n, dtype=torch.int32, device=dev)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    ppx, ppy = float(K[0, 2]), float(K[1, 2])

    chunk = auto_chunk(n, dev, chunk_override)
    if chunk < n:
        print("Voting in chunks of %d points (%d chunks)"
              % (chunk, (n + chunk - 1) // chunk))

    with torch.no_grad():
        for fid in tqdm(frames, desc="vote (GPU)"):
            mask, depth = loaded[fid]
            if mask is None or depth is None:
                continue
            Tinv = np.linalg.inv(Ts[fid // label_interval])
            R = torch.from_numpy(np.ascontiguousarray(Tinv[:3, :3], np.float32)).to(dev)
            t = torch.from_numpy(np.ascontiguousarray(Tinv[:3, 3], np.float32)).to(dev)
            # depth * scale gathers the same float32 values the CPU multiplies
            # after its gather -- float32 elementwise multiply either way.
            depth_t = (torch.from_numpy(depth).to(dev) * scale).reshape(-1)
            mask_t = torch.from_numpy(np.ascontiguousarray(mask)).to(dev).reshape(-1)

            for s in range(0, n, chunk):
                cam = pts[s:s + chunk] @ R.T + t
                z = cam[:, 2]
                u = fx * cam[:, 0] / z + ppx
                v = fy * cam[:, 1] / z + ppy
                vis = (z > 1e-6) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
                # CPU: u[idx].astype(np.int32) -- truncation; u,v >= 0 where
                # vis, so truncation == floor. Clamp only guards the gather
                # for the non-vis lanes, which never vote.
                ui = torch.clamp(u, 0, width - 1).to(torch.int64)
                vi = torch.clamp(v, 0, height - 1).to(torch.int64)
                flat = vi * width + ui
                measured = depth_t[flat]
                on_surface = vis & (measured > 0) \
                    & ((z - measured).abs() <= depth_tol)
                inside = mask_t[flat]
                obj[s:s + chunk] += (on_surface & inside).to(torch.int32)
                bg[s:s + chunk] += (on_surface & ~inside).to(torch.int32)

    return obj.cpu().numpy(), bg.cpu().numpy()


def load_frames_threaded(path, mask_dir, frames, erode, io_threads):
    """{fid: (eroded mask, float32 depth)} -- the vote loop's inputs, once."""
    def worker(fid):
        return fid, (read_mask(mask_dir, fid, erode), read_depth(path, fid))
    with futures.ThreadPoolExecutor(io_threads) as pool:
        return dict(tqdm(pool.map(worker, frames), total=len(frames),
                         desc="load masks+depth (%d threads)" % io_threads))


def gpu_crop_scene(scene_file, path, mask_dir, frames, Ts, K, scale, width, height,
                   label_interval, args, timings):
    """crop_scene() with the vote loop on the GPU. Same prints, same rules."""
    print("Loading %s ..." % scene_file)
    pcd = o3d.io.read_point_cloud(scene_file)
    P = np.asarray(pcd.points)
    C = np.asarray(pcd.colors)
    if not len(P):
        sys.exit("%s is empty." % scene_file)
    print("Scene: %d points; voting with %d frames ..." % (len(P), len(frames)))

    t0 = time.perf_counter()
    loaded = load_frames_threaded(path, mask_dir, frames, args.erode, args.io_threads)
    timings["load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    obj, bg = gpu_vote(P, frames, loaded, Ts, K, scale, width, height,
                       label_interval, args.depth_tol, "cuda:%d" % args.gpu, args.chunk)
    timings["vote"] = time.perf_counter() - t0

    return keep_voted(P, C, obj, bg, args)


def gpu_fuse_depth(path, mask_dir, frames, Ts, K, scale, args, timings):
    """fuse_depth() with step 3's cv2 loader and its CUDA vote-merge."""
    step3 = load_step3()
    Ktuple = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))

    def worker(fid):
        mask = read_mask(mask_dir, fid, args.erode)
        depth = cv2.imread(os.path.join(path, "depth", "%06d.png" % fid),
                           cv2.IMREAD_UNCHANGED)
        if mask is None or depth is None or not mask.any():
            return None
        if not os.path.isfile(os.path.join(path, "JPEGImages", "%06d.jpg" % fid)):
            return None  # the CPU path skips unreadable frames
        sel = mask & (depth > 0)
        if not sel.any():
            return None
        # load_frame orders points by np.nonzero(depth) (row-major), exactly
        # the order of the mask values gathered below.
        pts, col = step3.load_frame(path, fid, Ktuple, scale)
        v, u = np.nonzero(depth)
        inmask = mask[v, u]
        T = Ts[fid // args.label_interval]
        world = pts[inmask] @ T[:3, :3].T.astype(np.float32) \
            + T[:3, 3].astype(np.float32)
        return world, col[inmask]

    t0 = time.perf_counter()
    with futures.ThreadPoolExecutor(args.io_threads) as pool:
        segments = [s for s in tqdm(pool.map(worker, frames), total=len(frames),
                                    desc="unproject (%d threads)" % args.io_threads)
                    if s is not None]
    timings["load"] = time.perf_counter() - t0

    if not segments:
        sys.exit("No frame produced any masked depth. Dark or glossy surfaces return no depth "
                 "at all -- check depth/ before blaming the masks.")

    # One depth pixel at the object's range: the same rule the CPU path uses.
    centre = np.median(np.vstack([s[0] for s in segments]).astype(np.float64), axis=0)
    voxel_radius = float(np.linalg.norm(centre)) / float(K[0, 0])
    print("Merging %d masked segments (merge radius %.4f m = one depth pixel at %.3f m) ..."
          % (len(segments), voxel_radius, float(np.linalg.norm(centre))))

    t0 = time.perf_counter()
    points, colors, vote = step3.gpu_post_process(segments, voxel_radius,
                                                  voxel_radius * 2.5,
                                                  "CUDA:%d" % args.gpu)
    timings["vote"] = time.perf_counter() - t0

    keep = vote >= args.min_votes
    print("Kept %d of %d fused points (%.2f %%) with >= %d confirming observations"
          % (keep.sum(), len(points), 100.0 * keep.sum() / max(1, len(points)),
             args.min_votes))
    # gpu_post_process hands colors back 0..255; the CPU path works in 0..1.
    return points[keep], colors[keep] / 255.0


if __name__ == "__main__":
    main()
