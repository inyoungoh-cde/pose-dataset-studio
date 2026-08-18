"""
3_register_scene.py
-------------------

Create registered scene pointcloud with ambient noise removal
The registered pointcloud includes the table top, markers, and some noise
This mesh needs to be processed in a mesh processing tool to remove the artifact

    python 3_register_scene.py mycup
    python 3_register_scene.py mycup --reconstruction-interval 8
    python 3_register_scene.py mycup --gpu 0        # same output, on CUDA:0

One script, two paths. Without `--gpu` the original CPU code runs exactly as it
always has and remains the reference. With `--gpu N` the merge runs on CUDA
device N: same inputs, same output (`registeredScene.ply`), same vote-merge
semantics -- only faster.

What is slow on the CPU path, and what the GPU path replaces it with:

1. **Frame loading** decodes each 16-bit depth PNG with `pypng`, a pure-Python
   decoder, one frame at a time. Replaced by `cv2.imread(IMREAD_UNCHANGED)`
   (the exact decoder steps 2/3b already use for the same files -- identical
   pixels) on a thread pool.
2. **The vote merge** (`post_process`) rebuilds a KD-tree over the *growing*
   merged cloud for every segment and queries it on the CPU -- quadratic-ish
   in total points, and by far the dominant cost. Replaced by Open3D's CUDA
   `NearestNeighborSearch.hybrid_search` (nearest neighbour within the vote
   radius): index rebuild over 5M points is ~30 ms on an A100, the query is
   exact, and the merge rules are bit-for-bit the CPU path's:

       no neighbour within 2.5r  -> genuinely new point, no vote
       neighbour at d > r        -> new point; votes its neighbour if d < 2.5r
       neighbour at d <= r       -> duplicate, dropped; votes its neighbour

   (r = the merge radius; votes use numpy fancy-indexing semantics, i.e. one
   vote per *unique* neighbour per segment, exactly as before.)

Everything around the merge -- the auto merge radius, the sparse-shell warning,
`--min-votes`, the marker-board crop, the PLY writer -- is shared by both
paths. If `--gpu` is given and no CUDA device is available the script says so
and exits; it never falls back silently.
"""
import concurrent.futures as futures
import png
import random
import cv2.aruco as aruco
import open3d as o3d
import numpy as np
import cv2
import os
import glob
from utils.ply import Ply
from utils.camera import *
from utils.cli import build_parser, resolve_dataset, normalize_root
from tqdm import tqdm, trange
from pykdtree.kdtree import KDTree
import time
import sys
from config.registrationParameters import *
import json

# open3d >= 0.10 moved registration under o3d.pipelines
geometry = o3d.geometry
utility = o3d.utility
try:
    registration = o3d.registration
except AttributeError:
    registration = o3d.pipelines.registration

# Set up parameters for registration
# voxel sizes use to down sample raw pointcloud for fast ICP
voxel_size = VOXEL_SIZE
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5

# Set up parameters for post-processing
# Voxel size for the complete mesh
voxel_Radius = VOXEL_R

# Point considered an outlier if more than inlier_Radius away from other points
inlier_Radius = voxel_Radius * 2.5

# search for up to N frames for registration, odometry only N=1, all frames N = np.inf
N_Neighbours = K_NEIGHBORS


from utils.markers import detect_markers, marker_region_world  # noqa: E402


def post_process(originals, voxel_Radius, inlier_Radius):
     """
    Merge segments so that new points will not be add to the merged
    model if within voxel_Radius to the existing points, and keep a vote
    for if the point is issolated outside the radius of inlier_Radius at
    the timeof the merge

    Parameters
    ----------
    originals : List of open3d.Pointcloud classe
      6D pontcloud of the segments transformed into the world frame
    voxel_Radius : float
      Reject duplicate point if the new point lies within the voxel radius
      of the existing point
    inlier_Radius : float
      Point considered an outlier if more than inlier_Radius away from any
      other points

    Returns
    ----------
    points : (n,3) float
      The (x,y,z) of the processed and filtered pointcloud
    colors : (n,3) float
      The (r,g,b) color information corresponding to the points
    vote : (n, ) int
      The number of vote (seen duplicate points within the voxel_radius) each
      processed point has reveived
    """

     for point_id in trange(len(originals)):

          if point_id == 0:
               vote = np.zeros(len(originals[point_id].points))
               points = np.array(originals[point_id].points,dtype = np.float64)
               colors = np.array(originals[point_id].colors,dtype = np.float64)

          else:

               points_temp = np.array(originals[point_id].points,dtype = np.float64)
               colors_temp = np.array(originals[point_id].colors,dtype = np.float64)

               dist , index = nearest_neighbour(points_temp, points)
               new_points = np.where(dist > voxel_Radius)
               points_temp = points_temp[new_points]
               colors_temp = colors_temp[new_points]
               inliers = np.where(dist < inlier_Radius)
               vote[(index[inliers],)] += 1
               vote = np.concatenate([vote, np.zeros(len(points_temp))])
               points = np.concatenate([points, points_temp])
               colors = np.concatenate([colors, colors_temp])

     return (points,colors,vote)


def load_pcds(path, downsample = True, interval = 1):

    """
    load pointcloud by path and down samle (if True) based on voxel_size

    """


    global voxel_size, camera_intrinsics
    pcds= []

    for Filename in trange(int(len(glob.glob1(path+"JPEGImages","*.jpg"))/interval)):
        img_file = path + 'JPEGImages/' + str('%06d' % (Filename*interval)) + '.jpg'

        cad = cv2.imread(img_file)
        cad = cv2.cvtColor(cad, cv2.COLOR_BGR2RGB)
        depth_file = path + 'depth/' + str('%06d' % (Filename*interval)) + '.png'
        reader = png.Reader(depth_file)
        pngdata = reader.read()
        depth = np.array(tuple(map(np.uint16, pngdata[2])))
        mask = depth.copy()
        depth = convert_depth_frame_to_pointcloud(depth, camera_intrinsics)


        source = geometry.PointCloud()
        source.points = utility.Vector3dVector(depth[mask>0])
        source.colors = utility.Vector3dVector(cad[mask>0])

        if downsample == True:
            pcd_down = source.voxel_down_sample(voxel_size = voxel_size)
            pcd_down.estimate_normals(geometry.KDTreeSearchParamHybrid(radius = 0.002 * 2, max_nn = 30))
            pcds.append(pcd_down)
        else:
            pcds.append(source)
    return pcds



def nearest_neighbour(a, b):
    """
    find the nearest neighbours of a in b using KDTree
    Parameters
    ----------
    a : (n, ) numpy.ndarray
    b : (n, ) numpy.ndarray

    Returns
    ----------
    dist : n float
      Euclidian distance of the closest neighbour in b to a
    index : n float
      The index of the closest neighbour in b to a in terms of Euclidian distance
    """
    tree = KDTree(b)
    dist, index = tree.query(a)
    return (dist, index)


def load_frame(path, fid, K, scale):
    """One frame -> (points (n,3) float32 camera-space, colors (n,3) uint8).

    The GPU path's frame loader (also used by 3b's `--source depth`): the same
    pixels `load_pcds` reads, decoded with cv2 instead of pypng.
    """
    bgr = cv2.imread(path + "JPEGImages/%06d.jpg" % fid)
    depth = cv2.imread(path + "depth/%06d.png" % fid, cv2.IMREAD_UNCHANGED)
    if bgr is None or depth is None:
        sys.exit("Could not read frame %06d under %s" % (fid, path))
    fx, fy, ppx, ppy = K
    v, u = np.nonzero(depth)
    z = depth[v, u].astype(np.float32) * scale
    x = (u.astype(np.float32) - ppx) / fx * z
    y = (v.astype(np.float32) - ppy) / fy * z
    pts = np.stack([x, y, z], axis=1)
    colors = bgr[v, u][:, ::-1]  # BGR -> RGB, kept 0..255 like the CPU path
    return pts, colors


def gpu_post_process(segments, voxel_Radius, inlier_Radius, device):
    """post_process(), with the nearest-neighbour step on CUDA.

    segments: iterable of (points float32 (n,3) world-space, colors uint8)
    device  : an Open3D device string, e.g. "CUDA:0"
    Returns (points float64 (n,3), colors float64 0..255, vote int64) --
    the same dtypes the CPU version hands onward.

    Vote rules are the CPU version's, bit for bit. Three implementation choices
    make this fast without touching the semantics:

    * the merged cloud lives in ONE preallocated GPU buffer -- the first
      version concatenated a fresh copy of everything per segment (~22 GB of
      device copies over the box sequence);
    * distances/votes stay on the GPU (torch via zero-copy dlpack views of
      the same buffer) -- no per-segment round trip of full-size arrays;
    * the spatial index over the merged cloud is rebuilt only when the
      un-indexed tail has grown past max(1M, 25%); queries take the min over
      (main index, small tail index), and the nearest neighbour over a
      partition's union IS the min over its parts -- exact, not approximate.
    """
    import torch
    import open3d.core as o3c
    from open3d.core import nns
    from torch.utils import dlpack as tdl

    tdev = torch.device(device.lower())
    r = inlier_Radius
    INF = float("inf")

    def as_o3c(t):
        return o3c.Tensor.from_dlpack(tdl.to_dlpack(t.contiguous()))

    def as_torch(t):
        return tdl.from_dlpack(t.to_dlpack())

    def query(index, seg_t):
        ind, d2, cnt = index.hybrid_search(as_o3c(seg_t), r, 1)
        found = as_torch(cnt).view(-1) > 0
        dist = torch.where(found, as_torch(d2).view(-1).clamp(min=0).sqrt(),
                           torch.full((len(seg_t),), INF, device=tdev))
        idx = torch.where(found, as_torch(ind).view(-1),
                          torch.zeros(len(seg_t), dtype=torch.int64, device=tdev))
        return dist, idx

    capacity = sum(len(p) for p, _ in segments)
    try:
        pts = torch.empty((capacity, 3), dtype=torch.float32, device=tdev)
        vote = torch.zeros(capacity, dtype=torch.int32, device=tdev)
    except torch.cuda.OutOfMemoryError:
        sys.exit("Not enough GPU memory for %d points (%.1f GB needed) -- raise "
                 "--reconstruction-interval, or drop --gpu to run on the CPU."
                 % (capacity, capacity * 16 / 1e9))
    colors = np.empty((capacity, 3), np.uint8)  # only needed on the CPU, at the end

    n = 0          # merged points so far
    indexed = 0    # prefix of pts covered by main_index
    main_index = None

    for pts_np, col_np in tqdm(segments, desc="vote-merge (GPU)"):
        seg = torch.from_numpy(np.ascontiguousarray(pts_np)).to(tdev)
        if n == 0:
            pts[:len(seg)] = seg
            colors[:len(seg)] = col_np
            n = len(seg)
            continue

        dist = torch.full((len(seg),), INF, device=tdev)
        idx = torch.zeros(len(seg), dtype=torch.int64, device=tdev)
        if main_index is not None:
            dist, idx = query(main_index, seg)
        if n > indexed:  # points merged since the last main-index rebuild
            t_index = nns.NearestNeighborSearch(as_o3c(pts[indexed:n]))
            t_index.hybrid_index(r)
            d_t, i_t = query(t_index, seg)
            better = d_t < dist
            dist = torch.where(better, d_t, dist)
            idx = torch.where(better, i_t + indexed, idx)

        # the CPU path's exact rules (see module docstring)
        new = dist > voxel_Radius                      # inf (no neighbour) is new too
        inliers = dist < inlier_Radius
        vote[torch.unique(idx[inliers])] += 1          # fancy-indexing semantics

        k = int(new.sum())
        if k:
            pts[n:n + k] = seg[new]
            colors[n:n + k] = col_np[new.cpu().numpy()]
            n += k
        if n - indexed > max(1_000_000, indexed // 4):
            main_index = nns.NearestNeighborSearch(as_o3c(pts[:n]))
            main_index.hybrid_index(r)
            indexed = n

    return (pts[:n].cpu().numpy().astype(np.float64),
            colors[:n].astype(np.float64),
            vote[:n].cpu().numpy().astype(np.int64))


def require_cuda(gpu, dataset):
    """--gpu N: verify Open3D sees CUDA device N, or exit with a clear message."""
    import open3d.core as o3c
    hint = ("Drop --gpu to run the CPU path:\n"
            "  python 3_register_scene.py %s" % dataset)
    if not o3c.cuda.is_available():
        sys.exit("--gpu %d was given, but Open3D sees no CUDA device on this machine.\n%s"
                 % (gpu, hint))
    count = o3c.cuda.device_count()
    if not 0 <= gpu < count:
        sys.exit("--gpu %d: no such CUDA device -- this machine has %d (0..%d).\n%s"
                 % (gpu, count, count - 1, hint))


def merge_cpu(path, Ts, R, L, voxel_Radius, inlier_Radius):
    """Load every Nth frame with pypng/open3d and vote-merge it on the CPU."""
    print("Merge segments")
    originals = load_pcds(path, downsample = False, interval = R)
    for point_id in range(len(originals)):
         originals[point_id].transform(Ts[int(R/L)*point_id])
    print("Apply post processing")
    return post_process(originals, voxel_Radius, inlier_Radius)


def merge_gpu(path, Ts, R, L, K, scale, voxel_Radius, inlier_Radius, device,
              io_threads, timings):
    """The same merge with threaded cv2 frame loading and the CUDA vote-merge."""
    n_frames = len([f for f in os.listdir(path + "JPEGImages") if f.endswith(".jpg")])
    # exactly the CPU path's segment list: range(n_frames // R) * R
    frame_ids = [k * R for k in range(n_frames // R) if (R // L) * k < len(Ts)]
    print("Merging %d segments (every %dth frame) on %s" % (len(frame_ids), R, device))

    # ---- load + transform to world (threaded IO, vectorised maths) --------
    t0 = time.perf_counter()
    def worker(fid):
        pts, col = load_frame(path, fid, K, scale)
        T = Ts[(R // L) * (fid // R)]
        return (pts @ T[:3, :3].T.astype(np.float32)) + T[:3, 3].astype(np.float32), col

    with futures.ThreadPoolExecutor(io_threads) as pool:
        segments = list(tqdm(pool.map(worker, frame_ids), total=len(frame_ids),
                             desc="load+pose (%d threads)" % io_threads))
    timings["load"] = time.perf_counter() - t0
    timings["n_in"] = sum(len(s[0]) for s in segments)
    timings["n_segments"] = len(frame_ids)

    # ---- vote merge on the GPU --------------------------------------------
    t0 = time.perf_counter()
    out = gpu_post_process(segments, voxel_Radius, inlier_Radius, device)
    timings["merge"] = time.perf_counter() - t0
    return out


def main():
    global camera_intrinsics

    parser = build_parser("Merge all frames into registeredScene.ply for manual segmentation")
    parser.add_argument("--reconstruction-interval", type=int, default=RECONSTRUCTION_INTERVAL,
                        help="use every Nth frame for the scene reconstruction")
    parser.add_argument("--label-interval", type=int, default=LABEL_INTERVAL,
                        help="frame interval used in 2_compute_gt_poses.py")
    parser.add_argument("--voxel-r", type=float, default=None,
                        help="merge radius (m): new points within this radius of existing "
                             "points are dropped, and a later segment's point within 2.5x it "
                             "counts as a confirming observation (a 'vote'). Default: derived "
                             "from the data as one depth pixel at the marker board's range, so "
                             "the vote tolerance matches the sensor's actual sampling. The "
                             "paper's fixed value was %g" % VOXEL_R)
    parser.add_argument("--out-dir", default=None,
                        help="where to write registeredScene.ply (default: the sequence folder, "
                             "which is where step 4 expects it)")
    parser.add_argument("--crop-margin", type=float, default=0.15,
                        help="keep only points within (marker-board radius + this many metres) "
                             "of the board centre. Removes the re-posed background, which in a "
                             "turntable capture sweeps into rings and a bowl-shaped shell "
                             "around the object")
    parser.add_argument("--no-crop", action="store_true",
                        help="keep the whole re-posed scene instead of cropping to the marker "
                             "board (the original ODT behaviour)")
    parser.add_argument("--min-votes", type=int, default=2,
                        help="keep a point only if this many later segments confirmed it. "
                             "Higher = cleaner but sparser; lower = denser but noisier. "
                             "Surfaces with poor depth (dark, glossy or steeply slanted -- "
                             "typically the object itself, not the marker board) are the first "
                             "to be thinned out, so lower this if the object comes out sparse")
    parser.add_argument("--gpu", nargs="?", const=0, default=None, type=int, metavar="N",
                        help="run the merge on CUDA device N (a bare --gpu means device 0). "
                             "Omit it and the CPU path runs, unchanged")
    parser.add_argument("--io-threads", type=int, default=8,
                        help="parallel frame decoders (--gpu only)")
    args = parser.parse_args()
    dataset, path = resolve_dataset(
        args, require=("intrinsics.json", "transforms.npy", "JPEGImages", "depth"))

    use_gpu = args.gpu is not None
    device = "CUDA:%d" % args.gpu if use_gpu else None
    if use_gpu:
        require_cuda(args.gpu, args.dataset or dataset)

    R, L = args.reconstruction_interval, args.label_interval
    # poses exist only every --label-interval frames (Ts[(R/L)*i])
    if R < L or R % L != 0:
        sys.exit("--reconstruction-interval (%d) must be a multiple of --label-interval (%d) "
                 "and not smaller than it." % (R, L))
    with open(path+'intrinsics.json', 'r') as f:
         camera_intrinsics = json.load(f)
    fx = float(camera_intrinsics['fx'])

    # The marker board locates both the auto merge radius and the crop region.
    board_centre, board_radius = marker_region_world(path, camera_intrinsics)

    voxel_Radius = args.voxel_r
    if voxel_Radius is None:
        if board_centre is None:
            voxel_Radius = VOXEL_R
            print("Merge radius: %.4f m (markers not found in frame 0, using the "
                  "config default)" % voxel_Radius)
        else:
            # one depth pixel at the board's range
            board_range = float(np.linalg.norm(board_centre))
            voxel_Radius = board_range / fx
            print("Merge radius: %.4f m (auto = one depth pixel at the board's range of "
                  "%.3f m; vote tolerance %.4f m)"
                  % (voxel_Radius, board_range, voxel_Radius * 2.5))
    else:
        print("Merge radius: %.4f m (set explicitly)" % voxel_Radius)
    inlier_Radius = voxel_Radius * 2.5

    if board_centre is not None:
        pitch = float(np.linalg.norm(board_centre)) / fx
        if inlier_Radius < pitch:
            print("WARNING: the vote tolerance (%.2f mm) is finer than the depth sampling pitch "
                  "at\n  this range (%.2f mm), so repeated observations rarely confirm each other "
                  "and\n  --min-votes deletes them. Measured on a black DSLR at 0.64 m: 97.5 %% of "
                  "the\n  object's points deleted versus 84 %% of the marker board's, leaving the "
                  "object a\n  sparse shell. Drop --voxel-r to let it be chosen automatically, or "
                  "pass >= %.4f."
                  % (inlier_Radius * 1000, pitch * 1000, pitch / 2.5))

    out_path = path if args.out_dir is None else normalize_root(args.out_dir)
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    Ts = np.load(path + 'transforms.npy')
    timings = {}
    if use_gpu:
        K = (fx, float(camera_intrinsics["fy"]),
             float(camera_intrinsics["ppx"]), float(camera_intrinsics["ppy"]))
        points, colors, vote = merge_gpu(path, Ts, R, L, K,
                                         float(camera_intrinsics["depth_scale"]),
                                         voxel_Radius, inlier_Radius, device,
                                         args.io_threads, timings)
    else:
        points, colors, vote = merge_cpu(path, Ts, R, L, voxel_Radius, inlier_Radius)

    keep = vote >= args.min_votes
    print("Keeping %d of %d points (%.2f %%) with >= %d confirming observations"
          % (keep.sum(), len(points), 100.0 * keep.sum() / max(1, len(points)), args.min_votes))

    if not args.no_crop:
        centre, radius = board_centre, board_radius
        if centre is None:
            print("Could not locate the marker board in frame 0 -- skipping the crop.")
        else:
            r = radius + args.crop_margin
            inside = np.linalg.norm(points - centre, axis=1) <= r
            print("Cropping to the marker board: centre (%.3f, %.3f, %.3f) m, radius %.3f m "
                  "-> drops %d of the %d surviving points"
                  % (centre[0], centre[1], centre[2], r,
                     int((keep & ~inside).sum()), int(keep.sum())))
            keep &= inside

    t0 = time.perf_counter()
    ply = Ply(points[keep], colors[keep])
    meshfile = out_path + 'registeredScene.ply'
    ply.write(meshfile)
    t_write = time.perf_counter() - t0
    print("Saved %d points to %s" % (int(keep.sum()), meshfile))

    if use_gpu:
        n_seg = timings["n_segments"]
        per = (timings["load"] + timings["merge"]) / max(1, n_seg)
        print("[timing] load+pose %.1fs | vote-merge %.1fs | write %.1fs | "
              "%.2f s/segment over %d segments (%d -> %d points)"
              % (timings["load"], timings["merge"], t_write, per, n_seg,
                 timings["n_in"], int(keep.sum())))
        print("Next: python 3a_segment_object_masks.py %s --prompt \"...\" --device cuda:%d  "
              "(or segment by hand in CloudCompare)" % (args.dataset or dataset, args.gpu))
    else:
        print("Next: open it in CloudCompare, delete everything except the object, subsample to "
              "~2 mm\n  spacing (Edit > Subsample > Space) and save as object.ply next to the "
              "sequence.")


if __name__ == "__main__":
    main()
