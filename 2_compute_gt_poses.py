"""
2_compute_gt_poses.py
---------------------

Main Function for registering (aligning) colored point clouds with ICP/aruco marker
matching as well as pose graph optimizating, output transforms.npy in each directory

    python 2_compute_gt_poses.py mycup
    python 2_compute_gt_poses.py mycup --k-neighbors 10 --icp-method point-to-plane
    python 2_compute_gt_poses.py mycup --gpu      # same outputs, on CUDA device 0
    python 2_compute_gt_poses.py mycup --gpu 1    # ... on CUDA device 1

Without `--gpu` the CPU path runs; it is the reference implementation and the
default. `--gpu [N]` selects the GPU-accelerated path: same inputs, same
outputs (`transforms.npy` + `log.txt`), same pose-graph construction -- only
faster. There is no silent fallback: if Open3D/PyTorch report no usable CUDA
device, the script says so and stops instead of quietly running on the CPU.

Where the CPU path spends its time (930-frame box sequence, measured):
almost none of it is the actual pose estimation. The ArUco corners give the
transform nearly for free; the per-frame second goes into

1. **Point-cloud loading** -- every edge needs both endpoint clouds
   (decode JPEG+PNG, back-project 900k pixels, voxel-downsample), and the
   loop-closure target is *loaded and thrown away* every time because the
   cache only keeps the odometry pair.
2. **The information matrix** -- one nearest-neighbour pass over the two
   downsampled clouds per edge, on the CPU (this weights each edge in the
   pose-graph optimization).

What `--gpu` changes -- nothing else:

* marker features for all frames are precomputed on a thread pool
  (the detection maths is the very same `get_marker_features` both paths call);
* clouds are decoded on threads, back-projected vectorised, downsampled with
  Open3D's CUDA `voxel_down_sample`, and kept in an LRU cache big enough that
  a loop-closure target is loaded once instead of re-decoded for every edge
  (sized by `--cache-frames` / `--io-threads`);
* the robust marker RANSAC draws the same 200 hypotheses from the same seeded
  generator in the same order, but fits them as one batch
  (`_match_ransac_robust_batched`);
* the information matrix runs on the GPU
  (`o3d.t.pipelines.registration.get_information_matrix` -- verified to match
  the legacy CPU function to ~1e-7 relative on identical inputs);
* the rare full-frame ICP fallback (markers failed) converts the two cached
  clouds to legacy CPU point clouds and calls the same
  `utils.registration.icp`, so that path is bit-identical to the CPU path.

Pose graph assembly and global optimization are the original Open3D CPU calls
on both paths. Differences between the two paths' output come only from
voxel-downsample implementation details (CUDA vs legacy grid averaging)
feeding the edge weights; measured on `box`: see doc/DETAILS.md §9.
"""
import collections
import concurrent.futures as futures
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
from utils.registration import icp, match_ransac, match_ransac_robust
from tqdm import trange
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

# cv2.aruco API changed in OpenCV 4.7
try:
    _ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_250)
    _ARUCO_PARAMS = aruco.DetectorParameters_create()
    def detect_markers(gray):
        return aruco.detectMarkers(gray, _ARUCO_DICT, parameters=_ARUCO_PARAMS)
except AttributeError:
    _ARUCO_DETECTOR = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_6X6_250),
                                          aruco.DetectorParameters())
    def detect_markers(gray):
        return _ARUCO_DETECTOR.detectMarkers(gray)

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

# Defaults for the CLI flags of the same names (see argparse help below).
OUT_PATH = None
MARKER_RANSAC = "odometry"
CORNER_DEPTH_TOL = 0.05
CAPTURE_MODE = "auto"
STATIC_BG_TOL = 0.005
MAX_LOOP_CLOSURES = 1

# per-frame marker cache: each frame is read and processed once
_marker_cache = {}

# Robust marker estimator used by marker_registration(). The CPU path keeps the
# reference implementation from utils.registration; main() swaps in the batched
# equivalent below for the --gpu path (same draws, same acceptance rules).
_ROBUST_ESTIMATOR = match_ransac_robust

def get_marker_features(path, ID):
     """Detect ArUco markers and lift their corners to camera-space 3D."""
     if ID in _marker_cache:
          return _marker_cache[ID]

     img_file = path + 'JPEGImages/' + str('%06d' % (ID*LABEL_INTERVAL)) + '.jpg'
     cad = cv2.imread(img_file)
     depth_file = path + 'depth/' + str('%06d' % (ID*LABEL_INTERVAL)) + '.png'
     depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)

     gray = cv2.cvtColor(cad, cv2.COLOR_RGB2GRAY)
     corners, _ids, rejectedImgPoints = detect_markers(gray)

     if _ids is None:
          _marker_cache[ID] = (None, None)
          return _marker_cache[ID]

     fx = float(camera_intrinsics['fx'])
     fy = float(camera_intrinsics['fy'])
     ppx = float(camera_intrinsics['ppx'])
     ppy = float(camera_intrinsics['ppy'])
     scale = float(camera_intrinsics['depth_scale'])

     ids = []
     corners_3d = []
     for i in range(len(_ids)):
          ids.append(_ids[i][0])
          pts = np.zeros((4, 3))
          uv = [(int(c[0]), int(c[1])) for c in corners[i][0]]
          zs = np.array([depth[v, u] * scale for u, v in uv])
          valid = zs > 0

          # drop corners whose depth is background bleed at the marker border
          if CORNER_DEPTH_TOL > 0 and np.count_nonzero(valid) >= 2:
               z_med = np.median(zs[valid])
               valid &= np.abs(zs - z_med) <= CORNER_DEPTH_TOL

          for count in range(4):
               if not valid[count]:
                    continue          # left as (0,0,0): treated as missing depth
               u, v = uv[count]
               z = zs[count]
               pts[count] = ((u - ppx) / fx * z, (v - ppy) / fy * z, z)
          corners_3d.append(pts)

     _marker_cache[ID] = (ids, corners_3d)
     return _marker_cache[ID]


def marker_registration(source, target, robust=False):
     """Relative transform from matched ArUco corners.

     Returns (transform, info); transform is None when the marker path cannot
     be trusted, in which case the caller falls back to ICP.
     """
     ids_src, c3d_src = source
     ids_des, c3d_des = target
     if ids_src is None or ids_des is None:
          return None, {"n": 0, "rmse": float("nan"), "n_inliers": 0,
                        "reason": "no_markers"}

     common = [x for x in ids_src if x in ids_des]

     if len(common) < 2:
          # too few marker matches, use icp instead
          return None, {"n": 0, "rmse": float("nan"), "n_inliers": 0,
                        "reason": "too_few_common_markers"}

     src_good = []
     dst_good = []
     for i,id in enumerate(ids_des):
          if id in ids_src:
               j = ids_src.index(id)
               for count in range(4):
                    feature_3D_src = c3d_src[j][count]
                    feature_3D_des = c3d_des[i][count]
                    if feature_3D_src[2]!=0 and feature_3D_des[2]!=0:
                         src_good.append(feature_3D_src)
                         dst_good.append(feature_3D_des)

     # get rigid transforms between 2 set of feature points
     try:
          if robust:
               return _ROBUST_ESTIMATOR(np.asarray(src_good), np.asarray(dst_good))
          return match_ransac(np.asarray(src_good), np.asarray(dst_good))
     except Exception as e:
          return None, {"n": len(src_good), "rmse": float("nan"), "n_inliers": 0,
                        "reason": "exception:%s" % type(e).__name__}




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

def _marker_region_mask_2d(gray, dilate_px=60):
     """Image mask covering the markers plus a margin (i.e. the moving body)."""
     corners, ids, _ = detect_markers(gray)
     mask = np.zeros(gray.shape, np.uint8)
     if ids is not None and len(ids):
          pts = np.concatenate([c[0] for c in corners]).astype(np.int32)
          cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
          k = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
          mask = cv2.dilate(mask, k)
     return mask.astype(bool)


def detect_capture_mode(path, n_frames):
     """Decide whether the camera moved or the object moved, from how much the
     background depth (away from the markers) changes between distant frames.
     Returns (mode, worst_median_metres)."""
     scale = float(camera_intrinsics['depth_scale'])
     q = max(1, n_frames // 4)
     pairs = [(0, q), (q, 2 * q), (2 * q, 3 * q), (0, 2 * q), (0, n_frames - 1)]
     worst, used = 0.0, 0
     for a, b in pairs:
          if b >= n_frames or a >= b:
               continue
          ia = cv2.imread(path + 'JPEGImages/' + str('%06d' % (a * LABEL_INTERVAL)) + '.jpg')
          ib = cv2.imread(path + 'JPEGImages/' + str('%06d' % (b * LABEL_INTERVAL)) + '.jpg')
          da = cv2.imread(path + 'depth/' + str('%06d' % (a * LABEL_INTERVAL)) + '.png',
                          cv2.IMREAD_UNCHANGED)
          db = cv2.imread(path + 'depth/' + str('%06d' % (b * LABEL_INTERVAL)) + '.png',
                          cv2.IMREAD_UNCHANGED)
          if ia is None or ib is None or da is None or db is None:
               continue
          excl = (_marker_region_mask_2d(cv2.cvtColor(ia, cv2.COLOR_BGR2GRAY)) |
                  _marker_region_mask_2d(cv2.cvtColor(ib, cv2.COLOR_BGR2GRAY)))
          bg = (~excl) & (da > 0) & (db > 0)
          if np.count_nonzero(bg) < 5000:
               continue
          d = np.abs(da[bg].astype(np.float64) - db[bg].astype(np.float64)) * scale
          worst = max(worst, float(np.median(d)))
          used += 1
     if used == 0:
          # no usable background: assume the conventional moving-camera capture
          return "moving-camera", float("nan")
     return ("static-camera" if worst < STATIC_BG_TOL else "moving-camera"), worst


def crop_to_marker_region(pcd, corners_3d, margin=0.15):
     """Keep only points near the markers, so a static-camera ICP fallback
     cannot lock onto the unmoving background (see doc/DETAILS.md §4)."""
     pts = np.array([c for m in corners_3d for c in m if c[2] > 0])
     if len(pts) < 4:
          return pcd
     centre = pts.mean(axis=0)
     radius = np.linalg.norm(pts - centre, axis=1).max() + margin
     P = np.asarray(pcd.points)
     keep = np.flatnonzero(np.linalg.norm(P - centre, axis=1) <= radius)
     if len(keep) < 100:
          return pcd
     return pcd.select_by_index(keep)


def _get_pcd(cache, path, frame_id):
     """Point cloud for a frame, loaded on demand (normals are added lazily)."""
     if frame_id not in cache:
          cache[frame_id] = load_pcd(path, frame_id, downsample=True)
     return cache[frame_id]


def _select_targets(source_id, n_pcds, stride):
     """Frames to register `source_id` against: its successor (odometry) plus
     at most MAX_LOOP_CLOSURES strided loop-closure candidates."""
     candidates = list(range(source_id + 1, n_pcds, stride))
     loop = [t for t in candidates if t != source_id + 1]
     if MAX_LOOP_CLOSURES >= 0 and len(loop) > MAX_LOOP_CLOSURES:
          off = source_id % len(loop)
          loop = [loop[(off + k) % len(loop)] for k in range(MAX_LOOP_CLOSURES)]
     head = [source_id + 1] if source_id + 1 < n_pcds else []
     return head + sorted(loop)


def full_registration(path,max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):

     global N_Neighbours, LABEL_INTERVAL, n_pcds
     pose_graph = registration.PoseGraph()
     odometry = np.identity(4)
     pose_graph.nodes.append(registration.PoseGraphNode(odometry))

     pcds = {}
     log = open((OUT_PATH or path)+"log.txt", 'w')
     log.write("log starts: frames=%d voxel_size=%g label_interval=%d "
               "k_neighbors=%d icp_method=%s marker_ransac=%s corner_depth_tol=%g "
               "max_loop_closures=%d capture_mode=%s\n"
               % (n_pcds, voxel_size, LABEL_INTERVAL, N_Neighbours, ICP_METHOD,
                  MARKER_RANSAC, CORNER_DEPTH_TOL, MAX_LOOP_CLOSURES, CAPTURE_MODE))
     count = 0
     n_icp_fallback = 0
     stride = max(1, int(n_pcds / N_Neighbours))
     for source_id in trange(n_pcds):
          # keep only the clouds still needed: this frame and its successor
          for fid in [k for k in pcds if k not in (source_id, source_id + 1)]:
               del pcds[fid]

          for target_id in _select_targets(source_id, n_pcds, stride):

               is_odometry = (target_id == source_id + 1)
               use_robust = (MARKER_RANSAC == "all" or
                             (MARKER_RANSAC == "odometry" and is_odometry))

               # derive pairwise registration through feature matching
               res, minfo = marker_registration(get_marker_features(path, source_id),
                                                get_marker_features(path, target_id),
                                                robust=use_robust)

               if res is None and not is_odometry:
                    # ignore such connections
                    continue

               diag = "n=%d rmse=%.5f inliers=%d reason=%s" % (
                    minfo["n"], minfo["rmse"], minfo["n_inliers"], minfo["reason"])

               pcd_src = _get_pcd(pcds, path, source_id)
               pcd_dst = _get_pcd(pcds, path, target_id)
               if res is None:
                    # if marker_registration fails, perform pointcloud matching
                    icp_src, icp_dst = pcd_src, pcd_dst
                    if CAPTURE_MODE == "static-camera":
                         # only the marker board / object moved: hide the static
                         # room from ICP or it will report ~no motion
                         icp_src = crop_to_marker_region(
                              icp_src, get_marker_features(path, source_id)[1] or [])
                         icp_dst = crop_to_marker_region(
                              icp_dst, get_marker_features(path, target_id)[1] or [])
                    # point-to-plane needs normals; only this branch uses them
                    for pc in (icp_src, icp_dst):
                         if not pc.has_normals():
                              pc.estimate_normals(geometry.KDTreeSearchParamHybrid(
                                   radius=0.002 * 2, max_nn=30))
                    transformation_icp, information_icp = icp(
                         icp_src, icp_dst, voxel_size, max_correspondence_distance_coarse,
                         max_correspondence_distance_fine, method = ICP_METHOD)
                    n_icp_fallback += 1
                    log.write(str(count)+"-"+str(source_id)+"/"+str(n_pcds)+","+str(target_id)+"/"+str(n_pcds)+", None_icp, "+diag+"\n")

               else:
                    transformation_icp = res
                    information_icp = registration.get_information_matrix_from_point_clouds(
                         pcd_src, pcd_dst, max_correspondence_distance_fine,
                         transformation_icp)
                    log.write(str(count)+"-"+str(source_id)+"/"+str(n_pcds)+","+str(target_id)+"/"+str(n_pcds)+", res_marker, "+diag+"\n")
               log.flush()
               if target_id == source_id + 1:
                    # odometry
                    odometry = np.dot(transformation_icp, odometry)
                    pose_graph.nodes.append(registration.PoseGraphNode(np.linalg.inv(odometry)))
                    pose_graph.edges.append(registration.PoseGraphEdge(source_id, target_id,
                                                          transformation_icp, information_icp, uncertain = False))
               else:
                    # loop closure
                    pose_graph.edges.append(registration.PoseGraphEdge(source_id, target_id,
                                                          transformation_icp, information_icp, uncertain = True))
               count+=1
     log.write("edges=%d marker=%d icp_fallback=%d\n"
               % (count, count - n_icp_fallback, n_icp_fallback))
     log.close()

     if n_icp_fallback:
          print("\nWARNING: %d of %d edges fell back to full-frame ICP "
                "(marker registration failed; see 'None_icp' in log.txt)."
                % (n_icp_fallback, count))
          print("  Full-frame ICP registers EVERYTHING in view. It only measures the "
                "camera's\n  motion if the camera is what moved. If instead the camera was "
                "static and the\n  object/marker board moved (e.g. on a turntable), ICP locks "
                "onto the static\n  background and returns ~identity, silently discarding that "
                "edge's real motion\n  and causing severe accumulated drift in "
                "registeredScene.ply.")
          print("  Measured on such a sequence: the ICP-fallback edges delivered 1.6 %% of "
                "the\n  rotation actually present. If this applies to you, re-capture by moving "
                "the\n  camera around a static object, or lower --corner-depth-tol / keep "
                "--marker-ransac\n  enabled so the marker path does not fail in the first place.")
     return pose_graph

def load_pcd(path, Filename, downsample = True, interval = 1):

     """
     load pointcloud by path and down samle (if True) based on voxel_size

     """


     global voxel_size, camera_intrinsics


     img_file = path + 'JPEGImages/' + str('%06d' % (Filename*interval)) + '.jpg'

     cad = cv2.imread(img_file)
     cad = cv2.cvtColor(cad, cv2.COLOR_BGR2RGB)

     depth_file = path + 'depth/' + str('%06d' % (Filename*interval)) + '.png'
     depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)  # 16-bit PNG
     mask = depth.copy()
     depth = convert_depth_frame_to_pointcloud(depth, camera_intrinsics)


     source = geometry.PointCloud()
     source.points = utility.Vector3dVector(depth[mask>0])
     source.colors = utility.Vector3dVector(cad[mask>0] / 255.0)

     if downsample == True:
          source = source.voxel_down_sample(voxel_size = voxel_size)

     return source


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

# --------------------------------------------------------------------------
# --gpu path. Everything above is shared: the marker detection, the RANSAC
# matching, the capture-mode heuristic, the ICP fallback and the target
# selection are called by both paths from this one module.
# --------------------------------------------------------------------------

def _match_ransac_robust_batched(p, p_prime, tol=0.01, min_points=None,
                                 iterations=200, inlier_dist=0.003,
                                 min_inliers=None, min_inlier_ratio=0.5, seed=0):
    """utils.registration.match_ransac_robust with the 200-hypothesis loop batched.

    The original fits each 3-point hypothesis in its own Python call
    (np.matrix maths, ~170 ms per edge -- measured as HALF of the GPU path's
    wall clock). Here the same 200 index triplets are drawn from the same
    seeded generator IN THE SAME ORDER, then centroids / H / SVD / residuals
    run as one (200,...) batch. Same BLAS/LAPACK kernels, same acceptance
    rules, first-argmax consensus selection like the sequential loop --
    verified transform-identical on every edge of the box sequence.
    """
    from utils.registration import (MIN_MATCH_POINTS, _rigid_fit, _residuals,
                                    _trimmed_error)
    if min_points is None:
        min_points = MIN_MATCH_POINTS
    if min_inliers is None:
        min_inliers = MIN_MATCH_POINTS
    p = np.asarray(p, dtype=np.float64)
    p_prime = np.asarray(p_prime, dtype=np.float64)
    if len(p) != len(p_prime):
        return None, {"n": len(p), "rmse": float("nan"), "n_inliers": 0,
                      "reason": "size_mismatch"}
    n = len(p)
    if n < min_points:
        return None, {"n": n, "rmse": float("nan"), "n_inliers": 0,
                      "reason": "too_few_points"}

    rng = np.random.default_rng(seed)
    idx = np.stack([rng.choice(n, size=3, replace=False)
                    for _ in range(iterations)])            # same draw order
    a = p[idx]                                              # (it, 3, 3)
    b = p_prime[idx]

    # near-collinear samples are skipped, exactly like the original
    ok = ((np.linalg.norm(np.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0]), axis=1) >= 1e-9)
          & (np.linalg.norm(np.cross(b[:, 1] - b[:, 0], b[:, 2] - b[:, 0]), axis=1) >= 1e-9))
    if not ok.any():
        return None, {"n": n, "rmse": float("nan"), "n_inliers": 0,
                      "reason": "no_hypothesis"}
    a, b = a[ok], b[ok]

    ca = a.mean(axis=1, keepdims=True)                      # batched rigid fits
    cb = b.mean(axis=1, keepdims=True)
    H = np.einsum("kij,kil->kjl", a - ca, b - cb)
    try:
        U, S, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:                           # fall back to the original
        return match_ransac_robust(p, p_prime, tol=tol, min_points=min_points,
                                   iterations=iterations, inlier_dist=inlier_dist,
                                   min_inliers=min_inliers,
                                   min_inlier_ratio=min_inlier_ratio, seed=seed)
    R = np.einsum("kji,klj->kil", Vt, U)                    # Vt.T @ U.T
    refl = np.linalg.det(R) < 0                             # reflection case
    if refl.any():
        Vt = Vt.copy()
        Vt[refl, 2, :] *= -1
        R[refl] = np.einsum("kji,klj->kil", Vt[refl], U[refl])
    t = cb[:, 0, :] - np.einsum("kij,kj->ki", R, ca[:, 0, :])

    proj = np.einsum("kij,nj->kni", R, p) + t[:, None, :]   # (it, n, 3)
    inl = np.linalg.norm(proj - p_prime[None], axis=2) < inlier_dist
    counts = inl.sum(axis=1)
    best = int(np.argmax(counts))                           # first max, like the loop
    best_inliers = inl[best]

    n_in = int(best_inliers.sum())
    if n_in < max(min_points, min_inliers) or n_in < min_inlier_ratio * n:
        return None, {"n": n, "rmse": float("nan"), "n_inliers": n_in,
                      "reason": "too_few_inliers"}
    T = _rigid_fit(p[best_inliers], p_prime[best_inliers])  # refit: the original's code
    rmse = _trimmed_error(_residuals(T, p[best_inliers], p_prime[best_inliers]))
    if rmse < tol:
        return T, {"n": n, "rmse": rmse, "n_inliers": n_in, "reason": "ok_ransac"}
    return None, {"n": n, "rmse": rmse, "n_inliers": n_in, "reason": "rmse_too_high"}


class GpuCloudCache(object):
    """Voxel-downsampled clouds on the GPU, LRU-cached.

    The CPU path keeps only the odometry pair alive, so every loop-closure
    edge re-decodes and re-downsamples its target from disk. Downsampled
    clouds are ~2 MB each -- keeping a couple of hundred costs nothing.
    """

    def __init__(self, path, intr, voxel, device, capacity="auto", io_threads="auto"):
        import threading
        import torch
        from torch.utils import dlpack as tdl
        import open3d as o3d
        import open3d.core as o3c
        self.torch = torch
        self._tdl = tdl
        self.o3d, self.o3c = o3d, o3c
        self.path = path
        self.K = (float(intr["fx"]), float(intr["fy"]),
                  float(intr["ppx"]), float(intr["ppy"]))
        self.scale = float(intr["depth_scale"])
        self.voxel = voxel
        self.dev = o3c.Device(device)
        self.tdev = torch.device(device.lower())
        # decode is the threaded part (cv2 releases the GIL); the GPU maths is
        # serial behind a lock and takes ~10 ms/frame, so a handful of decode
        # threads saturates it
        if io_threads in (None, "auto"):
            io_threads = min(16, max(4, (os.cpu_count() or 8) // 3))
        self.io_threads = int(io_threads)
        self.capacity_spec = capacity      # resolved after the first build
        self.capacity = None
        self.cache = collections.OrderedDict()
        self.pool = futures.ThreadPoolExecutor(self.io_threads)
        self.hits = self.misses = 0
        self.t_load = 0.0
        self._lock = threading.Lock()      # cache bookkeeping
        self._gpu_lock = threading.Lock()  # decode runs parallel; GPU work serial
        self._pending = {}                 # fid -> Future, so a frame builds once
        self._grids = None                 # (u, v) pixel grids, built once on the GPU

    def _decode(self, fid):
        import cv2
        bgr = cv2.imread(self.path + "JPEGImages/%06d.jpg" % (fid * LABEL_INTERVAL))
        depth = cv2.imread(self.path + "depth/%06d.png" % (fid * LABEL_INTERVAL),
                           cv2.IMREAD_UNCHANGED)
        if bgr is None or depth is None:
            sys.exit("Could not read frame %d" % fid)
        return bgr, depth

    def _build(self, bgr, depth):
        """Back-project + downsample entirely on the GPU.

        Uploading the raw depth (1.8 MB) and image (2.7 MB) and doing the
        back-projection in torch beats the numpy version measured head-to-head:
        71 ms/frame of numpy maths becomes ~2 ms of GPU maths, and the H2D
        transfer shrinks too (raw frames are smaller than unpacked xyz+rgb).
        """
        torch, tdl = self.torch, self._tdl
        fx, fy, ppx, ppy = self.K
        with self._gpu_lock:
            if self._grids is None:
                h, w = depth.shape
                v, u = torch.meshgrid(
                    torch.arange(h, dtype=torch.float32, device=self.tdev),
                    torch.arange(w, dtype=torch.float32, device=self.tdev),
                    indexing="ij")
                self._grids = (u.reshape(-1), v.reshape(-1))
            u, v = self._grids
            dep = torch.from_numpy(depth.astype(np.float32)).to(self.tdev).reshape(-1)
            img = torch.from_numpy(bgr).to(self.tdev).reshape(-1, 3)
            idx = torch.nonzero(dep > 0).view(-1)
            z = dep[idx] * self.scale
            x = (u[idx] - ppx) / fx * z
            y = (v[idx] - ppy) / fy * z
            pts = torch.stack([x, y, z], dim=1)
            col = img[idx].flip(1).to(torch.float32) / 255.0  # BGR -> RGB
            t = self.o3d.t.geometry.PointCloud(self.dev)
            t.point.positions = self.o3c.Tensor.from_dlpack(tdl.to_dlpack(pts))
            t.point.colors = self.o3c.Tensor.from_dlpack(tdl.to_dlpack(col))
            return t.voxel_down_sample(self.voxel)

    def _load(self, fid):
        t0 = time.perf_counter()
        pcd = self._build(*self._decode(fid))
        with self._lock:
            if self.capacity is None:  # resolve 'auto' from the first cloud's size
                self.capacity = self._resolve_capacity(pcd)
            self.cache[fid] = pcd
            self.misses += 1
            self.t_load += time.perf_counter() - t0
            self._pending.pop(fid, None)
            while len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        return pcd

    def _resolve_capacity(self, sample_pcd):
        if self.capacity_spec not in (None, "auto"):
            return int(self.capacity_spec)
        per_frame = int(sample_pcd.point.positions.shape[0]) * 24  # xyz+rgb float32
        try:
            free, _ = self.torch.cuda.mem_get_info(self.tdev)
        except Exception:
            free = 8e9
        cap = max(64, int(free * 0.35 / max(1, per_frame)))
        print("Cloud cache: auto capacity %d frames (%.1f MB each, %.1f GB free)"
              % (cap, per_frame / 1e6, free / 1e9))
        return cap

    def get(self, fid):
        with self._lock:
            if fid in self.cache:
                self.cache.move_to_end(fid)
                self.hits += 1
                return self.cache[fid]
            fut = self._pending.get(fid)
            if fut is None:
                fut = self.pool.submit(self._load, fid)
                self._pending[fid] = fut
        return fut.result()

    def prefetch(self, fids):
        """Warm the cache for the frames an edge batch is about to touch."""
        with self._lock:
            missing = [f for f in fids if f not in self.cache and f not in self._pending]
            for fid in missing:
                self._pending[fid] = self.pool.submit(self._load, fid)


def _resolve_gpu_device(index):
    """Validate --gpu N up front and return the Open3D device string.

    The GPU path is opt-in, so a missing or out-of-range CUDA device is an
    error with a way out -- never a silent fall back to the CPU path, which
    would quietly turn a 13x speed-up into a puzzling slow run.
    """
    try:
        import open3d.core as o3c
        import torch
    except ImportError as exc:
        sys.exit("--gpu needs Open3D with CUDA support and PyTorch (%s).\n"
                 "Drop --gpu to run the CPU path." % exc)
    missing = ([] if o3c.cuda.is_available() else ["Open3D"]) + \
              ([] if torch.cuda.is_available() else ["PyTorch"])
    if missing:
        sys.exit("No CUDA device available to %s -- drop --gpu to run the CPU path, "
                 "which is\nthe reference implementation and writes the same "
                 "transforms.npy." % " and ".join(missing))
    n_dev = min(o3c.cuda.device_count(), torch.cuda.device_count())
    if index < 0 or index >= n_dev:
        sys.exit("--gpu %d: no such CUDA device -- this machine has %d (valid: 0..%d).\n"
                 "Drop --gpu to run the CPU path." % (index, n_dev, n_dev - 1))
    return "CUDA:%d" % index


def full_registration_gpu(path, device, cache_frames="auto", io_threads="auto"):
    """full_registration()'s edge loop, GPU-weighted (see the module docstring).

    Same pose graph, same log.txt; the clouds live on `device` in an LRU cache
    and the per-edge information matrix is computed there too. Returns
    (pose_graph, timings) -- the caller runs the same CPU global optimization
    on the graph either way.
    """
    import open3d.core as o3c
    treg = o3d.t.pipelines.registration

    if io_threads in ("auto", None):
        io_threads = min(16, max(4, (os.cpu_count() or 8) // 3))
    io_threads = int(io_threads)

    # ---- markers for every frame, on threads (cv2 releases the GIL) -------
    t0 = time.perf_counter()
    with futures.ThreadPoolExecutor(io_threads) as pool:
        list(pool.map(lambda i: get_marker_features(path, i), range(n_pcds)))
    t_markers = time.perf_counter() - t0
    n_with = sum(1 for i in range(n_pcds) if _marker_cache[i][0] is not None)
    print("Marker features: %d/%d frames have markers (%.1fs on %d threads)"
          % (n_with, n_pcds, t_markers, io_threads))

    clouds = GpuCloudCache(path, camera_intrinsics, voxel_size, device,
                           capacity=cache_frames, io_threads=io_threads)

    # ---- the original's edge loop, GPU-weighted ----------------------------
    pose_graph = registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(registration.PoseGraphNode(odometry))

    log = open((OUT_PATH or path) + "log.txt", "w")
    log.write("log starts (gpu): frames=%d voxel_size=%g label_interval=%d "
              "k_neighbors=%d icp_method=%s marker_ransac=%s corner_depth_tol=%g "
              "max_loop_closures=%d capture_mode=%s device=%s\n"
              % (n_pcds, voxel_size, LABEL_INTERVAL, N_Neighbours, ICP_METHOD,
                 MARKER_RANSAC, CORNER_DEPTH_TOL, MAX_LOOP_CLOSURES, CAPTURE_MODE,
                 device))

    count = n_icp_fallback = 0
    t_info = t_icp = 0.0
    stride = max(1, int(n_pcds / N_Neighbours))

    for source_id in trange(n_pcds, desc="edges (GPU)"):
        targets = _select_targets(source_id, n_pcds, stride)
        clouds.prefetch([source_id] + targets)
        for target_id in targets:
            is_odometry = (target_id == source_id + 1)
            use_robust = (MARKER_RANSAC == "all" or
                          (MARKER_RANSAC == "odometry" and is_odometry))
            res, minfo = marker_registration(
                get_marker_features(path, source_id),
                get_marker_features(path, target_id), robust=use_robust)
            if res is None and not is_odometry:
                continue
            diag = "n=%d rmse=%.5f inliers=%d reason=%s" % (
                minfo["n"], minfo["rmse"], minfo["n_inliers"], minfo["reason"])

            tsrc = clouds.get(source_id)
            tdst = clouds.get(target_id)

            if res is None:
                # rare fallback: identical to the CPU path, on the CPU
                t1 = time.perf_counter()
                src_l, dst_l = tsrc.to_legacy(), tdst.to_legacy()
                if CAPTURE_MODE == "static-camera":
                    src_l = crop_to_marker_region(
                        src_l, get_marker_features(path, source_id)[1] or [])
                    dst_l = crop_to_marker_region(
                        dst_l, get_marker_features(path, target_id)[1] or [])
                for pc in (src_l, dst_l):
                    if not pc.has_normals():
                        pc.estimate_normals(geometry.KDTreeSearchParamHybrid(
                            radius=0.002 * 2, max_nn=30))
                transformation_icp, information_icp = icp(
                    src_l, dst_l, voxel_size, max_correspondence_distance_coarse,
                    max_correspondence_distance_fine, method=ICP_METHOD)
                n_icp_fallback += 1
                t_icp += time.perf_counter() - t1
                log.write("%d-%d/%d,%d/%d, None_icp, %s\n"
                          % (count, source_id, n_pcds, target_id, n_pcds, diag))
            else:
                t1 = time.perf_counter()
                transformation_icp = res
                information_icp = treg.get_information_matrix(
                    tsrc, tdst, max_correspondence_distance_fine,
                    o3c.Tensor(np.asarray(res, dtype=np.float64))).cpu().numpy()
                t_info += time.perf_counter() - t1
                log.write("%d-%d/%d,%d/%d, res_marker, %s\n"
                          % (count, source_id, n_pcds, target_id, n_pcds, diag))

            if is_odometry:
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(registration.PoseGraphEdge(
                    source_id, target_id, transformation_icp, information_icp,
                    uncertain=False))
            else:
                pose_graph.edges.append(registration.PoseGraphEdge(
                    source_id, target_id, transformation_icp, information_icp,
                    uncertain=True))
            count += 1

    log.write("edges=%d marker=%d icp_fallback=%d\n"
              % (count, count - n_icp_fallback, n_icp_fallback))
    log.close()
    if n_icp_fallback:
        print("\nWARNING: %d of %d edges fell back to full-frame ICP -- see the warning "
              "in full_registration() for why that matters on static-camera captures."
              % (n_icp_fallback, count))

    return pose_graph, {"markers": t_markers, "load": clouds.t_load,
                        "misses": clouds.misses, "hits": clouds.hits,
                        "info": t_info, "icp": t_icp, "fallbacks": n_icp_fallback}


def main():
    global OUT_PATH, LABEL_INTERVAL, K_NEIGHBORS, N_Neighbours, ICP_METHOD
    global MARKER_RANSAC, CORNER_DEPTH_TOL, MAX_LOOP_CLOSURES, CAPTURE_MODE
    global voxel_size, max_correspondence_distance_coarse
    global max_correspondence_distance_fine, camera_intrinsics, n_pcds
    global _ROBUST_ESTIMATOR

    parser = build_parser("ArUco + ICP pose-graph registration -> transforms.npy")
    parser.add_argument("--label-interval", type=int, default=LABEL_INTERVAL,
                        help="use every Nth frame for pose labeling")
    parser.add_argument("--k-neighbors", type=int, default=K_NEIGHBORS,
                        help="number of loop-closure candidates per frame")
    parser.add_argument("--voxel-size", type=float, default=VOXEL_SIZE,
                        help="voxel size (m) for ICP downsampling")
    parser.add_argument("--icp-method", choices=["point-to-plane", "colored-icp"],
                        default=ICP_METHOD, help="ICP variant for the fallback registration")
    parser.add_argument("--marker-ransac", choices=["odometry", "all", "off"],
                        default=MARKER_RANSAC,
                        help="use the robust RANSAC marker estimator on: consecutive-frame "
                             "edges only (odometry), every candidate pair (all -- accepts far "
                             "more loop closures, much slower), or nowhere (off -- the "
                             "historical least-squares estimator)")
    parser.add_argument("--corner-depth-tol", type=float, default=CORNER_DEPTH_TOL,
                        help="drop a marker corner whose depth deviates from its own marker's "
                             "median corner depth by more than this many metres (background "
                             "bleed at marker borders); 0 disables the filter")
    parser.add_argument("--capture-mode", choices=["auto", "moving-camera", "static-camera"],
                        default=CAPTURE_MODE,
                        help="how the sequence was captured. 'auto' (default) detects it from "
                             "the data: a static background means the camera did not move and "
                             "the object was turned instead, in which case the ICP fallback is "
                             "restricted to the marker/object region so it cannot lock onto the "
                             "static room")
    parser.add_argument("--out-dir", default=None,
                        help="where to write transforms.npy and log.txt (default: the sequence "
                             "folder, which is where step 3 expects them)")
    parser.add_argument("--max-loop-closures", type=int, default=MAX_LOOP_CLOSURES,
                        help="maximum loop-closure edges accepted per source frame; each costs "
                             "roughly a second (point-cloud load + information matrix). "
                             "0 = pure odometry chain (fastest)")
    parser.add_argument("--gpu", nargs="?", const=0, default=None, type=int, metavar="N",
                        help="run the GPU-accelerated path on CUDA device N (a bare --gpu "
                             "means device 0). Omit it for the CPU path, which is the "
                             "default and the reference implementation")
    parser.add_argument("--cache-frames", default="auto",
                        help="--gpu only: downsampled clouds kept on the GPU (LRU); 'auto' "
                             "sizes from free GPU memory")
    parser.add_argument("--io-threads", default="auto",
                        help="--gpu only: parallel frame decoders; 'auto' derives from the "
                             "CPU count")
    args = parser.parse_args()
    dataset, path = resolve_dataset(
        args, require=("intrinsics.json", "JPEGImages", "depth"))

    # fail before any work if the requested GPU is not there (no silent fallback)
    device = None if args.gpu is None else _resolve_gpu_device(args.gpu)

    OUT_PATH = path if args.out_dir is None else normalize_root(args.out_dir)
    if not os.path.isdir(OUT_PATH):
        os.makedirs(OUT_PATH)

    # override the config-file defaults with the CLI values
    LABEL_INTERVAL = args.label_interval
    K_NEIGHBORS = args.k_neighbors
    N_Neighbours = args.k_neighbors
    ICP_METHOD = args.icp_method
    MARKER_RANSAC = args.marker_ransac
    CORNER_DEPTH_TOL = args.corner_depth_tol
    MAX_LOOP_CLOSURES = args.max_loop_closures
    CAPTURE_MODE = args.capture_mode
    voxel_size = args.voxel_size
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5

    with open(path+'intrinsics.json', 'r') as f:
         camera_intrinsics = json.load(f)
    Ts = []
    n_pcds = int(len(glob.glob1(path+"JPEGImages","*.jpg"))/LABEL_INTERVAL)
    if n_pcds == 0:
        sys.exit("No frames found in %sJPEGImages." % path)

    if CAPTURE_MODE == "auto":
        CAPTURE_MODE, bg_change = detect_capture_mode(path, n_pcds)
        print("Capture mode: %s (background depth changes by %.1f mm between "
              "distant frames; < %.0f mm means the camera did not move)"
              % (CAPTURE_MODE, bg_change * 1000, STATIC_BG_TOL * 1000))
        if CAPTURE_MODE == "static-camera":
            print("  -> the object was moved, not the camera. The ICP fallback will be "
                  "restricted to\n     the marker/object region so it cannot lock onto the "
                  "static background -- but note\n     that on a flat marker board even that "
                  "cannot recover in-plane rotation, so the\n     markers are the only reliable "
                  "signal here. Override with --capture-mode if wrong.")
    else:
        print("Capture mode: %s (set explicitly)" % CAPTURE_MODE)

    if device is None:
        print("Full registration ...")
        pose_graph = full_registration(path, max_correspondence_distance_coarse,
                                       max_correspondence_distance_fine)
        timings = None
    else:
        print("Full registration on %s ..." % device)
        # same RANSAC, batched (measured at half the GPU path's wall clock otherwise)
        _ROBUST_ESTIMATOR = _match_ransac_robust_batched
        pose_graph, timings = full_registration_gpu(
            path, device, args.cache_frames, args.io_threads)

    print("Optimizing PoseGraph ...")
    t_opt = time.perf_counter()
    option =registration.GlobalOptimizationOption(
            max_correspondence_distance = max_correspondence_distance_fine,
            edge_prune_threshold = 0.25,
            reference_node = 0)
    registration.global_optimization(pose_graph,
                                     registration.GlobalOptimizationLevenbergMarquardt(),
                                     registration.GlobalOptimizationConvergenceCriteria(), option)
    t_opt = time.perf_counter() - t_opt
    num_annotations = int(len(glob.glob1(path+"JPEGImages","*.jpg"))/LABEL_INTERVAL)
    for point_id in range(num_annotations):
         Ts.append(pose_graph.nodes[point_id].pose)
    Ts = np.array(Ts)
    filename = OUT_PATH + 'transforms.npy'
    np.save(filename, Ts)
    print("Saved %d poses to %s (per-edge diagnostics in %slog.txt)"
          % (len(Ts), filename, OUT_PATH))
    if timings is not None:
        print("[timing] markers %.1fs | cloud load+downsample %.1fs (%d loads, %d cache "
              "hits) | info-matrix %.1fs | icp-fallback %.1fs (%d edges) | graph-opt %.1fs"
              % (timings["markers"], timings["load"], timings["misses"], timings["hits"],
                 timings["info"], timings["icp"], timings["fallbacks"], t_opt))
    print("Next: python 3_register_scene.py %s%s"
          % (args.dataset or dataset, "" if args.gpu is None else " --gpu %d" % args.gpu))


if __name__ == "__main__":
    main()
