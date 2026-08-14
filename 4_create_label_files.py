"""
4_create_label_files.py
-----------------------

This script produces:

1. Reorient the processed registered_scene mesh in a mesh with an AABB centered at the
   origin and the same dimensions as the OBB, saved under the name foldername.ply
2. Create label files with class labels and projections of 3D BBs in the format
   singleshotpose requires, saved under labels
3. Create pixel-wise masks, saved under mask
4. Save the homogeneous transform of object in regards to the foldername.ply in each
   frame

How it works: the object mesh is sampled ONCE in its OBB/centroid frame and each
frame only rigid-transforms those sample points (rigid transforms preserve face
areas, so the sampling distribution is identical). The mask is rasterized by
plotting the truncated-int sample pixels and dilating them with the exact disk
kernel cv2.circle(radius=5) produces -- a union of disks is a dilation by that
same disk. Frames are split into contiguous chunks across worker processes
(--workers). Together this runs ~8x faster end to end than the per-frame
formulation it replaces (~100x on the label loop itself).

    python 4_create_label_files.py mycup
    python 4_create_label_files.py mycup --class-label 3 --sample-points 100000
    python 4_create_label_files.py mycup --input-cloud clean_sfm.ply   # older datasets
    python 4_create_label_files.py mycup --workers 4
"""

import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import trimesh
import open3d as o3d

from config.registrationParameters import *
from utils.cli import build_parser, resolve_dataset

MASK_RADIUS = 5  # cv2.circle radius used by the original rasterizer


def get_camera_intrinsic(folder):
    with open(folder+'intrinsics.json', 'r') as f:
        camera_intrinsics = json.load(f)


    K = np.zeros((3, 3), dtype='float64')
    K[0, 0], K[0, 2] = float(camera_intrinsics['fx']), float(camera_intrinsics['ppx'])
    K[1, 1], K[1, 2] = float(camera_intrinsics['fy']), float(camera_intrinsics['ppy'])

    K[2, 2] = 1.
    return (camera_intrinsics, K)

def compute_projection(points_3D,internal_calibration):
    points_3D = points_3D.T
    projections_2d = np.zeros((2, points_3D.shape[1]), dtype='float32')
    camera_projection = (internal_calibration).dot(points_3D)
    projections_2d[0, :] = camera_projection[0, :]/camera_projection[2, :]
    projections_2d[1, :] = camera_projection[1, :]/camera_projection[2, :]
    return projections_2d


def _disk_kernel(radius=MASK_RADIUS):
    """The exact filled disk cv2.circle draws, as a dilation kernel."""
    k = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    cv2.circle(k, (radius, radius), radius, 255, -1)
    return k


def _mask_path(path_mask, frame_id):
    if frame_id < 10000:
        return path_mask + "/" + str("%04d" % frame_id) + ".png"
    return path_mask + "/" + str("%06d" % frame_id) + ".png"


def _rasterize(masks, img_h, img_w, kernel, radius=MASK_RADIUS):
    """Union of filled radius-5 disks at int-truncated sample pixels.

    Replicates a per-point `cv2.circle(..., int(pixel), 5, 255, -1)` loop
    guarded by a bare try/except: non-finite coords are skipped; points
    outside the image still contribute clipped disks (handled by the 5px pad).
    """
    finite = np.isfinite(masks).all(axis=1)
    pts = np.trunc(masks[finite]).astype(np.int64)  # int() truncates toward zero
    pad = radius
    hp, wp = img_h + 2 * pad, img_w + 2 * pad
    px = pts[:, 0] + pad
    py = pts[:, 1] + pad
    keep = (px >= 0) & (px < wp) & (py >= 0) & (py < hp)
    canvas = np.zeros((hp, wp), dtype=np.uint8)
    canvas[py[keep], px[keep]] = 255
    canvas = cv2.dilate(canvas, kernel)
    return np.ascontiguousarray(canvas[pad:pad + img_h, pad:pad + img_w])


def _process_chunk(start, transforms_chunk, sample_points_obb, points_original,
                   K, cam_w, cam_h, img_h, img_w, inv_Tform,
                   path_transforms, path_mask, path_label, classlabel,
                   label_interval):
    """Label/mask/transform generation for frames [start, start+len(chunk)).

    Returns the list of frame indices that had no contour while no fallback
    contour existed yet inside this chunk (repaired serially afterwards).
    """
    kernel = _disk_kernel()
    cnt = None  # falls back to the previous frame's contour when a frame has none
    unresolved = []
    for j in range(len(transforms_chunk)):
        i = start + j

        transform = np.linalg.inv(transforms_chunk[j])
        transformed = trimesh.transformations.transform_points(points_original, transform)

        corners = compute_projection(transformed, K)
        corners = corners.T
        corners[:, 0] = corners[:, 0] / cam_w
        corners[:, 1] = corners[:, 1] / cam_h

        T = np.dot(transform, inv_Tform)
        filename = path_transforms + "/" + str(i * label_interval) + ".npy"
        np.save(filename, T)

        sample_points = trimesh.transformations.transform_points(sample_points_obb, T)
        masks = compute_projection(sample_points, K)
        masks = masks.T

        # bbox extents from the RAW projected coords, before any clipping
        min_x = np.min(masks[:, 0])
        min_y = np.min(masks[:, 1])
        max_x = np.max(masks[:, 0])
        max_y = np.max(masks[:, 1])

        image_mask = _rasterize(masks, img_h, img_w, kernel)

        thresh = cv2.threshold(image_mask, 30, 255, cv2.THRESH_BINARY)[1]

        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[-2]  # OpenCV 3/4 compatible
        try:
            cnt = max(contours, key=cv2.contourArea)
        except Exception:
            pass

        image_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        if cnt is not None:
            cv2.drawContours(image_mask, [cnt], -1, 255, -1)
        else:
            # No fallback available inside this chunk; repaired serially later
            unresolved.append(i)

        frame_id = i * label_interval
        cv2.imwrite(_mask_path(path_mask, frame_id), image_mask)

        file = open(path_label + "/" + str("%06d" % (i * label_interval)) + ".txt", "w")
        message = str(classlabel)[:8] + " "
        file.write(message)
        for pixel in corners:
            for digit in pixel:
                message = str(digit)[:8] + " "
                file.write(message)
        message = str((max_x - min_x) / float(cam_w))[:8] + " "
        file.write(message)
        message = str((max_y - min_y) / float(cam_h))[:8]
        file.write(message)
        file.close()
    return unresolved


def _fixup_unresolved(unresolved, path_mask, img_h, img_w, label_interval):
    """Serial repair of chunk-leading frames that had no contour.

    The `cnt` fallback crosses frame boundaries: a frame with no contour
    silently reuses the previous frame's contour. Re-fill such frames in order
    from the previous frame's written mask; only frames whose predecessor is
    also empty (i.e. frames a serial run would also leave empty) keep the
    empty mask + warning.
    """
    for i in sorted(unresolved):
        frame_id = i * label_interval
        cnt = None
        if i > 0:
            prev = cv2.imread(_mask_path(path_mask, (i - 1) * label_interval),
                              cv2.IMREAD_GRAYSCALE)
            if prev is not None:
                contours = cv2.findContours(prev, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)[-2]
                if len(contours):
                    cnt = max(contours, key=cv2.contourArea)
        image_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        if cnt is not None:
            cv2.drawContours(image_mask, [cnt], -1, 255, -1)
        else:
            print("WARNING: frame %d has no mask contour (object out of view?) "
                  "-- empty mask written" % frame_id)
        cv2.imwrite(_mask_path(path_mask, frame_id), image_mask)


def DoLabel(path, number, defaultSnumber=100000, class_label=0,
            input_cloud="object.ply", label_interval=1, workers=1):
    t0 = time.time()
    folders = [path + str(number) + "/"]
    pcd = o3d.io.read_point_cloud(
        path + str(number) + "/" + input_cloud)  # the manually segmented object cloud
    for folder in folders:
        classlabel = class_label
        print("%s is assigned class label %d." % (folder, classlabel))
        camera_intrinsics, K = get_camera_intrinsic(folder)
        path_label = folder + "labels"
        if not os.path.exists(path_label):
            os.makedirs(path_label)

        path_mask = folder + "mask"
        if not os.path.exists(path_mask):
            os.makedirs(path_mask)

        path_transforms = folder + "transforms"
        if not os.path.exists(path_transforms):
            os.makedirs(path_transforms)

        transforms_file = folder + 'transforms.npy'
        transforms = np.load(transforms_file)

        print("Load ply Input file ...")

        pcd.estimate_normals()

        # estimate radius for rolling ball
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 1.5 * avg_dist

        pointcloud = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector([radius, radius * 2]))

        # forward the input cloud's vertex colours to the mesh (see doc/DETAILS.md §4)
        vertex_colors = np.asarray(pointcloud.vertex_colors)
        if vertex_colors.size:
            vertex_colors = (vertex_colors * 255).astype(np.uint8)
        else:
            vertex_colors = None
            print("Input cloud has no colour; %s.ply will be geometry-only." % number)
        mesh = trimesh.Trimesh(np.asarray(pointcloud.vertices), np.asarray(pointcloud.triangles),
                               vertex_normals=np.asarray(pointcloud.vertex_normals),
                               vertex_colors=vertex_colors)

        # world-frame mesh, for reference only
        trimesh.exchange.export.export_mesh(mesh, folder + "mesh_world.ply",
                                            encoding='ascii')

        ####################################################################

        # Align the mesh to its oriented bounding box (baseline behaviour).
        Tform = mesh.apply_obb()

        # Put the centroid on the origin: label keypoint 0 (mesh.centroid) and
        # singleshotpose's first model point (the origin) must be the same
        # physical point. See doc/DETAILS.md §4.
        centroid_obb = np.array(mesh.centroid, dtype=np.float64)
        mesh.apply_translation(-centroid_obb)
        Tform = trimesh.transformations.translation_matrix(-centroid_obb).dot(Tform)

        # The mesh singleshotpose loads; must be ASCII (MeshPly reads text).
        # Vertex layout: x y z  nx ny nz  red green blue alpha (colours in 6:9).
        mesh_name = str(number) + ".ply"
        trimesh.exchange.export.export_mesh(mesh, folder + mesh_name, encoding='ascii')
        print("Object mesh (centroid on the origin, ASCII): %s" % (folder + mesh_name))

        points = mesh.bounding_box.vertices
        center = mesh.centroid

        min_x = np.min(points[:, 0])
        min_y = np.min(points[:, 1])
        min_z = np.min(points[:, 2])
        max_x = np.max(points[:, 0])
        max_y = np.max(points[:, 1])
        max_z = np.max(points[:, 2])
        points = np.array([[min_x, min_y, min_z], [min_x, min_y, max_z], [min_x, max_y, min_z],
                           [min_x, max_y, max_z], [max_x, min_y, min_z], [max_x, min_y, max_z],
                           [max_x, max_y, min_z], [max_x, max_y, max_z]])

        points_original = np.concatenate((np.array([[center[0], center[1], center[2]]]), points))
        points_original = trimesh.transformations.transform_points(points_original,
                                                                   np.linalg.inv(Tform))  # Tform

        # Sample the (OBB/centroid frame) mesh ONCE; per frame the samples are
        # only rigid-transformed, which preserves the sampling distribution
        # exactly (face areas are invariant under rigid T).
        sample_points_obb = mesh.sample(defaultSnumber)
        inv_Tform = np.linalg.inv(Tform)

        # image shape is constant across the sequence; read one frame once
        img = cv2.imread(folder + "JPEGImages/" + "{:06d}".format(0) + ".jpg")
        if img is None:
            import glob as _glob
            first = sorted(_glob.glob(folder + "JPEGImages/*.jpg"))
            if not first:
                sys.exit("No JPEG frames found in %sJPEGImages/" % folder)
            img = cv2.imread(first[0])
        img_h, img_w = img.shape[:2]
        cam_w = int(camera_intrinsics['width'])
        cam_h = int(camera_intrinsics['height'])

        t_setup = time.time() - t0

        n = len(transforms)
        n_workers = max(1, min(workers, n))
        t1 = time.time()
        common = (sample_points_obb, points_original, K, cam_w, cam_h,
                  img_h, img_w, inv_Tform, path_transforms, path_mask,
                  path_label, classlabel, label_interval)
        if n_workers == 1:
            unresolved = _process_chunk(0, transforms, *common)
        else:
            bounds = np.linspace(0, n, n_workers + 1).astype(int)
            unresolved = []
            ctx = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                futures = [pool.submit(_process_chunk, int(bounds[w]),
                                       transforms[bounds[w]:bounds[w + 1]], *common)
                           for w in range(n_workers) if bounds[w + 1] > bounds[w]]
                for f in futures:
                    unresolved.extend(f.result())

        if unresolved:
            _fixup_unresolved(unresolved, path_mask, img_h, img_w, label_interval)

        t_loop = time.time() - t1
        print("[timing] setup=%.2fs loop=%.2fs per_frame=%.4fs workers=%d"
              % (t_setup, t_loop, t_loop / max(n, 1), n_workers))

    print("\n%s is done: %d frames labeled." % (number, len(transforms)))


def _parse_workers(value):
    if value == "auto":
        return min(os.cpu_count() or 1, 8)
    try:
        w = int(value)
    except ValueError:
        sys.exit("--workers must be 'auto' or a positive integer, got %r" % value)
    if w < 1:
        sys.exit("--workers must be >= 1")
    return w


if __name__ == "__main__":
    parser = build_parser("Ball-pivoting mesh + OBB + label/mask/transform generation")
    parser.add_argument("--class-label", type=int, default=0,
                        help="class id written as the first value of each label file "
                             "(must match the object's index in the sspose .data cfg)")
    parser.add_argument("--sample-points", type=int, default=100000,
                        help="number of mesh sample points used to rasterize the mask")
    parser.add_argument("--input-cloud", default="object.ply",
                        help="the manually segmented object point cloud inside the sequence "
                             "folder -- i.e. registeredScene.ply with everything except the "
                             "object deleted in CloudCompare/MeshLab, saved as ASCII PLY. "
                             "Older datasets from this project called it clean_sfm.ply")
    parser.add_argument("--label-interval", type=int, default=LABEL_INTERVAL,
                        help="frame interval used in 2_compute_gt_poses.py")
    parser.add_argument("--workers", default="auto",
                        help="worker processes for the frame loop: 'auto' = "
                             "min(cpu_count, 8); '1' disables multiprocessing")
    args = parser.parse_args()
    dataset, dpath = resolve_dataset(
        args, require=("intrinsics.json", "transforms.npy", "JPEGImages", args.input_cloud))

    workers = _parse_workers(args.workers)
    parent = dpath[:-(len(dataset) + 1)]  # resolve_dataset guarantees dpath == parent + dataset + "/"
    DoLabel(parent, dataset, defaultSnumber=args.sample_points,
            class_label=args.class_label, input_cloud=args.input_cloud,
            label_interval=args.label_interval, workers=workers)
    print("Next: python 5_create_config2split.py %s" % (args.dataset or dataset))
