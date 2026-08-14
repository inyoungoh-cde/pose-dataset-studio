"""
utils/registration.py
---------------------

Rigid-alignment estimators used by `2_compute_gt_poses.py`: given two sets of
corresponding 3D points, or two point clouds, return the 4x4 transform between
them. Nothing else in the pipeline imports this -- step 3 reads the poses step 2
already wrote and never registers anything itself.

Baseline ODT code, with three changes: `xrange` -> `range`, an open3d shim (see
below), and `match_ransac_robust`, added in Pass 4 and now step 2's default
estimator. The baseline's SIFT-based `feature_registration` was removed with it:
it was never called from anywhere, and it needed `cv2.xfeatures2d`, which is not
in the pinned opencv build.

"""

import open3d as o3d
import numpy as np

# open3d <= 0.9 keeps registration at the top level; >= 0.10 moved it
# under o3d.pipelines. Resolve once so the rest of the code is agnostic.
geometry = o3d.geometry
utility = o3d.utility
try:
    registration = o3d.registration
except AttributeError:
    registration = o3d.pipelines.registration

def icp(source,target,voxel_size,max_correspondence_distance_coarse,max_correspondence_distance_fine,
        method = "colored-icp"):

    """
    Perform pointcloud registration using iterative closest point.

    Parameters
    ----------
    source : An open3d.Pointcloud instance
      6D pontcloud of a source segment
    target : An open3d.Pointcloud instance
      6D pointcloud of a target segment
    method : string
      colored-icp, as in Park, Q.-Y. Zhou, and V. Koltun, Colored Point Cloud 
      Registration Revisited, ICCV, 2017 (slower)
      point-to-plane, a coarse to fine implementation of point-to-plane icp (faster)
    max_correspondence_distance_coarse : float
      The max correspondence distance used for the course ICP during the process
      of coarse to fine registration (if point-to-plane)
    max_correspondence_distance_fine : float
      The max correspondence distance used for the fine ICP during the process 
      of coarse to fine registration (if point-to-plane)

    Returns
    ----------
    transformation_icp: (4,4) float
      The homogeneous rigid transformation that transforms source to the target's
      frame
    information_icp:
      An information matrix returned by open3d.get_information_matrix_from_ \
      point_clouds function
    """


    assert method in ["point-to-plane","colored-icp"],"point-to-plane or colored-icp"
    if method == "point-to-plane":
        icp_coarse = registration.registration_icp(source, target,
                                                   max_correspondence_distance_coarse, np.identity(4),
                                                   registration.TransformationEstimationPointToPlane())
        icp_fine = registration.registration_icp(source, target,
                max_correspondence_distance_fine, icp_coarse.transformation,
                registration.TransformationEstimationPointToPlane())

        transformation_icp = icp_fine.transformation


    if method == "colored-icp":
        criteria = registration.ICPConvergenceCriteria(relative_fitness = 1e-8,
                                                       relative_rmse = 1e-8, max_iteration = 50)
        try:
            # open3d <= 0.11: criteria is the 5th positional argument
            result_icp = registration.registration_colored_icp(source, target, voxel_size,
                                                               np.identity(4), criteria)
        except TypeError:
            # open3d >= 0.12: 5th slot is the estimation method, criteria is 6th
            result_icp = registration.registration_colored_icp(
                source, target, voxel_size, np.identity(4),
                registration.TransformationEstimationForColoredICP(), criteria)

        transformation_icp = result_icp.transformation

        
    information_icp = registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        transformation_icp)
    
    return transformation_icp, information_icp


# Minimum number of 3D correspondences for a well-conditioned rigid fit.
# With only 2 pairs the SVD is rank-deficient: the rotation about the axis
# through the two points is arbitrary, and because the acceptance test scores
# only the smallest int(n*0.7) residuals it still passes (an exact 2-point fit
# scores ~0). Measured effect of accepting such a fit: up to 800 mm / 76 deg of
# error on an edge that is then trusted as *certain* odometry -> one visibly
# misaligned frame in the registered scene.
MIN_MATCH_POINTS = 6


def _rigid_fit(p, p_prime):
    """SVD rigid fit of p onto p_prime, returned as a 4x4 homogeneous matrix."""
    R, t = rigid_transform_3D(p, p_prime)
    R = np.array(R)
    t = (np.array(t).T)[0]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _residuals(T, p, p_prime):
    """Per-correspondence Euclidean residual after applying T to p."""
    transformed = (np.dot(T[:3, :3], p.T).T) + T[:3, 3]
    return np.sqrt(np.sum((transformed - p_prime) ** 2, axis=1))


def _trimmed_error(residuals, frac=0.7):
    """Mean of the smallest `frac` of the residuals (the original criterion)."""
    n = len(residuals)
    k = max(1, min(n, int(n * frac)))
    return float(np.sum(residuals[np.argpartition(residuals, k - 1)[:k]]) / k)


def match_ransac(p, p_prime, tol=0.01, min_points=MIN_MATCH_POINTS):
    """
    Estimate the rigid transform between two ordered sets of 3D points by a
    single least-squares (SVD) fit, accepting it if the mean of the smallest
    70 % of the residuals is below `tol`.

    This is the pipeline's historical estimator (the name is a misnomer: there
    is no sampling and no outlier rejection, so a single grossly wrong
    correspondence — e.g. a marker corner whose depth reading is off by
    metres — contaminates the fit and causes rejection). See
    `match_ransac_robust` for a real RANSAC.

    Parameters
    ----------
    p, p_prime : (n,3) float
      Corresponding source / target 3D points.
    tol : float
      Acceptance threshold (m) on the trimmed mean residual.
    min_points : int
      Reject outright if fewer correspondences than this are available; below
      MIN_MATCH_POINTS the fit is not well conditioned (see above).

    Returns
    ----------
    (transform, info)
      transform : (4,4) numpy.ndarray or None
      info : dict with 'n', 'rmse', 'n_inliers', 'reason'
    """
    p = np.asarray(p, dtype=np.float64)
    p_prime = np.asarray(p_prime, dtype=np.float64)
    if len(p) != len(p_prime):
        return None, {"n": len(p), "rmse": float("nan"), "n_inliers": 0,
                      "reason": "size_mismatch"}
    n = len(p)
    if n < min_points:
        return None, {"n": n, "rmse": float("nan"), "n_inliers": 0,
                      "reason": "too_few_points"}

    T = _rigid_fit(p, p_prime)
    err = _residuals(T, p, p_prime)
    rmse = _trimmed_error(err)
    n_in = int(np.count_nonzero(err < tol))
    if rmse < tol:
        return T, {"n": n, "rmse": rmse, "n_inliers": n_in, "reason": "ok"}
    return None, {"n": n, "rmse": rmse, "n_inliers": n_in, "reason": "rmse_too_high"}


def match_ransac_robust(p, p_prime, tol=0.01, min_points=MIN_MATCH_POINTS,
                        iterations=200, inlier_dist=0.003,
                        min_inliers=MIN_MATCH_POINTS, min_inlier_ratio=0.5, seed=0):
    """
    Estimate the rigid transform between two ordered sets of 3D points by a
    real RANSAC: repeatedly fit from a minimal 3-point sample, keep the largest
    consensus set, then refit on that set only.

    This tolerates the gross depth outliers that occur at ArUco marker corners
    (a corner landing on a depth discontinuity can be off by metres), which the
    least-squares `match_ransac` cannot: there, one bad correspondence out of 70
    misaligns the whole fit and the pair is rejected, falling back to a slow ICP.

    The `min_inliers` / `min_inlier_ratio` gate is essential, not optional: a
    3-point sample always fits its own 3 points exactly, so without a minimum
    consensus requirement the trimmed-residual test is self-fulfilling and every
    pair would be accepted (measured: 2948 of 2948 candidate pairs). The default
    floor is twice the minimal sample, which rejects such a self-fit while still
    accepting a clean pair that only sees two markers (8 corners).

    Parameters
    ----------
    p, p_prime : (n,3) float
      Corresponding source / target 3D points.
    tol : float
      Acceptance threshold (m) on the trimmed mean residual of the inliers.
    iterations : int
      Number of minimal-sample hypotheses to try.
    inlier_dist : float
      Residual (m) below which a correspondence counts as an inlier. This must
      be on the order of the depth-measurement noise (≈2.5 mm for a clean ArUco
      corner) and, critically, **below the inter-frame motion of the corners**.
      Measured on a 529-frame turntable sequence whose corners move 3.0 mm per
      frame: at inlier_dist=0.01 the identity transform is itself a ~100 %-inlier
      hypothesis, so max-consensus cannot discriminate and the refit collapses
      back to least squares — 77 % of the true rotation recovered. At 0.003 the
      same data yields 99 %. Raise it only for sequences with much larger
      inter-frame motion.
    min_inliers, min_inlier_ratio :
      Minimum absolute / relative consensus required to accept.
    seed : int
      RNG seed; fixed so runs are reproducible.

    Returns
    ----------
    (transform, info) — same contract as `match_ransac`.
    """
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
    best_inliers = None
    for _ in range(iterations):
        idx = rng.choice(n, size=3, replace=False)
        a, b = p[idx], p_prime[idx]
        # skip near-collinear samples: they leave the rotation underdetermined
        if (np.linalg.norm(np.cross(a[1] - a[0], a[2] - a[0])) < 1e-9 or
                np.linalg.norm(np.cross(b[1] - b[0], b[2] - b[0])) < 1e-9):
            continue
        try:
            T_h = _rigid_fit(a, b)
        except Exception:
            continue
        inliers = _residuals(T_h, p, p_prime) < inlier_dist
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    if best_inliers is None:
        return None, {"n": n, "rmse": float("nan"), "n_inliers": 0,
                      "reason": "no_hypothesis"}

    n_in = int(best_inliers.sum())
    if n_in < max(min_points, min_inliers) or n_in < min_inlier_ratio * n:
        return None, {"n": n, "rmse": float("nan"), "n_inliers": n_in,
                      "reason": "too_few_inliers"}

    # refit on the consensus set only, so the result is outlier-free
    T = _rigid_fit(p[best_inliers], p_prime[best_inliers])
    rmse = _trimmed_error(_residuals(T, p[best_inliers], p_prime[best_inliers]))
    if rmse < tol:
        return T, {"n": n, "rmse": rmse, "n_inliers": n_in, "reason": "ok_ransac"}
    return None, {"n": n, "rmse": rmse, "n_inliers": n_in, "reason": "rmse_too_high"}


def rigid_transform_3D(A, B):
    """
    Estimate a rigid transform between 2 set of points of equal length
    through singular value decomposition(svd), return a rotation and a 
    transformation matrix

    Parameters
    ----------
    A : (n,3) float
      The source 3d pointcloud as a numpy.ndarray
    B : (n,3) float
      The target 3d pointcloud as a numpy.ndarray

    Returns
    ----------
    R: (3,3) float
      A rigid rotation matrix
    t: (3) float
      A translation vector
 
    """

    assert len(A) == len(B)
    A=  np.asmatrix(A)
    B=  np.asmatrix(B)
    N = A.shape[0]; 

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))
    H = AA.T * BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T * U.T

    # reflection case
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T * U.T

    t = -R*centroid_A.T + centroid_B.T

    return (R, t)
