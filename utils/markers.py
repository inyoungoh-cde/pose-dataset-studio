"""
utils/markers.py
----------------

ArUco detection and the marker-board region, shared by the scripts that need to
know where the board is: `3_register_scene.py` (which crops the merge to it) and
the marker-board crop in step 3 (which uses the same region).

Both used to carry their own copy; a crop that differs between the tool you
diagnose with and the tool you build with is worse than no tool.
"""

import cv2
import cv2.aruco as aruco
import numpy as np

# cv2.aruco API changed in OpenCV 4.7 (Dictionary_get -> getPredefinedDictionary).
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


def marker_region_world(path, camera_intrinsics, frame_id=0, depth_tol=0.05):
     """(centre, radius) of the marker board, in world = frame-0 camera coordinates.

     Used to discard everything that is not the object or its marker board.
     Needed because the merge re-poses the WHOLE scene by the board's motion: in
     a turntable capture the static room is counter-rotated and sweeps into
     surfaces of revolution -- concentric rings and a bowl-shaped shell around
     the object. Surfaces perpendicular to the rotation axis (table top, floor)
     map onto themselves under that rotation, so they confirm each other and
     survive the vote filter no matter how it is tuned.
     """
     img = cv2.imread(path + 'JPEGImages/' + str('%06d' % frame_id) + '.jpg')
     depth = cv2.imread(path + 'depth/' + str('%06d' % frame_id) + '.png',
                        cv2.IMREAD_UNCHANGED)
     if img is None or depth is None:
          return None, None
     corners, ids, _ = detect_markers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
     if ids is None or not len(ids):
          return None, None
     fx = float(camera_intrinsics['fx'])
     fy = float(camera_intrinsics['fy'])
     ppx = float(camera_intrinsics['ppx'])
     ppy = float(camera_intrinsics['ppy'])
     scale = float(camera_intrinsics['depth_scale'])
     pts = []
     for c in corners:
          # Reject corners whose depth disagrees with their own marker's median:
          # a corner landing on the marker border reads the background instead,
          # and a single such reading (measured up to 3.6 m against a 0.6 m
          # board) would blow the crop radius up to metres.
          uv = [(int(p[0]), int(p[1])) for p in c[0]]
          zs = np.array([depth[v, u] * scale for u, v in uv])
          valid = zs > 0
          if np.count_nonzero(valid) >= 2:
               valid &= np.abs(zs - np.median(zs[valid])) <= depth_tol
          for k in range(len(uv)):
               if valid[k]:
                    u, v = uv[k]
                    z = zs[k]
                    pts.append(((u - ppx) / fx * z, (v - ppy) / fy * z, z))
     if len(pts) < 4:
          return None, None
     pts = np.array(pts)
     centre = np.median(pts, axis=0)
     dists = np.linalg.norm(pts - centre, axis=1)
     # 99th percentile rather than max: robust to a corner that still slipped through
     return centre, float(np.percentile(dists, 99))
