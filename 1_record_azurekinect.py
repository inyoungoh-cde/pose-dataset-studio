"""
1_record_azurekinect.py
-----------------------

Record an RGB-D sequence with an Azure Kinect (pykinect_azure / K4A).

Writes into <data-root>/<dataset>/:
    JPEGImages/%06d.jpg   720p BGRA color frames
    depth/%06d.png        16-bit depth, transformed into the color frame
    intrinsics.json       color-camera intrinsics (+ depth_scale)

Keys: space = pause/resume, q = stop recording.

    python 1_record_azurekinect.py mycup                       # creates ./mycup/
    python 1_record_azurekinect.py mycup --countdown 10 --data-root captures/
"""
import sys
import cv2
import os
import time
import png
import numpy as np
import json
from utils.cli import build_parser, resolve_dataset
sys.path.insert(1, '../')
import pykinect_azure as pykinect

def make_directories(folder):
    if not os.path.exists(folder+"JPEGImages/"):
        os.makedirs(folder+"JPEGImages/")
    if not os.path.exists(folder+"depth/"):
        os.makedirs(folder+"depth/")

if __name__ == "__main__":
    parser = build_parser(__doc__.strip().splitlines()[3],
                          dataset_help="sequence folder to create under the data root")
    parser.add_argument("--countdown", type=int, default=5,
                        help="seconds of countdown before recording starts")
    parser.add_argument("--depth-scale", type=float, default=0.001,
                        help="meters per depth unit written to intrinsics.json "
                             "(K4A depth is in millimeters -> 0.001)")
    args = parser.parse_args()
    dataset, folder = resolve_dataset(args, must_exist=False)

    FileName = 0
    COUNTDOWN = args.countdown
    make_directories(folder)
    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED

    frame_count = 0
    # Start device
    device = pykinect.start_device(config=device_config)
    intr = device.calibration.color_params
    intrinsics_written = False
    T_start = time.time()
    while True:
        # Get capture
        capture = device.update()

        # Get the color image from the capture
        ret, color_image = capture.get_color_image()

        if not ret:
            continue

        # Get the depth
        ret, transformed_depth_image = capture.get_transformed_depth_image()

        if not ret:
            continue

        d = np.asanyarray(transformed_depth_image)
        c = np.asanyarray(color_image)

        if not intrinsics_written:
            # width/height taken from the actual captured frame
            camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                                 'ppx': intr.cx, 'ppy': intr.cy,
                                 'height': c.shape[0], 'width': c.shape[1],
                                 'depth_scale': args.depth_scale}
            with open(folder+'intrinsics.json', 'w') as fp:
                json.dump(camera_parameters, fp)
            intrinsics_written = True

        if time.time() - T_start > COUNTDOWN:
            filecad = folder + "JPEGImages/"+str("%06d" % FileName)+".jpg"
            filedepth = folder + "depth/"+str("%06d" % FileName)+".png"
            cv2.imwrite(filecad, c)

            with open(filedepth, 'wb') as f:
                writer = png.Writer(width=d.shape[1], height=d.shape[0],
                                    bitdepth=16, greyscale=True)
                zgray2list = d.tolist()
                writer.write(f, zgray2list)

            FileName += 1

        if time.time() - T_start < COUNTDOWN:
            cv2.putText(c, str(COUNTDOWN - int(time.time() - T_start)), (240, 320),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 4,
                        (0, 0, 255), 2, cv2.LINE_AA)

        if time.time() - T_start > COUNTDOWN:
            cv2.putText(c, str(frame_count), (10, 90), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 4,
                        (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('COLOR IMAGE', c)

        if time.time() - T_start > COUNTDOWN: frame_count+=1

        # space bar = pause (any key resumes; q while paused quits), q = quit
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            key = cv2.waitKey() & 0xFF
        if key == ord('q'):
            break

    # Release everything if job is finished
    cv2.destroyAllWindows()
    print("Saved %d frames to %s" % (FileName, folder))
