"""
6_inspect_labels.py
-------------------

Visualize the generated masks and 3D bounding-box labels for a visual QA pass.

    python 6_inspect_labels.py mycup
    python 6_inspect_labels.py mycup --delay 0        # step frame-by-frame with any key
    python 6_inspect_labels.py mycup --save-dir qa/   # also write the overlays to disk
"""
import glob
import sys
import cv2
import os
from utils.cli import build_parser, resolve_dataset


def visualize(path, delay=30, save_dir=None):
    resultIDs = []

    for file in os.listdir(path + "mask"):
        if file.endswith(".png"):
            resultIDs.append(int(file[:-4]))

    resultIDs.sort()

    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for id in resultIDs:
        filename_img = path + 'JPEGImages/' + str("%06d" % id) + '.jpg'
        img = cv2.imread(filename_img)
        if id < 10000:
            filename_result = path + 'mask/' + str("%04d" % id) + '.png'
        else:
            filename_result = path + 'mask/' + str("%06d" % id) + '.png'
        overlay = cv2.imread(filename_result)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        labelfile = path + 'labels/' + str("%06d" % id) + '.txt'
        if os.path.exists(labelfile):
            with open(labelfile, 'r') as fp:
                lines = fp.readlines()
                for line in lines:
                    info = line.split()
            info = [float(i) for i in info]
            width, length = img.shape[:2]

            center= (int(info[1] * length), int(info[2] * width))
            one = (int(info[3] * length), int(info[4] * width))
            two = (int(info[5] * length), int(info[6] * width))
            three = (int(info[7] * length), int(info[8] * width))
            four = (int(info[9] * length), int(info[10] * width))
            five = (int(info[11] * length), int(info[12] * width))
            six = (int(info[13] * length), int(info[14] * width))
            seven = (int(info[15] * length), int(info[16] * width))
            eight = (int(info[17] * length), int(info[18] * width))
            cv2.line(img, center, center, (255, 0, 255), 3)
            cv2.line(img, one, two, (255, 0, 0), 3)
            cv2.line(img, one, three, (255, 0, 0), 3)
            cv2.line(img, two, four, (255, 0, 0), 3)
            cv2.line(img, three, four, (255, 0, 0), 3)
            cv2.line(img, one, five, (255, 0, 0), 3)
            cv2.line(img, three, seven, (255, 0, 0), 3)
            cv2.line(img, five, seven, (255, 0, 0), 3)
            cv2.line(img, two, six, (255, 0, 0), 3)
            cv2.line(img, four, eight, (255, 0, 0), 3)
            cv2.line(img, six, eight, (255, 0, 0), 3)
            cv2.line(img, five, six, (255, 0, 0), 3)
            cv2.line(img, seven, eight, (255, 0, 0), 3)

        cv2.imshow("Masked", img)
        if save_dir:
            cv2.imwrite(os.path.join(save_dir, "{:06d}.jpg".format(id)), img)
        k = 0xFF & cv2.waitKey(delay)
        if k == ord('q'):
            break


if __name__ == "__main__":
    parser = build_parser("Visual QA: overlay mask + 3D bounding box on every frame")
    parser.add_argument("--delay", type=int, default=30,
                        help="ms between frames (0 = wait for a key press each frame); q quits")
    parser.add_argument("--save-dir", default=None,
                        help="optional folder to save the overlay images into")
    args = parser.parse_args()
    dataset, path = resolve_dataset(args, require=("JPEGImages", "mask", "labels"))
    print(path)
    visualize(path, delay=args.delay, save_dir=args.save_dir)
    cv2.destroyAllWindows()
