"""
5_create_config2split.py
------------------------

Finish a labeled sequence: split it into train/test lists, measure the object
diameter, and write the singleshotpose `.data` config. Everything (mesh, diam,
intrinsics, image size, paths) is derived from the sequence folder itself.

    python 5_create_config2split.py mycup
    python 5_create_config2split.py mycup --split eval
    python 5_create_config2split.py mycup --split random --train-frac 0.3
"""

import glob
import json
import os
import sys

import cv2
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

from utils.cli import build_parser, resolve_dataset

# singleshotpose's read_data_cfg() does `key, value = line.split('=')` on every
# non-empty line, so the .data file must contain no comments.
CFG_DEFAULTED_KEYS = ("gpus", "num_workers")


def object_diameter(mesh_path):
    """Max distance between any two vertices -- the `diam` the ADD metric uses."""
    import trimesh
    pts = np.asarray(trimesh.load(mesh_path).vertices)
    if len(pts) < 2:
        sys.exit("'%s' has fewer than 2 vertices." % mesh_path)
    if len(pts) > 3:
        try:
            pts = pts[ConvexHull(pts).vertices]   # hull first: same answer, far cheaper
        except Exception:
            pass
    return float(pdist(pts).max())


def find_mesh(path, dataset, explicit=None):
    """The object mesh step 4 wrote, without asking the user for a filename."""
    if explicit:
        if not os.path.exists(path + explicit):
            sys.exit("'%s' not found in %s." % (explicit, path))
        return explicit
    for candidate in (dataset + ".ply", "test_mesh_after.ply"):
        if os.path.exists(path + candidate):
            return candidate
    sys.exit("No object mesh in %s (looked for %s.ply and test_mesh_after.ply) -- "
             "run 4_create_label_files.py first." % (path, dataset))


def labeled_frames(path):
    """Frame ids that actually have a mask, i.e. that step 4 labeled."""
    ids = [int(f[:-4]) for f in os.listdir(path + "mask") if f.endswith(".png")]
    if not ids:
        sys.exit("No masks in %smask -- run 4_create_label_files.py first." % path)
    return sorted(ids)


def choose_split(ids, mode, train_every, train_frac, seed):
    """(train_ids, test_ids) for the requested strategy."""
    if mode == "eval":
        return [], list(ids)
    if mode == "interleave":
        # every Nth frame of the sequence (the baseline's behaviour)
        train = [f for k, f in enumerate(ids) if k % train_every == 0]
    elif mode == "random":
        rng = np.random.default_rng(seed)
        n = max(1, int(round(len(ids) * train_frac)))
        train = sorted(np.array(ids)[rng.choice(len(ids), size=n, replace=False)].tolist())
    else:
        sys.exit("unknown --split %r" % mode)
    train_set = set(train)
    return train, [f for f in ids if f not in train_set]


def image_size(path):
    """Actual image dimensions, read from a real frame rather than trusted."""
    jpgs = sorted(glob.glob(path + "JPEGImages/*.jpg"))
    if not jpgs:
        sys.exit("No JPEGs in %sJPEGImages." % path)
    img = cv2.imread(jpgs[0])
    if img is None:
        sys.exit("Could not read %s." % jpgs[0])
    return img.shape[1], img.shape[0]


if __name__ == "__main__":
    parser = build_parser("Split a labeled sequence and write its singleshotpose .data cfg")
    parser.add_argument("--split", choices=["interleave", "eval", "random"],
                        default="interleave",
                        help="interleave: every Nth labeled frame trains (--train-every); "
                             "eval: no training frames, everything goes to test (for building "
                             "a pure evaluation sequence); random: a random --train-frac share")
    parser.add_argument("--train-every", type=int, default=5,
                        help="interleave stride: 5 means every 5th frame trains (20 %%)")
    parser.add_argument("--train-frac", type=float, default=0.2,
                        help="random split: fraction of frames used for training")
    parser.add_argument("--seed", type=int, default=0,
                        help="random split: RNG seed, fixed so the split is reproducible")
    parser.add_argument("--sspose-root", default="custom/",
                        help="path prefix written INTO the lists and the cfg. It must be what "
                             "singleshotpose resolves against from its own working directory, "
                             "which is independent of where this sequence sits on disk")
    parser.add_argument("--cfg-dir", default=None,
                        help="where to write <name>.data (default: the sequence folder itself, "
                             "so that folder is a self-contained deliverable you can move into "
                             "singleshotpose's data root as one unit)")
    parser.add_argument("--no-cfg", action="store_true",
                        help="only write the split lists, skip the .data file")
    parser.add_argument("--name", default=None,
                        help="value of the cfg 'name' key and the backup folder "
                             "(default: the dataset folder name)")
    parser.add_argument("--backup", default=None,
                        help="cfg 'backup' path for weights (default: backup/<name>)")
    parser.add_argument("--mesh", default=None,
                        help="object mesh filename inside the sequence folder "
                             "(default: <dataset>.ply, else test_mesh_after.ply)")
    parser.add_argument("--gpus", default="0", help="cfg 'gpus' value")
    args = parser.parse_args()

    dataset, path = resolve_dataset(args, require=("intrinsics.json", "JPEGImages",
                                                  "mask", "labels"))
    name = args.name or dataset
    root = args.sspose_root.replace("\\", "/")
    if root and not root.endswith("/"):
        root += "/"

    # ---- geometry ---------------------------------------------------------
    mesh_file = find_mesh(path, dataset, args.mesh)
    diam = object_diameter(path + mesh_file)

    # ---- intrinsics, and the image size that the labels were normalized by --
    with open(path + "intrinsics.json") as f:
        ci = json.load(f)
    width, height = image_size(path)
    if int(ci.get("width", width)) != width or int(ci.get("height", height)) != height:
        print("WARNING: intrinsics.json says %sx%s but the JPEGs are %dx%d. Step 4 normalized "
              "the\n  labels by the intrinsics values, so if those are the wrong ones the labels "
              "are\n  skewed -- regenerate them after fixing intrinsics.json."
              % (ci.get("width"), ci.get("height"), width, height))

    # ---- split -----------------------------------------------------------
    ids = labeled_frames(path)
    train_ids, test_ids = choose_split(ids, args.split, args.train_every,
                                       args.train_frac, args.seed)

    def write_list(fname, frames):
        with open(path + fname, "w") as fp:
            for i in frames:
                fp.write("%s%s/JPEGImages/%06d.jpg\n" % (root, name, i))

    write_list("train.txt", train_ids)
    write_list("test.txt", test_ids)
    with open(path + "training_range.txt", "w") as fp:
        for i in train_ids:
            fp.write("%d\n" % i)

    print("Split '%s': %d labeled frames -> %d train / %d test"
          % (args.split, len(ids), len(train_ids), len(test_ids)))
    print("  %strain.txt, test.txt, training_range.txt" % path)

    # ---- cfg -------------------------------------------------------------
    cfg = [("train", "%s%s/train.txt" % (root, name)),
           ("valid", "%s%s/test.txt" % (root, name)),
           ("backup", args.backup or ("backup/" + name)),
           ("mesh", "%s%s/%s" % (root, name, mesh_file)),
           ("tr_range", "%s%s/training_range.txt" % (root, name)),
           ("name", name),
           ("diam", "%.6f" % diam),
           ("gpus", args.gpus),
           ("width", str(width)),
           ("height", str(height)),
           ("fx", str(ci["fx"])),
           ("fy", str(ci["fy"])),
           ("u0", str(ci["ppx"])),
           ("v0", str(ci["ppy"]))]

    print("\nObject mesh %s, diameter %.6f m" % (mesh_file, diam))
    for k, v in cfg:
        print("  %-9s= %s" % (k, v))

    if args.no_cfg:
        print("\n--no-cfg given: .data file not written.")
    else:
        cfg_dir = (args.cfg_dir.replace("\\", "/").rstrip("/") if args.cfg_dir
                   else path.rstrip("/"))
        if not os.path.isdir(cfg_dir):
            os.makedirs(cfg_dir)
        cfg_path = "%s/%s.data" % (cfg_dir, name)
        with open(cfg_path, "w") as fp:
            for k, v in cfg:
                fp.write("%-9s= %s\n" % (k, v))
        print("\nWrote %s" % cfg_path)
        if not train_ids:
            print("  (train.txt is empty -- this is an evaluation-only sequence, so use this "
                  "cfg\n   with 3_eval.py, not with the training scripts.)")

        # cfg paths use the --sspose-root prefix: they describe where the folder
        # will live under singleshotpose, not where it sits now
        print("\nThe sequence folder is now self-contained (frames, labels, masks, mesh,")
        print("splits and this cfg). To train, move the whole folder to singleshotpose's")
        print("data root and point --datacfg at the cfg that travelled with it:")
        print("    copy  %s" % os.path.abspath(path.rstrip("/")))
        print("      to  <singleshotpose>/%s%s" % (root, name))
        print("    cd <singleshotpose>")
        print("    python 1_train_baseline.py --datacfg %s%s/%s.data --modelcfg cfg/yolo-pose.cfg"
              % (root, name, name))
        if root != "custom/":
            print("  (the paths inside the cfg start with '%s' because of --sspose-root; they "
                  "resolve\n   once the folder sits at that location relative to "
                  "singleshotpose's working directory)" % root)

        print("\nNext: python 6_inspect_labels.py %s   # visual QA of the labels"
              % (args.dataset or dataset))
