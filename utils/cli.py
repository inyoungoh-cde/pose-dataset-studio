"""
cli.py
------

Shared command-line helpers for the numbered pipeline scripts.

Every pipeline step accepts the same base arguments:

    python <step>.py [dataset] [--data-root .]

`dataset` may be a folder name next to the scripts, or a path (relative or
absolute) to the sequence folder anywhere on disk. The default data root is
the current directory deliberately: this toolchain never assumes, creates or
moves anything into a `custom/` layer — that name belongs to singleshotpose's
data root, and having two of them is what makes every path ambiguous. Where
the sequence physically lives is the user's choice; the prefix written into
the generated cfg is a separate knob (`5_create_config2split.py
--sspose-root`). Pass `--data-root` if you do keep all sequences under one
parent folder.

If `dataset` is omitted, the data root is scanned and the available
sequence folders are offered as an interactive numbered menu, so the
scripts can also be driven without memorizing folder names.
"""
import argparse
import os
import sys

# Which pipeline step produces each prerequisite (used in error hints).
PRODUCED_BY = {
    "JPEGImages": "1_record_azurekinect.py (or copy your own RGB frames in, see README)",
    "depth": "1_record_azurekinect.py (aligned 16-bit depth PNGs)",
    "intrinsics.json": "1_record_azurekinect.py (or write it manually, see README)",
    "transforms.npy": "2_compute_gt_poses.py",
    "registeredScene.ply": "3_register_scene.py",
    "objmask": "3a_segment_object_masks.py (text- or click-prompted object masks)",
    "object.ply": "3b_segment_object_cloud.py, or manual segmentation of registeredScene.ply "
                  "in CloudCompare/MeshLab (delete everything except the object, save as "
                  "ASCII PLY)",
    "clean_sfm.ply": "manual segmentation of registeredScene.ply in CloudCompare/MeshLab "
                     "(the older name for object.ply)",
    "labels": "4_create_label_files.py",
    "mask": "4_create_label_files.py",
    "transforms": "4_create_label_files.py",
    "test_mesh_after.ply": "4_create_label_files.py (older releases; now <dataset>.ply)",
}


def build_parser(description, dataset_help="sequence folder name under the data root"):
    """ArgumentParser pre-loaded with the common dataset/--data-root arguments."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataset", nargs="?", default=None,
                        help=dataset_help + " (omit to choose interactively)")
    parser.add_argument("--data-root", default=".",
                        help="parent folder to look the dataset name up in. Defaults to the "
                             "current directory -- sequence folders are taken where they are, "
                             "never assumed to sit under a 'custom/' layer. Ignored when "
                             "'dataset' is given as a path")
    return parser


def normalize_root(data_root):
    root = data_root.replace("\\", "/")
    if not root.endswith("/"):
        root += "/"
    return root


# A folder is offered in the interactive menu only if it looks like a capture
# sequence. Without this the menu would list utils/, config/, __pycache__/ and
# every other sibling folder, now that the data root defaults to '.'.
SEQUENCE_MARKERS = ("JPEGImages", "intrinsics.json", "transforms.npy")


def list_datasets(root):
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d))
                  and any(os.path.exists(os.path.join(root, d, m))
                          for m in SEQUENCE_MARKERS))


def choose_dataset(root):
    """Interactive numbered menu over the sequence folders under root."""
    candidates = list_datasets(root)
    if not candidates:
        sys.exit("No sequence folders found under '%s' (a sequence folder is one containing "
                 "%s).\nRun 1_record_azurekinect.py first, or pass the sequence folder as a "
                 "path, or point --data-root at the parent that holds them."
                 % (root, " / ".join(SEQUENCE_MARKERS)))
    print("Sequence folders under '%s':" % root)
    for i, name in enumerate(candidates):
        print("  [%d] %s" % (i, name))
    while True:
        try:
            sel = input("Select dataset (number or name, Enter to abort): ").strip()
        except EOFError:
            sys.exit("Aborted (no interactive input available -- pass the dataset argument).")
        if not sel:
            sys.exit("Aborted.")
        if sel in candidates:
            return sel
        if sel.isdigit() and int(sel) < len(candidates):
            return candidates[int(sel)]
        print("Invalid selection: %r" % sel)


def resolve_dataset(args, require=(), must_exist=True):
    """Validate the parsed args and return (dataset_name, 'parent/dataset/').

    The dataset argument may be either a bare sequence name (looked up under
    --data-root, which defaults to the current directory) or a direct path to
    the sequence folder -- relative or absolute -- in which case --data-root is
    ignored. Nothing is ever created or moved either way. The returned path
    always equals
    '<parent>/<dataset_name>/', so callers may reconstruct the parent with
    path[:-(len(dataset_name)+1)].

    require : iterable of str
        Files/folders that must already exist inside the sequence folder
        (e.g. 'transforms.npy'). Each miss aborts with a hint naming the
        pipeline step that produces it.
    must_exist : bool
        False for step 1, which creates the folder.
    """
    root = normalize_root(args.data_root)
    if root == "./":
        # The default root. Collapsing it keeps the printed paths and the
        # "not found" message free of a redundant './' that would also make
        # the two lookup candidates below look like two different places.
        root = ""
    dataset = args.dataset
    if dataset is None:
        if must_exist:
            dataset = choose_dataset(root or ".")
        else:
            try:
                dataset = input("New sequence folder name: ").strip()
            except EOFError:
                sys.exit("Aborted (no interactive input available -- pass the dataset argument).")
            if not dataset:
                sys.exit("A sequence folder name is required.")

    dataset = dataset.replace("\\", "/").rstrip("/")
    under_root = root + dataset + "/"
    direct = dataset + "/"

    if must_exist:
        if os.path.isdir(under_root):
            path = under_root
        elif os.path.isdir(direct):
            # dataset given as a direct path -- use it as-is
            path = direct
        else:
            available = ", ".join(list_datasets(root or ".")) or "(none)"
            tried = "'%s'" % under_root if under_root == direct \
                else "neither '%s' nor '%s'" % (under_root, direct)
            sys.exit("Sequence folder not found: %s.\n"
                     "Sequence folders under '%s': %s\n"
                     "Pass the folder as a path, or point --data-root at the parent that "
                     "holds it." % (tried, root or ".", available))
    else:
        # step-1 style: a bare name is created under the data root; anything
        # that looks like a path ('./x', 'E:/x', 'a/b') is created as given
        path = direct if "/" in dataset else under_root

    dataset = dataset.rsplit("/", 1)[-1]
    missing = [r for r in require if not os.path.exists(path + r)]
    if missing:
        hints = "\n".join("  %s  <- produced by %s" % (m, PRODUCED_BY.get(m, "?"))
                          for m in missing)
        sys.exit("'%s' is missing required inputs:\n%s" % (path, hints))
    return dataset, path
