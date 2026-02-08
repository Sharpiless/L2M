
import argparse
import os
import os.path as osp
import re
import json
import cv2
import numpy as np
import time
import warnings
import logging
from collections import defaultdict, OrderedDict
from tqdm import tqdm
from collections import Counter
import torch
torch.backends.cudnn.enabled = False

from load_model import load_model, choose_method_arguments, add_method_arguments
from src.utils.metrics import error_auc
from src.utils.plotting import make_matching_figure2



def load_cp_txt(cp_path: str):
    """
    Robust CP TXT loader.
    Supports comma-separated or whitespace-separated.
    Each row: mx,my,fx,fy  (or mx my fx fy)
    """
    # read a small sample to guess delimiter
    with open(cp_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip() and not ln.strip().startswith("#")]

    if len(lines) == 0:
        raise ValueError("Empty CP file")

    sample = lines[0]
    # guess delimiter
    if "," in sample:
        delim = ","
    elif "\t" in sample:
        delim = "\t"
    elif ";" in sample:
        delim = ";"
    else:
        delim = None  # whitespace

    try:
        if delim is None:
            pts = np.loadtxt(cp_path, dtype=np.float64)
        else:
            pts = np.loadtxt(cp_path, dtype=np.float64, delimiter=delim)
    except Exception as e:
        raise ValueError(f"np.loadtxt failed (delim={repr(delim)}): {e}")

    if pts.ndim == 1:
        pts = pts.reshape(1, -1)

    if pts.ndim != 2 or pts.shape[1] < 4:
        raise ValueError(f"Bad CP shape={pts.shape}, need Nx4+")

    pts = pts[:, :4]  # ensure Nx4
    if not np.isfinite(pts).all():
        raise ValueError("CP contains non-finite values")

    ref_pts = pts[:, 0:2]
    mov_pts = pts[:, 2:4]
    return ref_pts, mov_pts



def compute_gt_homography_from_cp_debug(ref_pts, mov_pts):
    n = min(len(ref_pts), len(mov_pts))
    if n < 4:
        raise ValueError(f"Not enough control points: {n} (<4)")
    ref = ref_pts[:n].astype(np.float64)
    mov = mov_pts[:n].astype(np.float64)

    H, _ = cv2.findHomography(mov, ref, method=0)
    if H is None:
        raise ValueError("cv2.findHomography returned None")
    if not np.isfinite(H).all():
        raise ValueError("Homography contains non-finite values")
    if abs(H[2, 2]) < 1e-12:
        raise ValueError("Homography H[2,2] is ~0 (cannot normalize)")

    H = H / H[2, 2]
    return H


def load_flori21_pairs_raw_txt(flori_root: str, debug_print=False, strict=True):
    """
    Debug version:
      - debug_print=True: print detailed progress
      - strict=True: require montage/fa/cp all match naming; else skip
    """
    reasons = Counter()

    def dprint(msg):
        if debug_print:
            print(msg)

    flori_root = osp.abspath(flori_root)
    dprint(f"[DEBUG] flori_root = {flori_root}")

    if not osp.isdir(flori_root):
        dprint(f"[ERROR] flori_root is NOT a directory.")
        dprint(f"        Exists? {osp.exists(flori_root)}")
        if osp.exists(flori_root):
            dprint(f"        It exists but isfile={osp.isfile(flori_root)}")
        raise FileNotFoundError(f"Missing directory: {flori_root}")

    # list direct children
    children = sorted(os.listdir(flori_root))
    dprint(f"[DEBUG] root children count={len(children)} sample={children[:10]}")

    subject_dirs = sorted([
        osp.join(flori_root, d) for d in children
        if osp.isdir(osp.join(flori_root, d)) and d.lower().startswith("subject_")
    ])
    dprint(f"[DEBUG] found subject_dirs={len(subject_dirs)}")
    if len(subject_dirs) == 0:
        dprint("[HINT] 0 subjects found. Your flori_root should be the folder that CONTAINS Subject_1, Subject_2, ...")
        dprint("       Based on your tree, this should end with: .../FLoRI21_DataPort/data")
        return {"FLoRI21": []}

    pairs = []
    for subj_dir in subject_dirs:
        subj_name = osp.basename(subj_dir)
        msub = re.match(r"Subject_(\d+)$", subj_name, flags=re.IGNORECASE)
        if not msub:
            reasons["bad_subject_name"] += 1
            dprint(f"[WARN] subject dir name not match: {subj_name}")
            continue
        subj_id = int(msub.group(1))

        montage_img = osp.join(subj_dir, "Montage", f"Montage_Subject_{subj_id}.tif")
        fa_dir = osp.join(subj_dir, "FA")
        cp_dir = osp.join(subj_dir, "ControlPoints")

        dprint(f"\n[DEBUG] === {subj_name} ===")
        dprint(f"[DEBUG] montage_img: {montage_img} exists={osp.exists(montage_img)}")
        dprint(f"[DEBUG] fa_dir: {fa_dir} exists={osp.isdir(fa_dir)}")
        dprint(f"[DEBUG] cp_dir: {cp_dir} exists={osp.isdir(cp_dir)}")

        if not osp.exists(montage_img):
            reasons["missing_montage"] += 1
            if strict:
                dprint("[SKIP] missing montage image")
                continue

        if not osp.isdir(fa_dir):
            reasons["missing_fa_dir"] += 1
            dprint("[SKIP] missing FA dir")
            continue

        if not osp.isdir(cp_dir):
            reasons["missing_cp_dir"] += 1
            dprint("[SKIP] missing ControlPoints dir")
            continue

        cp_files = sorted([
            osp.join(cp_dir, f) for f in os.listdir(cp_dir)
            if f.lower().endswith(".txt")
        ])
        dprint(f"[DEBUG] cp_files found={len(cp_files)} sample={[osp.basename(x) for x in cp_files[:5]]}")

        if len(cp_files) == 0:
            reasons["no_cp_files"] += 1
            dprint("[SKIP] no cp txt files")
            continue

        for cp in cp_files:
            base = osp.basename(cp)
            m = re.match(r"ControlPoints_Montage_FA_(\d+)_Subject_(\d+)\.txt$", base, flags=re.IGNORECASE)
            if not m:
                reasons["cp_name_not_match_regex"] += 1
                dprint(f"[SKIP] cp filename not match regex: {base}")
                continue

            fa_id = int(m.group(1))
            subj_id2 = int(m.group(2))
            if subj_id2 != subj_id:
                reasons["cp_subject_mismatch"] += 1
                dprint(f"[SKIP] subject mismatch in cp: {base} (cp says {subj_id2}, folder says {subj_id})")
                continue

            raw_img = osp.join(fa_dir, f"Raw_FA_{fa_id}_Subject_{subj_id}.tif")
            if not osp.exists(raw_img):
                reasons["missing_fa_image"] += 1
                dprint(f"[SKIP] missing FA image: {osp.basename(raw_img)} (expected at {raw_img})")
                continue

            # load points
            try:
                ref_pts, mov_pts = load_cp_txt(cp)
                dprint(f"[DEBUG] {base}: loaded cp shape ref={ref_pts.shape} mov={mov_pts.shape}")
            except Exception as e:
                reasons["cp_load_failed"] += 1
                dprint(f"[SKIP] cp load failed: {base} reason={e}")
                continue

            # compute gt H
            try:
                H_gt = compute_gt_homography_from_cp_debug(ref_pts, mov_pts)
                dprint(f"[DEBUG] {base}: H_gt OK")
            except Exception as e:
                reasons["gt_h_failed"] += 1
                dprint(f"[SKIP] gt homography failed: {base} reason={e}")
                continue

            pair_id = f"Subject_{subj_id}_FA_{fa_id}"
            pairs.append({
                "im0": raw_img,
                "im1": montage_img,
                "H": H_gt,
                "pair_id": pair_id,
                "ref_pts": ref_pts,
                "mov_pts": mov_pts
            })

    dprint("\n[DEBUG] ===== Summary =====")
    dprint(f"[DEBUG] total loaded pairs = {len(pairs)}")
    if len(reasons) > 0:
        dprint("[DEBUG] skip/fail reasons:")
        for k, v in reasons.most_common():
            dprint(f"  - {k}: {v}")
    else:
        dprint("[DEBUG] no failures recorded")

    # Optional: sanity check expected 15
    if debug_print:
        dprint("[DEBUG] (info) expected total pairs from your tree = 15")

    return {"FLoRI21": pairs}


# =========================================================
# Metrics
# =========================================================

def order_corners(corners):
    rect = np.zeros((4, 2), dtype="float32")
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]
    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]
    return rect


def compute_mean_corner_distance(gt_H, pred_H, H_img, W_img):
    """
    Corner reprojection mean distance between gt_H and pred_H (both map im0->im1).
    Corners are from im0 coordinate system (matcher coordinate system).
    """
    corners = np.array([
        [0, 0, 1],
        [W_img - 1, 0, 1],
        [0, H_img - 1, 1],
        [W_img - 1, H_img - 1, 1],
    ], dtype=np.float64)

    gt_w = (corners @ gt_H.T)
    gt_w = gt_w[:, :2] / gt_w[:, 2:]

    pr_w = (corners @ pred_H.T)
    pr_w = pr_w[:, :2] / pr_w[:, 2:]

    gt_w = order_corners(gt_w.astype(np.float32))
    pr_w = order_corners(pr_w.astype(np.float32))
    return float(np.mean(np.linalg.norm(gt_w - pr_w, axis=1)))


def compute_controlpoint_mre(pred_H, mov_pts, ref_pts):
    """
    Mean registration error on GT control points for pred_H (im0->im1).
    mov_pts: in im0 (FA)
    ref_pts: in im1 (montage)
    """
    n = min(len(mov_pts), len(ref_pts))
    if n == 0:
        return np.inf
    mov = mov_pts[:n].astype(np.float64)
    ref = ref_pts[:n].astype(np.float64)

    mov_h = np.hstack([mov, np.ones((n, 1), dtype=np.float64)])
    proj = (pred_H @ mov_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    dist = np.linalg.norm(proj - ref, axis=1)
    return float(np.mean(dist))


def compute_match_correct_mask(gt_H, mkpts0, mkpts1, thr_px=5.0):
    """
    correctness mask for visualization based on symmetric reprojection error under gt_H.
    mkpts0 in im0, mkpts1 in im1
    """
    n = mkpts0.shape[0]
    if n == 0:
        return np.array([], dtype=bool)

    p0 = mkpts0.astype(np.float64)
    p1 = mkpts1.astype(np.float64)

    p0_h = np.hstack([p0, np.ones((n, 1))])
    p1_h = np.hstack([p1, np.ones((n, 1))])

    p1_proj = (gt_H @ p0_h.T).T
    p1_proj = p1_proj[:, :2] / p1_proj[:, 2:3]

    invH = np.linalg.inv(gt_H)
    p0_proj = (invH @ p1_h.T).T
    p0_proj = p0_proj[:, :2] / p0_proj[:, 2:3]

    e1 = np.linalg.norm(p1 - p1_proj, axis=1)
    e0 = np.linalg.norm(p0 - p0_proj, axis=1)
    mean_e = 0.5 * (e0 + e1)
    return mean_e < float(thr_px)


# =========================================================
# Plotting helper (FIRE-style)
# =========================================================

def save_matching_figure(path, img0, img1, mkpts0, mkpts1, correct_mask, text_lines, svg=False):
    """
    green/red by correct_mask
    """
    n = mkpts0.shape[0]
    cm = correct_mask.astype(float) if n > 0 else np.array([], dtype=float)
    precision = float(np.mean(cm)) if len(cm) > 0 else 0.0
    n_correct = int(np.sum(cm)) if len(cm) > 0 else 0

    color = np.zeros((n, 3), dtype=np.uint8)
    if n > 0:
        color[cm == 0] = (255, 0, 0)
        color[cm == 1] = (0, 255, 0)

    text = list(text_lines) + [f"Prec({100*precision:.1f}%): {n_correct}/{n}"]
    make_matching_figure2(img0, img1, mkpts0, mkpts1, color, text=text, path=path, dpi=150, svg=svg)


# =========================================================
# Eval
# =========================================================

def eval_flori21(
    matcher,
    scene_pairs,
    save_figs=False,
    figures_dir=None,
    method="method",
    print_out=False,
    debug=False,
    svg=False,
    gt_thr_px=5.0,
    auc_thresholds=(1, 3, 5, 7, 10, 15, 20),
):
    """
    Per pair:
      - matcher(im0, im1) -> mkpts0, mkpts1
      - pred_H = findHomography(mkpts0, mkpts1, RANSAC)
      - corner_mre = mean corner distance between pred_H and gt_H
      - cp_mre = mean registration error on GT control points (mov_pts->ref_pts) using pred_H
    """
    results = {}

    for scene_name, pairs in scene_pairs.items():
        if save_figs:
            os.makedirs(figures_dir, exist_ok=True)
            scene_dir = osp.join(figures_dir, scene_name)
            os.makedirs(scene_dir, exist_ok=True)

        statis = defaultdict(list)
        logging.info(f"Start evaluation on {scene_name}, num_pairs={len(pairs)}")

        for i, pair in tqdm(list(enumerate(pairs)), total=len(pairs), smoothing=0.1):
            if debug and i > 10:
                break

            im0 = pair["im0"]          # moving FA
            im1 = pair["im1"]          # montage
            gt_H = pair["H"]           # im0 -> im1
            ref_pts = pair["ref_pts"]  # montage
            mov_pts = pair["mov_pts"]  # FA
            pid = pair.get("pair_id", f"pair_{i}")

            # run matcher
            match_res = matcher(im0, im1)
            mkpts0 = match_res["mkpts0"]
            mkpts1 = match_res["mkpts1"]
            img0 = match_res["img0"]
            img1 = match_res["img1"]

            # image shape for corner computation should match matcher keypoints coords
            if img0.ndim == 2:
                H_img, W_img = img0.shape
                img0_bgr = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
                img1_bgr = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            else:
                H_img, W_img = img0.shape[:2]
                img0_bgr, img1_bgr = img0, img1

            pred_H = None
            corner_mre = np.inf
            cp_mre = np.inf

            try:
                if mkpts0.shape[0] >= 4:
                    pred_H, _ = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC)
                if pred_H is not None and np.isfinite(pred_H).all():
                    pred_H = pred_H / pred_H[2, 2]
                    corner_mre = compute_mean_corner_distance(gt_H, pred_H, H_img, W_img)
                    cp_mre = compute_controlpoint_mre(pred_H, mov_pts, ref_pts)
            except Exception as e:
                logging.info(f"[{pid}] exception: {e}")

            if (pred_H is None) or (not np.isfinite(corner_mre)) or (not np.isfinite(cp_mre)):
                statis["failed"].append(i)

            statis["corner_mre"].append(float(corner_mre))
            statis["cp_mre"].append(float(cp_mre))
            statis["n_matches"].append(int(mkpts0.shape[0]))

            if save_figs:
                correct_mask = compute_match_correct_mask(gt_H, mkpts0, mkpts1, thr_px=gt_thr_px) if mkpts0.shape[0] > 0 else np.array([], dtype=bool)
                fig_path = osp.join(scene_dir, f"{pid}_{method}.jpg")
                img0_rgb = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2RGB)
                img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
                txt = [
                    f"corner_mre: {corner_mre:.2f}px",
                    f"cp_mre: {cp_mre:.2f}px",
                    f"#M: {mkpts0.shape[0]}",
                ]
                save_matching_figure(fig_path, img0_rgb, img1_rgb, mkpts0, mkpts1, correct_mask, txt, svg=svg)

            if print_out:
                logging.info(f"[{pid}] #M={mkpts0.shape[0]} corner_mre={corner_mre:.2f} cp_mre={cp_mre:.2f}")

        # aggregate
        corner_arr = np.asarray(statis["corner_mre"], dtype=np.float64)
        cp_arr = np.asarray(statis["cp_mre"], dtype=np.float64)

        auc_cp = error_auc(cp_arr, list(auc_thresholds))

        results[scene_name] = {
            "num_pairs": int(len(pairs)),
            "num_failed": int(len(statis["failed"])),
            "mean_corner_mre": float(np.nanmean(corner_arr)),
            "median_corner_mre": float(np.nanmedian(corner_arr)),
            "mean_cp_mre": float(np.nanmean(cp_arr)),
            "median_cp_mre": float(np.nanmedian(cp_arr)),
            "auc_cp": {f"auc@{t}": float(auc_cp[f"auc@{t}"]) for t in auc_thresholds},
        }

        logging.info(f"[{scene_name}] done. failed={len(statis['failed'])}")
        logging.info(f"[{scene_name}] AUC cp: {auc_cp}")

    return results


# =========================================================
# Runner
# =========================================================

def test_flori21(
    flori_root,
    method="xoftr",
    exp_name="FLoRI21",
    ransac_thres=1.5,   # kept for interface consistency
    print_out=False,
    save_dir="./results_flori21_homo/",
    save_figs=False,
    debug=False,
    args=None,
):
    save_ = method if getattr(args, "ckpt", None) is None else osp.basename(args.ckpt).replace(".ckpt", "")
    path_ = osp.join(save_dir, method, save_)
    if getattr(args, "debug", False):
        path_ = osp.join(save_dir, method, save_, "debug")
    os.makedirs(path_, exist_ok=True)

    path = osp.join(path_, f"{exp_name}")
    exp_dir = path
    os.makedirs(exp_dir, exist_ok=True)

    results_file = osp.join(exp_dir, "results.json")
    logging.basicConfig(
        filename=results_file.replace(".json", ".log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    figures_dir = osp.join(exp_dir, "match_figures")
    if save_figs:
        os.makedirs(figures_dir, exist_ok=True)

    logging.info(f"args: {args}")

    # Load pairs (RAW TXT CP)
    scene_pairs = load_flori21_pairs_raw_txt(flori_root)

    # Load matcher
    matcher = load_model(method, args)

    thresholds = [int(x) for x in args.auc_thresholds.split(",")]
    scene_results = eval_flori21(
        matcher,
        scene_pairs,
        save_figs=save_figs,
        figures_dir=figures_dir,
        method=method,
        print_out=print_out,
        debug=debug,
        svg=getattr(args, "svg", False),
        gt_thr_px=getattr(args, "gt_thr_px", 5.0),
        auc_thresholds=thresholds,
    )

    results = OrderedDict({
        "method": method,
        "exp_name": exp_name,
        "ransac_thres": float(ransac_thres),
        "auc_thresholds": thresholds,
    })
    results.update({k: v for k, v in vars(args).items() if k not in results})
    results["scenes"] = scene_results

    logging.info(f"Results: {json.dumps(results, indent=2)}")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved results to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Benchmark FLoRI21 (RAW TXT CP) - Homography + CP MRE")
    choose_method_arguments(parser)

    parser.add_argument("--flori_root", type=str, default="/data6/liangyingping/tmp/dataset/FLoRI21_DataPort/data_512/",
                        help="Root containing Subject_*/")
    parser.add_argument("--exp_name", type=str, default="FLoRI21")
    parser.add_argument("--save_dir", type=str, default="./results_flori21_homo/")
    parser.add_argument("--ransac_thres", type=float, default=1.5)
    parser.add_argument("--print_out", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_figs", action="store_true")
    parser.add_argument("--svg", action="store_true")

    parser.add_argument("--gt_thr_px", type=float, default=5.0,
                        help="Threshold(px) for correctness coloring using GT H (only for visualization).")

    # For retina you may want larger thresholds, e.g. "10,20,30,40,50,75,100"
    parser.add_argument("--auc_thresholds", type=str, default="1,3,5,7,10,15,20",
                        help="Comma-separated thresholds for AUC computation.")

    args, remaining_args = parser.parse_known_args()
    add_method_arguments(parser, args.method)
    args = parser.parse_args()

    print(args)

    tt = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_flori21(
            flori_root=args.flori_root,
            method=args.method,
            exp_name=args.exp_name,
            ransac_thres=args.ransac_thres,
            print_out=args.print_out,
            save_dir=args.save_dir,
            save_figs=args.save_figs,
            debug=args.debug,
            args=args,
        )
    print(f"Elapsed time: {time.time() - tt:.2f}s")
