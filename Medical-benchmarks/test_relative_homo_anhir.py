import argparse
import os
import os.path as osp
import json
import time
import warnings
import logging
from collections import defaultdict, OrderedDict
from tqdm import tqdm

import cv2
import numpy as np
import pandas as pd
import torch
torch.backends.cudnn.enabled = False

from load_model import load_model, choose_method_arguments, add_method_arguments
from src.utils.metrics import error_auc
from src.utils.plotting import make_matching_figure2


# =========================================================
# IO
# =========================================================

def read_anhir_landmarks_csv(csv_path: str) -> np.ndarray:
    """
    ANHIR landmarks:
      ,X,Y
      0,11237.2,3608.8
    or sometimes id,x,y
    Return: (N,2) float64 array.
    """
    df = pd.read_csv(csv_path)
    cols = [c.lower() for c in df.columns]

    if "x" in cols and "y" in cols:
        x_col = df.columns[cols.index("x")]
        y_col = df.columns[cols.index("y")]
        pts = df[[x_col, y_col]].to_numpy(dtype=np.float64)
    else:
        # fallback: last two columns
        pts = df.iloc[:, -2:].to_numpy(dtype=np.float64)

    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Bad landmarks shape {pts.shape} in {csv_path}")
    if not np.isfinite(pts).all():
        raise ValueError(f"Non-finite landmarks in {csv_path}")
    return pts


def load_anhir_pairs_from_index(index_csv: str,
                               data_root: str,
                               split: str = "all",
                               debug_print: bool = False,
                               strict: bool = False):
    """
    Read dataset_medium.csv and build pairs list.
    split: "training" | "evaluation" | "all"
    Each pair:
      im0 = Source image (moving)
      im1 = Target image (fixed)
      mov_pts = Source landmarks
      ref_pts = Target landmarks
    Only keeps rows where all 4 files exist (unless strict=False then skip quietly).
    """
    reasons = defaultdict(int)

    def dprint(msg):
        if debug_print:
            print(msg)

    df = pd.read_csv(index_csv)

    required = ["Source image", "Source landmarks", "Target image", "Target landmarks", "status"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Index missing column: {c}. Columns={list(df.columns)}")

    pairs = []
    for idx, r in df.iterrows():
        status = str(r["status"]).strip().lower()

        if split != "all" and status != split.lower():
            continue

        im0 = osp.join(data_root, str(r["Source image"]))
        lm0 = osp.join(data_root, str(r["Source landmarks"]))
        im1 = osp.join(data_root, str(r["Target image"]))
        lm1 = osp.join(data_root, str(r["Target landmarks"]))

        ok = osp.exists(im0) and osp.exists(im1) and osp.exists(lm0) and osp.exists(lm1)
        if not ok:
            reasons["missing_files"] += 1
            if strict:
                dprint(f"[MISS] {idx}: {im0} {lm0} {im1} {lm1}")
            continue

        try:
            mov_pts = read_anhir_landmarks_csv(lm0)  # source
            ref_pts = read_anhir_landmarks_csv(lm1)  # target
        except Exception:
            reasons["bad_landmarks"] += 1
            continue

        if len(mov_pts) == 0 or len(ref_pts) == 0:
            reasons["empty_landmarks"] += 1
            continue

        n = min(len(mov_pts), len(ref_pts))
        if n < 3:
            reasons["too_few_landmarks"] += 1
            continue

        pair_id = f"row{idx}__{osp.splitext(osp.basename(im0))[0]}_to_{osp.splitext(osp.basename(im1))[0]}"
        scene_name = osp.normpath(str(r["Source image"])).split(os.sep)[0]  # e.g. breast_3, COAD_01

        pairs.append({
            "scene": scene_name,
            "im0": im0,
            "im1": im1,
            "mov_pts": mov_pts[:n],
            "ref_pts": ref_pts[:n],
            "pair_id": pair_id,
            "status": status
        })

    # group by scene
    scene_pairs = defaultdict(list)
    for p in pairs:
        scene_pairs[p["scene"]].append(p)

    dprint(f"[DEBUG] loaded pairs={len(pairs)} split={split}")
    if reasons:
        dprint("[DEBUG] skip reasons:")
        for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
            dprint(f"  - {k}: {v}")

    return dict(scene_pairs)


# =========================================================
# Metrics
# =========================================================

def apply_homography_to_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    pts: (N,2), returns (N,2)
    """
    n = pts.shape[0]
    pts_h = np.hstack([pts.astype(np.float64), np.ones((n, 1), dtype=np.float64)])
    proj = (H @ pts_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    return proj


def compute_tre_mre(pred_map_fn, mov_pts, ref_pts):
    """
    TRE mean registration error using a mapping function.
    pred_map_fn: function(pts)->pts'
    """
    n = min(len(mov_pts), len(ref_pts))
    if n == 0:
        return np.inf
    p = pred_map_fn(mov_pts[:n])
    dist = np.linalg.norm(p - ref_pts[:n], axis=1)
    return float(np.mean(dist))


def compute_match_correct_mask_homo(gt_ref_pts, gt_mov_pts, pred_H, thr_px=5.0):
    """
    For visualization: treat GT correspondences as mov_pts->ref_pts and
    compute if pred_H maps mov_pts close to ref_pts.
    """
    if pred_H is None:
        return np.zeros((0,), dtype=bool)
    n = min(len(gt_mov_pts), len(gt_ref_pts))
    if n == 0:
        return np.zeros((0,), dtype=bool)

    proj = apply_homography_to_points(pred_H, gt_mov_pts[:n])
    e = np.linalg.norm(proj - gt_ref_pts[:n], axis=1)
    return e < float(thr_px)


# =========================================================
# Plotting helper
# =========================================================

def save_matching_figure(path, img0, img1, mkpts0, mkpts1, correct_mask, text_lines, svg=False):
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

def eval_anhir(
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
    ransac_reproj_thr=3.0,
):
    """
    For each pair:
      - matcher(im0, im1) -> mkpts0, mkpts1
      - pred_H = findHomography(mkpts0, mkpts1, RANSAC)
      - TRE(MRE): mean || H(mov_landmarks) - ref_landmarks ||
    Note: This is a global-homography baseline for ANHIR-style landmarks.

    Returns:
      results[scene] + results["average"] (micro-average over all pairs)
    """
    results = {}

    # Collect ALL pairs for micro-average
    all_tre = []
    all_n_matches = []
    all_failed = 0

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

            im0 = pair["im0"]
            im1 = pair["im1"]
            ref_pts = pair["ref_pts"]  # target landmarks
            mov_pts = pair["mov_pts"]  # source landmarks
            pid = pair.get("pair_id", f"pair_{i}")

            match_res = matcher(im0, im1)
            mkpts0 = match_res["mkpts0"]
            mkpts1 = match_res["mkpts1"]
            img0 = match_res["img0"]
            img1 = match_res["img1"]

            # for plotting
            if img0.ndim == 2:
                img0_bgr = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
                img1_bgr = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            else:
                img0_bgr, img1_bgr = img0, img1

            pred_H = None
            tre_mre = np.inf

            try:
                if mkpts0.shape[0] >= 4:
                    pred_H, _ = cv2.findHomography(
                        mkpts0, mkpts1, cv2.RANSAC,
                        ransacReprojThreshold=ransac_reproj_thr
                    )
                if pred_H is not None and np.isfinite(pred_H).all() and abs(pred_H[2, 2]) > 1e-12:
                    pred_H = pred_H / pred_H[2, 2]
                    tre_mre = compute_tre_mre(
                        lambda pts: apply_homography_to_points(pred_H, pts),
                        mov_pts, ref_pts
                    )
            except Exception as e:
                logging.info(f"[{pid}] exception: {e}")

            failed = (pred_H is None) or (not np.isfinite(tre_mre))
            if failed:
                statis["failed"].append(i)

            statis["tre_mre"].append(float(tre_mre))
            statis["n_matches"].append(int(mkpts0.shape[0]))

            # global collect (micro-average)
            all_tre.append(float(tre_mre))
            all_n_matches.append(int(mkpts0.shape[0]))
            if failed:
                all_failed += 1

            if save_figs:
                # simplest: all green
                correct_mask = np.ones((mkpts0.shape[0],), dtype=bool)
                fig_path = osp.join(scene_dir, f"{pid}_{method}.jpg")
                img0_rgb = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2RGB)
                img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
                txt = [
                    f"TRE(MRE): {tre_mre:.2f}px",
                    f"#M: {mkpts0.shape[0]}",
                ]
                save_matching_figure(fig_path, img0_rgb, img1_rgb, mkpts0, mkpts1, correct_mask, txt, svg=svg)

            if print_out:
                logging.info(f"[{pid}] #M={mkpts0.shape[0]} TRE(MRE)={tre_mre:.2f}")

        tre_arr = np.asarray(statis["tre_mre"], dtype=np.float64)
        auc_tre = error_auc(tre_arr, list(auc_thresholds))

        results[scene_name] = {
            "num_pairs": int(len(pairs)),
            "num_failed": int(len(statis["failed"])),
            "mean_tre_mre": float(np.nanmean(tre_arr)),
            "median_tre_mre": float(np.nanmedian(tre_arr)),
            "auc_tre": {f"auc@{t}": float(auc_tre[f"auc@{t}"]) for t in auc_thresholds},
            "mean_n_matches": float(np.mean(statis["n_matches"])) if len(statis["n_matches"]) else 0.0,
            "median_n_matches": float(np.median(statis["n_matches"])) if len(statis["n_matches"]) else 0.0,
        }

        logging.info(f"[{scene_name}] done. failed={len(statis['failed'])}")
        logging.info(f"[{scene_name}] AUC TRE: {auc_tre}")

    # ===== average (micro) =====
    all_tre_arr = np.asarray(all_tre, dtype=np.float64) if len(all_tre) else np.asarray([], dtype=np.float64)
    all_auc = error_auc(all_tre_arr, list(auc_thresholds)) if len(all_tre_arr) else {f"auc@{t}": 0.0 for t in auc_thresholds}

    results["average"] = {
        "num_pairs": int(len(all_tre_arr)),
        "num_failed": int(all_failed),
        "mean_tre_mre": float(np.nanmean(all_tre_arr)) if len(all_tre_arr) else float("nan"),
        "median_tre_mre": float(np.nanmedian(all_tre_arr)) if len(all_tre_arr) else float("nan"),
        "auc_tre": {f"auc@{t}": float(all_auc[f"auc@{t}"]) for t in auc_thresholds},
        "mean_n_matches": float(np.mean(all_n_matches)) if len(all_n_matches) else 0.0,
        "median_n_matches": float(np.median(all_n_matches)) if len(all_n_matches) else 0.0,
    }

    logging.info(f"[average] num_pairs={results['average']['num_pairs']} num_failed={results['average']['num_failed']}")
    logging.info(f"[average] mean_tre={results['average']['mean_tre_mre']:.4f} median_tre={results['average']['median_tre_mre']:.4f}")
    logging.info(f"[average] AUC TRE: {results['average']['auc_tre']}")

    return results


# =========================================================
# Runner
# =========================================================

def test_anhir(
    data_root,
    index_csv,
    split="all",
    method="xoftr",
    exp_name="ANHIR",
    print_out=False,
    save_dir="./results_anhir_homo/",
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

    # Load pairs from dataset_medium.csv
    scene_pairs = load_anhir_pairs_from_index(index_csv=index_csv, data_root=data_root, split=split)

    # Load matcher
    matcher = load_model(method, args)

    thresholds = [int(x) for x in args.auc_thresholds.split(",")]
    scene_results = eval_anhir(
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
        ransac_reproj_thr=getattr(args, "ransac_reproj_thr", 3.0),
    )

    results = OrderedDict({
        "method": method,
        "exp_name": exp_name,
        "split": split,
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
    parser = argparse.ArgumentParser("Benchmark ANHIR (index CSV) - Homography baseline + Landmark TRE")
    choose_method_arguments(parser)

    parser.add_argument("--data_root", type=str, default="/data6/liangyingping/tmp/dataset/anhir",
                        help="Dataset root, containing dataset_medium.csv and case folders")
    parser.add_argument("--index_csv", type=str, default="dataset_medium.csv",
                        help="Index csv name or full path")
    parser.add_argument("--split", type=str, default="all", choices=["training", "evaluation", "all"])

    parser.add_argument("--exp_name", type=str, default="ANHIR")
    parser.add_argument("--save_dir", type=str, default="./results_anhir_homo/")
    parser.add_argument("--print_out", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_figs", action="store_true")
    parser.add_argument("--svg", action="store_true")

    parser.add_argument("--gt_thr_px", type=float, default=5.0,
                        help="(unused for correctness now) kept for interface consistency.")
    parser.add_argument("--auc_thresholds", type=str, default="1,3,5,7,10,15,20",
                        help="Comma-separated thresholds for AUC computation.")
    parser.add_argument("--ransac_reproj_thr", type=float, default=3.0,
                        help="RANSAC reprojection threshold for cv2.findHomography")

    args, _ = parser.parse_known_args()
    add_method_arguments(parser, args.method)
    args = parser.parse_args()

    print(args)

    tt = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        index_csv = args.index_csv
        if not osp.isabs(index_csv):
            index_csv = osp.join(args.data_root, index_csv)

        test_anhir(
            data_root=args.data_root,
            index_csv=index_csv,
            split=args.split,
            method=args.method,
            exp_name=args.exp_name,
            print_out=args.print_out,
            save_dir=args.save_dir,
            save_figs=args.save_figs,
            debug=args.debug,
            args=args,
        )

    print(f"Elapsed time: {time.time() - tt:.2f}s")
