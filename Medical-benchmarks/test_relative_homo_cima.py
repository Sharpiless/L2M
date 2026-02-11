import argparse
import os
import os.path as osp
import json
import cv2
import numpy as np
import time
import warnings
import logging
from pathlib import Path
from collections import defaultdict, OrderedDict
from tqdm import tqdm

import torch
torch.backends.cudnn.enabled = False

from load_model import load_model, choose_method_arguments, add_method_arguments
from src.utils.metrics import error_auc
from src.utils.plotting import dynamic_alpha, make_matching_figure2


# ---------------------------
# Landmarks IO
# ---------------------------

def load_landmarks_csv(csv_path: str):
    """
    ImageJ Multi-point CSV example:
      ,X,Y
      0,829.5,262.2
    Return: pts [N,2] float32 (x,y)
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}

    if "x" in cols and "y" in cols:
        xcol, ycol = cols["x"], cols["y"]
    elif "x [px]" in cols and "y [px]" in cols:
        xcol, ycol = cols["x [px]"], cols["y [px]"]
    else:
        raise ValueError(f"CSV missing X/Y columns: {csv_path}. Columns: {list(df.columns)}")

    pts = df[[xcol, ycol]].to_numpy(dtype=np.float32)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        raise ValueError(f"No valid points in {csv_path}")
    return pts


def compute_gt_homography_from_landmarks(moving_pts, ref_pts):
    """
    moving -> ref, DLT (no RANSAC)
    """
    if len(moving_pts) < 4 or len(ref_pts) < 4:
        raise ValueError("Not enough landmarks to compute homography")

    moving_pts = moving_pts.astype(np.float64)
    ref_pts = ref_pts.astype(np.float64)

    if not np.isfinite(moving_pts).all() or not np.isfinite(ref_pts).all():
        raise ValueError("Landmarks contain NaN/Inf")

    H_gt, _ = cv2.findHomography(moving_pts, ref_pts, method=0)
    if H_gt is None or (not np.isfinite(H_gt).all()):
        raise ValueError("Failed to compute GT homography from landmarks")

    H_gt = H_gt / H_gt[2, 2]
    return H_gt.astype(np.float64)


# ---------------------------
# Build pairs: no naming rules, all-to-all within each scene directory
# ---------------------------

def list_samples_with_landmarks(scene_scale_dir: str, exts=(".jpg", ".png", ".jpeg")):
    """
    Find stems that have both image and csv:
      stem.jpg + stem.csv
    Return list of dicts: [{"stem","img","csv","pts"}]
    """
    files = os.listdir(scene_scale_dir)
    # collect possible image stems
    img_files = [f for f in files if f.lower().endswith(exts)]
    stems = [osp.splitext(f)[0] for f in img_files]

    samples = []
    for st in sorted(set(stems)):
        # find actual image path (prefer .jpg then .png etc)
        img_path = None
        for ext in exts:
            p = osp.join(scene_scale_dir, st + ext)
            if osp.exists(p):
                img_path = p
                break
        csv_path = osp.join(scene_scale_dir, st + ".csv")
        if img_path is None or (not osp.exists(csv_path)):
            continue
        samples.append({"stem": st, "img": img_path, "csv": csv_path})
    return samples


def build_all_pairs(samples, max_pairs=0, seed=0):
    """
    Build directed pairs (i -> j), i != j
    To avoid explosion, optionally subsample.
    """
    pairs = []
    n = len(samples)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pairs.append((i, j))

    if max_pairs and len(pairs) > max_pairs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[k] for k in idx]

    return pairs


def load_histology_pairs(root_dir: str, scale="scale-5pc",
                         max_pairs_per_scene=0, seed=0, img_exts=(".jpg", ".png", ".jpeg")):
    """
    No naming parsing. For each scene, collect all (img,csv) samples, then all-to-all pairing.

    Output:
      scene_pairs[scene] = list of {
        im0, im1, H, pair_id, lm0, lm1
      }
    where:
      im0 = moving, im1 = reference
      H_gt maps im0 -> im1 computed from GT landmarks (lm0 -> lm1)
    """
    root_dir = str(root_dir)
    scenes = ["mammary-gland_2", "lung-lesion_1", "lung-lobes_1", "lung-lobes_3", "lung-lobes_4"]
    scene_pairs = {}

    for scene in scenes:
        scene_scale_dir = osp.join(root_dir, scene, scale)
        if not osp.isdir(scene_scale_dir):
            raise FileNotFoundError(f"Missing directory: {scene_scale_dir}")

        samples = list_samples_with_landmarks(scene_scale_dir, exts=img_exts)
        print(f"[{scene}] found samples with (img+csv): {len(samples)} in {scene_scale_dir}")
        if len(samples) < 2:
            scene_pairs[scene] = []
            continue

        pair_indices = build_all_pairs(samples, max_pairs=max_pairs_per_scene, seed=seed)

        pairs = []
        for (i, j) in pair_indices:
            s0 = samples[i]
            s1 = samples[j]
            try:
                lm0 = load_landmarks_csv(s0["csv"])
                lm1 = load_landmarks_csv(s1["csv"])

                n = min(len(lm0), len(lm1))
                if n < 4:
                    continue
                lm0 = lm0[:n]
                lm1 = lm1[:n]

                H_gt = compute_gt_homography_from_landmarks(lm0, lm1)  # im0 -> im1
            except Exception as e:
                # 打印少量错误，避免刷屏
                # 你也可以注释掉
                # print(f"[PAIR BUILD FAIL] {scene}: {s0['stem']} -> {s1['stem']} err={e}")
                continue

            pair_id = f"{s0['stem']}__to__{s1['stem']}"
            pairs.append({
                "im0": s0["img"],
                "im1": s1["img"],
                "H": H_gt,
                "pair_id": pair_id,
                "lm0": lm0,
                "lm1": lm1,
            })

        scene_pairs[scene] = pairs
        print(f"[{scene}] Loaded pairs @ {scale}: {len(pairs)} (max_pairs_per_scene={max_pairs_per_scene or 'all'})")

    return scene_pairs


# ---------------------------
# Metrics & plotting helpers
# ---------------------------

def compute_mask_from_H(real_H, mkpts0, mkpts1, threshold=5):
    n = mkpts0.shape[0]
    if n == 0:
        return np.array([], dtype=bool)

    mk0_h = np.hstack([mkpts0, np.ones((n, 1))])
    mk1_h = np.hstack([mkpts1, np.ones((n, 1))])

    proj1 = (real_H @ mk0_h.T).T
    proj1 = proj1[:, :2] / proj1[:, 2:3]

    invH = np.linalg.inv(real_H)
    proj0 = (invH @ mk1_h.T).T
    proj0 = proj0[:, :2] / proj0[:, 2:3]

    e1 = np.linalg.norm(mkpts1 - proj1, axis=1)
    e0 = np.linalg.norm(mkpts0 - proj0, axis=1)
    mean_e = 0.5 * (e0 + e1)
    return mean_e < threshold


def save_matching_figure_histology(path, img0, img1, mkpts0, mkpts1, correct_mask, mean_distance, n_pix=5, svg=False):
    correct_mask = correct_mask.astype(float)
    precision = float(np.mean(correct_mask)) if len(correct_mask) > 0 else 0.0
    n_correct = int(np.sum(correct_mask))
    n = mkpts0.shape[0]

    color = np.zeros((n, 3), dtype=np.uint8)
    color[correct_mask == 0] = (255, 0, 0)
    color[correct_mask == 1] = (0, 255, 0)

    text = []
    text += [f"Mean Corner Dist: {mean_distance:.2f} px"]
    text += [f"Precision({n_pix}px) ({100*precision:.1f}%): {n_correct}/{n}"]

    make_matching_figure2(img0, img1, mkpts0, mkpts1, color, text=text, path=path, dpi=150, svg=svg)


def order_corners(corners):
    rect = np.zeros((4, 2), dtype="float32")
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]
    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]
    return rect


def compute_mean_corner_distance(real_H, pred_H, H, W):
    corners = np.array([[0, 0, 1],
                        [W - 1, 0, 1],
                        [0, H - 1, 1],
                        [W - 1, H - 1, 1]], dtype=np.float64)

    real_w = (corners @ real_H.T)
    real_w = real_w[:, :2] / real_w[:, 2:]

    pred_w = (corners @ pred_H.T)
    pred_w = pred_w[:, :2] / pred_w[:, 2:]

    real_w = order_corners(real_w.astype(np.float32))
    pred_w = order_corners(pred_w.astype(np.float32))
    return float(np.mean(np.linalg.norm(real_w - pred_w, axis=1)))


def eval_landmark_tre(pred_H, lm0, lm1):
    n = min(len(lm0), len(lm1))
    if n == 0:
        return np.inf
    pts0 = lm0[:n].astype(np.float64)
    pts1 = lm1[:n].astype(np.float64)

    pts0_h = np.hstack([pts0, np.ones((n, 1))])
    proj = (pred_H @ pts0_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    dist = np.linalg.norm(proj - pts1, axis=1)
    return float(np.mean(dist))


# ---------------------------
# Main eval loop
# ---------------------------

def eval_histology_homo(
    matcher,
    scene_pairs,
    save_figs,
    figures_dir=None,
    method=None,
    print_out=False,
    debug=False,
    svg=False,
    gt_thr_px=5
):
    scene_results = {}

    # 用于 overall
    all_corner_err = []
    all_tre = []
    all_failed = 0
    all_pairs = 0

    thresholds = [1, 3, 5, 7, 10, 15, 20]

    for scene_name, pairs in scene_pairs.items():
        scene_dir = osp.join(figures_dir, scene_name)
        if save_figs:
            os.makedirs(scene_dir, exist_ok=True)

        statis = defaultdict(list)
        logging.info(f"\nStart evaluation on Histology scene: {scene_name}\n")

        for i, pair in tqdm(enumerate(pairs), smoothing=.1, total=len(pairs)):
            if debug and i > 10:
                break

            im0 = pair["im0"]
            im1 = pair["im1"]
            gt_H = pair["H"]
            lm0 = pair["lm0"]
            lm1 = pair["lm1"]
            file_name = pair.get("pair_id", f"pair_{i}")

            match_res = matcher(im0, im1)
            mkpts0 = match_res["mkpts0"]
            mkpts1 = match_res["mkpts1"]
            img0 = match_res["img0"]
            img1 = match_res["img1"]

            if len(img0.shape) == 2:
                H_img, W_img = img0.shape
                img0_bgr = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
                img1_bgr = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            else:
                H_img, W_img, _ = img0.shape
                img0_bgr, img1_bgr = img0, img1

            pred_H = None
            corner_err = np.inf
            tre = np.inf

            try:
                if mkpts0.shape[0] >= 4:
                    pred_H, _ = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC)
                if pred_H is not None and np.isfinite(pred_H).all():
                    pred_H = pred_H / pred_H[2, 2]
                    corner_err = compute_mean_corner_distance(gt_H, pred_H, H_img, W_img)
                    tre = eval_landmark_tre(pred_H, lm0, lm1)
            except Exception as e:
                logging.info(f"[{scene_name}] {file_name} homography exception: {e}")

            if save_figs:
                if mkpts0.shape[0] > 0:
                    correct_mask = compute_mask_from_H(gt_H, mkpts0, mkpts1, threshold=gt_thr_px)
                else:
                    correct_mask = np.array([], dtype=bool)

                fig_path = osp.join(scene_dir, f"{file_name}_{method}.jpg")
                img0_rgb = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2RGB)
                img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
                save_matching_figure_histology(
                    path=fig_path,
                    img0=img0_rgb,
                    img1=img1_rgb,
                    mkpts0=mkpts0,
                    mkpts1=mkpts1,
                    correct_mask=correct_mask,
                    mean_distance=float(corner_err) if np.isfinite(corner_err) else 9999.0,
                    n_pix=gt_thr_px,
                    svg=svg,
                )

            # record
            if pred_H is None or (not np.isfinite(corner_err)) or (not np.isfinite(tre)):
                statis["failed"].append(i)
                all_failed += 1

            statis["corner_err"].append(corner_err)
            statis["tre"].append(tre)
            statis["n_matches"].append(int(mkpts0.shape[0]))

            all_pairs += 1
            all_corner_err.append(corner_err)
            all_tre.append(tre)

            if print_out:
                logging.info(f"[{scene_name}] {file_name} #M={mkpts0.shape[0]} corner_err={corner_err:.3f} tre={tre:.3f}")

        corner_err_all = np.array(statis["corner_err"], dtype=np.float64)
        tre_all = np.array(statis["tre"], dtype=np.float64)

        auc_corner = error_auc(corner_err_all, thresholds)
        auc_tre = error_auc(tre_all, thresholds)

        scene_results[scene_name] = {
            "num_pairs": len(pairs),
            "num_failed": len(statis["failed"]),
            "mean_corner_err": float(np.nanmean(corner_err_all)),
            "median_corner_err": float(np.nanmedian(corner_err_all)),
            "mean_tre": float(np.nanmean(tre_all)),
            "median_tre": float(np.nanmedian(tre_all)),
            "auc_corner": {f"auc@{t}": float(auc_corner[f"auc@{t}"]) for t in thresholds},
            "auc_tre": {f"auc@{t}": float(auc_tre[f"auc@{t}"]) for t in thresholds},
        }

        logging.info(f"[{scene_name}] pairs={len(pairs)} failed={len(statis['failed'])}")
        logging.info(f"[{scene_name}] AUC(corner)={auc_corner}")
        logging.info(f"[{scene_name}] AUC(TRE)={auc_tre}")

    # -------------------
    # overall summary
    # -------------------
    all_corner_err = np.array(all_corner_err, dtype=np.float64) if len(all_corner_err) else np.array([], dtype=np.float64)
    all_tre = np.array(all_tre, dtype=np.float64) if len(all_tre) else np.array([], dtype=np.float64)

    if len(all_corner_err) > 0:
        overall_auc_corner = error_auc(all_corner_err, thresholds)
        overall_auc_tre = error_auc(all_tre, thresholds)
        overall_results = {
            "num_pairs": int(all_pairs),
            "num_failed": int(all_failed),
            "mean_corner_err": float(np.nanmean(all_corner_err)),
            "median_corner_err": float(np.nanmedian(all_corner_err)),
            "mean_tre": float(np.nanmean(all_tre)),
            "median_tre": float(np.nanmedian(all_tre)),
            "auc_corner": {f"auc@{t}": float(overall_auc_corner[f"auc@{t}"]) for t in thresholds},
            "auc_tre": {f"auc@{t}": float(overall_auc_tre[f"auc@{t}"]) for t in thresholds},
        }
    else:
        overall_results = {
            "num_pairs": 0,
            "num_failed": 0,
            "mean_corner_err": float("nan"),
            "median_corner_err": float("nan"),
            "mean_tre": float("nan"),
            "median_tre": float("nan"),
            "auc_corner": {f"auc@{t}": float("nan") for t in thresholds},
            "auc_tre": {f"auc@{t}": float("nan") for t in thresholds},
        }

    return scene_results, overall_results



def test_histology(
    root_dir,
    method="xoftr",
    exp_name="Histology",
    print_out=False,
    save_dir="./results_histology_homo/",
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

    scene_pairs = load_histology_pairs(
        root_dir=root_dir,
        scale=args.scale,
        max_pairs_per_scene=args.max_pairs_per_scene,
        seed=args.seed,
        img_exts=tuple(args.img_exts.split(",")),
    )

    matcher = load_model(method, args)

    scene_results, overall_results = eval_histology_homo(
        matcher,
        scene_pairs,
        save_figs=save_figs,
        figures_dir=figures_dir,
        method=method,
        print_out=print_out,
        debug=debug,
        svg=getattr(args, "svg", False),
        gt_thr_px=getattr(args, "gt_thr_px", 5),
    )
    thresholds = [1, 3, 5, 7, 10, 15, 20]
    results = OrderedDict({
        "method": method,
        "exp_name": exp_name,
        "auc_thresholds": thresholds,
    })
    results.update({key: value for key, value in vars(args).items() if key not in results})
    results["scenes"] = scene_results
    results["overall"] = overall_results

    logging.info(f"Results: {json.dumps(results, indent=4)}")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    print(json.dumps(results, indent=2))
    print(f"Saved results to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Histology Landmarks (all pairs) in FIRE-style interface")
    choose_method_arguments(parser)

    parser.add_argument("--root_dir", type=str, default="/data6/liangyingping/tmp/dataset/Histology", help="Histology root")
    parser.add_argument("--scale", type=str, default="scale-5pc",
                        help="Which scale subfolder to use (e.g., scale-5pc / scale-10pc)")
    parser.add_argument("--exp_name", type=str, default="Histology_CIMA")
    parser.add_argument("--save_dir", type=str, default="./results_histology_homo/")
    parser.add_argument("--print_out", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_figs", action="store_true")
    parser.add_argument("--svg", action="store_true")

    # Histology specific
    parser.add_argument("--gt_thr_px", type=float, default=5.0,
                        help="Threshold (px) to color correct/incorrect matches using GT H")

    # all-pairs control
    parser.add_argument("--max_pairs_per_scene", type=int, default=300,
                        help="Limit number of pairs per scene (0 = all). Prevents combinational explosion.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for subsampling pairs")
    parser.add_argument("--img_exts", type=str, default=".jpg,.png,.jpeg",
                        help="Comma-separated image extensions to consider")

    args, remaining_args = parser.parse_known_args()
    add_method_arguments(parser, args.method)
    args = parser.parse_args()

    print(args)

    tt = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_histology(
            root_dir=args.root_dir,
            method=args.method,
            exp_name=args.exp_name,
            print_out=args.print_out,
            save_dir=args.save_dir,
            save_figs=args.save_figs,
            debug=args.debug,
            args=args
        )
    print(f"Elapsed time: {time.time() - tt:.2f}s")
