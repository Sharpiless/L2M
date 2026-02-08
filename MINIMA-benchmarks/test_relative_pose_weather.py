import argparse
import json
import logging
import os
import os.path as osp
import time
import warnings
from collections import defaultdict, OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

torch.backends.cudnn.enabled = False

from load_model import load_model, choose_method_arguments, add_method_arguments
from src.utils.metrics import (
    estimate_pose,
    relative_pose_error,
    error_auc,
    symmetric_epipolar_distance_numpy,
    epidist_prec,
)
from src.utils.plotting import dynamic_alpha, error_colormap


# ============================================================
# Helpers
# ============================================================

def _invert_se3(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4, dtype=np.float64)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv


def _relative_T_cam0_to_cam1_from_poses(T0, T1, pose_is_c2w: bool):
    """
    Return T_0to1 that maps X_cam0 -> X_cam1
    If pose_is_c2w: T0/T1 are c2w
      w2c1 @ c2w0
    Else: T0/T1 are w2c
      w2c1 @ c2w0
    """
    if pose_is_c2w:
        c2w0, c2w1 = T0, T1
        w2c1 = _invert_se3(c2w1)
        return w2c1 @ c2w0
    else:
        w2c0, w2c1 = T0, T1
        c2w0 = _invert_se3(w2c0)
        return w2c1 @ c2w0


def _to_root_relative(p: Path, data_root_dir: Path) -> str:
    try:
        return p.resolve().relative_to(data_root_dir.resolve()).as_posix()
    except Exception:
        return p.as_posix()


def _read_pose_txt(pose_path: Path):
    """
    Parse our exported pose txt:
      K 3 3
      ...
      T_cw 4 4  # world->camera
      ...
      T_wc 4 4  # camera->world
      ...
    Return: K(3,3), T_cw(4,4), T_wc(4,4)
    """
    raw_lines = pose_path.read_text(encoding="utf-8").splitlines()
    lines = []
    for ln in raw_lines:
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith("#"):
            continue
        lines.append(ln)

    def _read_block(tag, rows, cols):
        idx = None
        for i, ln in enumerate(lines):
            if ln.startswith(tag + " "):
                idx = i
                break
        if idx is None:
            raise ValueError(f"Cannot find block '{tag}' in {pose_path}")

        mat_lines = lines[idx + 1 : idx + 1 + rows]
        if len(mat_lines) != rows:
            raise ValueError(f"Block '{tag}' rows mismatch in {pose_path}")

        mat = []
        for ln in mat_lines:
            vals = [float(x) for x in ln.split()]
            if len(vals) != cols:
                raise ValueError(f"Block '{tag}' cols mismatch in {pose_path}: {len(vals)} != {cols}")
            mat.append(vals)
        return np.array(mat, dtype=np.float64)

    K = _read_block("K", 3, 3).astype(np.float32)
    T_cw = _read_block("T_cw", 4, 4)
    T_wc = _read_block("T_wc", 4, 4)
    return K, T_cw, T_wc


# ============================================================
# Multi-scene loader for REALWORLD_STREAK (images/ + poses/)
# ============================================================

def load_preprocessed_scenes_pairs(
    data_root_dir: Path,
    max_pairs_per_scene: int = -1,
    pair_stride: int = 1,
    pair_gap: int = 1,
    scene_glob: str = "*",
    pose_is_c2w: bool = True,
    images_dirname: str = "images",
    poses_dirname: str = "poses",
):
    """
    Scene structure expected:
      scene/
        images/
          xxx.jpg/png...
        poses/
          xxx.txt   (contains K, T_cw, T_wc)

    Pairing strategy:
      - sort by image filename
      - form pairs (i, i+pair_gap) with stride pair_stride
    """
    data_root_dir = Path(data_root_dir)
    scene_pairs = {}

    scene_dirs = sorted([p for p in data_root_dir.glob(scene_glob) if p.is_dir()])
    if len(scene_dirs) == 0:
        raise FileNotFoundError(f"No scene folders found under: {data_root_dir} (glob={scene_glob})")

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    for scene_dir in scene_dirs:
        img_dir = scene_dir / images_dirname
        pose_dir = scene_dir / poses_dirname

        if not img_dir.exists() or not pose_dir.exists():
            continue

        imgs = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
        if len(imgs) < (1 + pair_gap):
            continue

        Ks, dists, poses = [], [], []
        img_paths = []

        for img_p in imgs:
            base = img_p.stem
            pose_p = pose_dir / f"{base}.txt"
            if not pose_p.exists():
                # skip if pose missing
                continue

            K, T_cw, T_wc = _read_pose_txt(pose_p)

            # Use pose representation requested by downstream code
            # pose_is_c2w=True -> store c2w (T_wc)
            # pose_is_c2w=False -> store w2c (T_cw)
            T = T_wc if pose_is_c2w else T_cw

            img_paths.append(img_p)
            Ks.append(K)

            # undistorted images -> no distortion
            dists.append(np.zeros(5, dtype=float))

            poses.append(T)

        n_frames = len(img_paths)
        if n_frames < (1 + pair_gap):
            continue

        pairs = []
        count = 0
        for i in range(0, n_frames - pair_gap, pair_stride):
            j = i + pair_gap

            T_0to1 = _relative_T_cam0_to_cam1_from_poses(poses[i], poses[j], pose_is_c2w=pose_is_c2w)

            im0_rel = _to_root_relative(img_paths[i], data_root_dir)
            im1_rel = _to_root_relative(img_paths[j], data_root_dir)

            pairs.append(
                {
                    "im0": im0_rel,
                    "im1": im1_rel,
                    "dist0": dists[i],
                    "dist1": dists[j],
                    "K0": Ks[i],
                    "K1": Ks[j],
                    "T_0to1": T_0to1,
                }
            )
            count += 1
            if max_pairs_per_scene > 0 and count >= max_pairs_per_scene:
                break

        if len(pairs) > 0:
            scene_pairs[scene_dir.name] = pairs

    if len(scene_pairs) == 0:
        raise RuntimeError(
            f"Found scene folders but no valid pairs. "
            f"Check that each scene has '{images_dirname}/' and '{poses_dirname}/', "
            f"and pair_gap={pair_gap}, scene_glob={scene_glob}."
        )

    return scene_pairs

def calculate_epi_errs(mkpts0, mkpts1, inlier_mask, T_0to1, K0, K1):
    Tx = np.cross(np.eye(3), T_0to1[:3, 3])
    E_mat = Tx @ T_0to1[:3, :3]
    mkpts0_inliers = mkpts0[inlier_mask]
    mkpts1_inliers = mkpts1[inlier_mask]
    if inlier_mask is not None and len(inlier_mask) != 0:
        epi_errs = symmetric_epipolar_distance_numpy(mkpts0_inliers, mkpts1_inliers, E_mat, K0, K1)
    else:
        epi_errs = np.inf
    return epi_errs


def calculate_epi_errs_no_inlier(mkpts0, mkpts1, inlier_mask, T_0to1, K0, K1):
    Tx = np.cross(np.eye(3), T_0to1[:3, 3])
    E_mat = Tx @ T_0to1[:3, :3]
    mkpts0_inliers = mkpts0
    mkpts1_inliers = mkpts1
    epi_errs = symmetric_epipolar_distance_numpy(mkpts0_inliers, mkpts1_inliers, E_mat, K0, K1)
    return epi_errs


# ============================================================
# aggregation helpers (unchanged)
# ============================================================

def aggregiate_scenes(scene_pose_auc, thresholds):
    temp_pose_auc = {}
    for npz_name in scene_pose_auc.keys():
        scene_name = npz_name.split("_scene")[0]
        temp_pose_auc[scene_name] = [np.zeros(len(thresholds), dtype=np.float32), 0]
    for npz_name in scene_pose_auc.keys():
        scene_name = npz_name.split("_scene")[0]
        temp_pose_auc[scene_name][0] += scene_pose_auc[npz_name]
        temp_pose_auc[scene_name][1] += 1

    agg_pose_auc = {}
    for scene_name in temp_pose_auc.keys():
        agg_pose_auc[scene_name] = temp_pose_auc[scene_name][0] / temp_pose_auc[scene_name][1]

    return agg_pose_auc


def aggregate_precisions(precs, precs_no_inlier):
    temp_precs = defaultdict(lambda: defaultdict(list))
    temp_precs_no_inlier = defaultdict(lambda: defaultdict(list))

    for scene_name, precision_dict in precs.items():
        main_scene = scene_name.split("_scene")[0]
        for threshold, precision in precision_dict.items():
            temp_precs[main_scene][threshold].append(precision)

    for scene_name, precision_dict in precs_no_inlier.items():
        main_scene = scene_name.split("_scene")[0]
        for threshold, precision in precision_dict.items():
            temp_precs_no_inlier[main_scene][threshold].append(precision)

    agg_precs = {
        scene: {threshold: np.mean(values) for threshold, values in thresholds_dict.items()}
        for scene, thresholds_dict in temp_precs.items()
    }

    agg_precs_no_inlier = {
        scene: {threshold: np.mean(values) for threshold, values in thresholds_dict.items()}
        for scene, thresholds_dict in temp_precs_no_inlier.items()
    }

    return agg_precs, agg_precs_no_inlier


# ============================================================
# eval loop (unchanged)
# ============================================================

def eval_relapose(
    matcher,
    data_root,
    scene_pairs,
    ransac_thres,
    thresholds,
    save_figs,
    figures_dir=None,
    method=None,
    print_out=False,
    debug=False,
    args=None,
):
    scene_pose_auc = {}
    precs = {}
    precs_no_inlier = {}

    for scene_name in scene_pairs.keys():
        if args.svg:
            scene_dir = figures_dir
        else:
            scene_dir = osp.join(figures_dir, scene_name.split(".")[0])
        if save_figs and not osp.exists(scene_dir):
            os.makedirs(scene_dir)

        pairs = scene_pairs[scene_name]
        statis = defaultdict(list)
        np.set_printoptions(precision=2)

        logging.info(f"\nStart evaluation on Scene ({scene_name})\n")
        for i, pair in tqdm(enumerate(pairs), smoothing=0.1, total=len(pairs)):
            if debug and i > 10:
                break

            T_0to1 = pair["T_0to1"]
            im0 = str(data_root / pair["im0"])
            im1 = str(data_root / pair["im1"])

            match_res = matcher(im0, im1, pair["K0"], pair["K1"], pair["dist0"], pair["dist1"], read_color=False)
            matches = match_res["matches"]
            new_K0 = match_res["new_K0"]
            new_K1 = match_res["new_K1"]
            mkpts0 = match_res["mkpts0"]
            mkpts1 = match_res["mkpts1"]
            n = len(matches)

            ret = estimate_pose(mkpts0, mkpts1, new_K0, new_K1, thresh=ransac_thres)

            if ret is None:
                R, t, inliers = None, None, None
                t_err, R_err = np.inf, np.inf
                epi_errs = np.array([]).astype(np.float32)
                epi_errs_no_inlier = np.array([]).astype(np.float32)
                statis["failed"].append(i)
                statis["R_errs"].append(R_err)
                statis["t_errs"].append(t_err)
                statis["epi_errs"].append(epi_errs)
                statis["epi_errs_no_inlier"].append(epi_errs_no_inlier)
                statis["inliers"].append(np.array([]).astype(np.bool_))
                statis["match_nums"].append(n)
            else:
                R, t, inliers = ret
                t_err, R_err = relative_pose_error(T_0to1, R, t)
                epi_errs = calculate_epi_errs(mkpts0, mkpts1, inliers, T_0to1, new_K0, new_K1)
                epi_errs_no_inlier = calculate_epi_errs_no_inlier(mkpts0, mkpts1, inliers, T_0to1, new_K0, new_K1)
                statis["epi_errs"].append(epi_errs)
                statis["epi_errs_no_inlier"].append(epi_errs_no_inlier)
                statis["R_errs"].append(R_err)
                statis["t_errs"].append(t_err)
                statis["inliers"].append(inliers.sum() / len(mkpts0))
                statis["match_nums"].append(n)
                if print_out:
                    logging.info(f"#M={len(matches)} R={R_err:.3f}, t={t_err:.3f}")


        logging.info(f"Scene: {scene_name} Total samples: {len(pairs)} Failed:{len(statis['failed'])}. \n")
        pose_errors = np.max(np.stack([statis["R_errs"], statis["t_errs"]]), axis=0)
        pose_auc = error_auc(pose_errors, thresholds)

        epi_err_thr = 5e-4
        dist_thresholds = [epi_err_thr]
        precs[scene_name] = epidist_prec(np.array(statis["epi_errs"], dtype=object), dist_thresholds, True, True)
        precs_no_inlier[scene_name] = epidist_prec(
            np.array(statis["epi_errs_no_inlier"], dtype=object), dist_thresholds, True, False
        )

        scene_pose_auc[scene_name] = 100 * np.array([pose_auc[f"auc@{t}"] for t in thresholds])
        logging.info(f"{scene_name} {pose_auc} {precs} {precs_no_inlier}")

    agg_pose_auc = aggregiate_scenes(scene_pose_auc, thresholds)
    agg_precs, agg_precs_no_inlier = aggregate_precisions(precs, precs_no_inlier)
    return scene_pose_auc, agg_pose_auc, precs, precs_no_inlier, agg_precs, agg_precs_no_inlier


# ============================================================
# entry (multi-scene)
# ============================================================

def test_relative_pose_multiscene(
    data_root_dir,
    method="xoftr",
    exp_name="REALWORLD_STREAK_MultiScene",
    ransac_thres=1.5,
    print_out=False,
    save_dir=None,
    save_figs=False,
    debug=False,
    args=None,
):
    if method == "roma":
        if args.ckpt is None:
            save_ = "roma"
        else:
            save_ = args.ckpt.split("/")[-1].replace(".ckpt", "")
    elif args.ckpt is None:
        save_ = method
    else:
        save_ = args.ckpt.split("/")[-1].replace(".ckpt", "")

    path_ = osp.join(save_dir, method, save_)
    if args.debug:
        path_ = osp.join(save_dir, method, save_, "debug")
    if not osp.exists(path_):
        os.makedirs(path_)

    if hasattr(args, "thr"):
        path = osp.join(path_, f"{exp_name}_thresh_{args.thr}")
    else:
        path = osp.join(path_, f"{exp_name}")
    exp_dir = path
    os.mkdir(exp_dir)

    results_file = osp.join(exp_dir, "results.json")
    logging.basicConfig(
        filename=results_file.replace(".json", ".log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    figures_dir = osp.join(exp_dir, "match_figures")
    if save_figs:
        os.mkdir(figures_dir)

    logging.info(f"args: {args}")

    data_root = Path(data_root_dir)

    # Load pairs (multi-scene) from images/ + poses/
    scene_pairs = load_preprocessed_scenes_pairs(
        data_root_dir=data_root,
        max_pairs_per_scene=args.max_pairs_per_scene,
        pair_stride=args.pair_stride,
        pair_gap=args.pair_gap,
        scene_glob=args.scene_glob,
        pose_is_c2w=(not args.pose_is_w2c),
        images_dirname=args.images_dirname,
        poses_dirname=args.poses_dirname,
    )

    # Load method
    matcher = load_model(method, args)

    thresholds = [5, 10, 20]

    scene_pose_auc, agg_pose_auc, precs, precs_no_inlier, agg_precs, agg_precs_no_inlier = eval_relapose(
        matcher,
        data_root,
        scene_pairs,
        ransac_thres=ransac_thres,
        thresholds=thresholds,
        save_figs=save_figs,
        figures_dir=figures_dir,
        method=method,
        print_out=print_out,
        debug=debug,
        args=args,
    )

    results = OrderedDict(
        {
            "method": method,
            "exp_name": exp_name,
            "ransac_thres": ransac_thres,
            "auc_thresholds": thresholds,
        }
    )
    results.update({key: value for key, value in vars(args).items() if key not in results})
    results.update({key: value.tolist() for key, value in agg_pose_auc.items()})
    results.update({key: value.tolist() for key, value in scene_pose_auc.items()})

    # -----------------------------
    # Add overall average results
    # -----------------------------
    def _mean_auc_dict(d):
        if len(d) == 0:
            return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        return np.mean(np.stack(list(d.values()), axis=0), axis=0)

    def _mean_prec_dict(d):
        if len(d) == 0:
            return {}
        tmp = defaultdict(list)
        for _, thr_map in d.items():
            for thr, val in thr_map.items():
                if val is None:
                    continue
                tmp[thr].append(float(val))
        return {thr: float(np.mean(vals)) for thr, vals in tmp.items() if len(vals) > 0}

    avg_auc_over_agg_scenes = _mean_auc_dict(agg_pose_auc)
    avg_precs_over_agg_scenes = _mean_prec_dict(agg_precs)
    avg_precs_no_inlier_over_agg_scenes = _mean_prec_dict(agg_precs_no_inlier)

    avg_auc_over_all_npz_scenes = _mean_auc_dict(scene_pose_auc)
    avg_precs_over_all_npz_scenes = _mean_prec_dict(precs)
    avg_precs_no_inlier_over_all_npz_scenes = _mean_prec_dict(precs_no_inlier)

    results["avg_auc_over_agg_scenes"] = (
        (100.0 * (avg_auc_over_agg_scenes / 100.0)).tolist()
        if np.isfinite(avg_auc_over_agg_scenes).all()
        else avg_auc_over_agg_scenes.tolist()
    )
    results["avg_auc_over_all_npz_scenes"] = (
        (100.0 * (avg_auc_over_all_npz_scenes / 100.0)).tolist()
        if np.isfinite(avg_auc_over_all_npz_scenes).all()
        else avg_auc_over_all_npz_scenes.tolist()
    )

    print("avg_auc_over_agg_scenes:", results["avg_auc_over_agg_scenes"])
    print("avg_auc_over_all_npz_scenes:", results["avg_auc_over_all_npz_scenes"])

    results["avg_precs_over_agg_scenes"] = avg_precs_over_agg_scenes
    results["avg_precs_no_inlier_over_agg_scenes"] = avg_precs_no_inlier_over_agg_scenes
    results["avg_precs_over_all_npz_scenes"] = avg_precs_over_all_npz_scenes
    results["avg_precs_no_inlier_over_all_npz_scenes"] = avg_precs_no_inlier_over_all_npz_scenes

    results.update({f"precs_{key}": value for key, value in precs.items()})
    results.update({f"precs_no_inlier_{key}": value for key, value in precs_no_inlier.items()})
    results.update({f"agg_precs_{key}": value for key, value in agg_precs.items()})
    results.update({f"agg_precs_no_inlier_{key}": value for key, value in agg_precs_no_inlier.items()})

    logging.info(f"Results: {json.dumps(results, indent=4)}")

    with open(results_file, "w") as outfile:
        json.dump(results, outfile, indent=4)

    logging.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Relative Pose (REALWORLD_STREAK preprocessed multi-scene)")

    choose_method_arguments(parser)

    # ---- multi-scene args ----
    parser.add_argument("--exp_name", type=str, default="REALWORLD_STREAK_MultiScene")
    parser.add_argument(
        "--data_root_dir",
        type=str,
        default="/data6/liangyingping/tmp/dataset/weather/realworld_streak",
        help="root folder containing many scene subfolders",
    )
    parser.add_argument(
        "--scene_glob",
        type=str,
        default="*",
        help="glob pattern for selecting scene folders, e.g. 'scene_*'",
    )

    parser.add_argument("--pair_gap", type=int, default=2, help="pair (i, i+pair_gap) within a scene")
    parser.add_argument("--pair_stride", type=int, default=1, help="stride when sampling i")
    parser.add_argument("--max_pairs_per_scene", type=int, default=100, help="max pairs per scene (-1 all)")

    parser.add_argument(
        "--pose_is_w2c",
        action="store_true",
        help="use T_cw (world->camera) from poses txt (default uses T_wc camera->world)",
    )

    # dataset layout args
    parser.add_argument("--images_dirname", type=str, default="images", help="scene subdir for images")
    parser.add_argument("--poses_dirname", type=str, default="poses", help="scene subdir for per-image pose txt")

    # ---- output / eval args ----
    parser.add_argument("--save_dir", type=str, default="./results_realworldstreak_pose/")
    parser.add_argument("--ransac_thres", type=float, default=1.5)
    parser.add_argument("--e_name", type=str, default=None)
    parser.add_argument("--print_out", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_figs", action="store_true")
    parser.add_argument("--svg", action="store_true")

    args, remaining_args = parser.parse_known_args()
    add_method_arguments(parser, args.method)
    args = parser.parse_args()

    print(args)

    if args.e_name is not None:
        save_dir = osp.join(args.save_dir, args.e_name)
    else:
        save_dir = args.save_dir

    tt = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_relative_pose_multiscene(
            Path(args.data_root_dir),
            args.method,
            args.exp_name,
            ransac_thres=args.ransac_thres,
            print_out=args.print_out,
            save_dir=save_dir,
            save_figs=args.save_figs,
            debug=args.debug,
            args=args,
        )
    print(f"Elapsed time: {time.time() - tt}")
