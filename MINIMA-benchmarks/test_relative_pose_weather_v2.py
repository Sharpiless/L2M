import argparse
import json
import logging
import os
import os.path as osp
import time
import warnings
from collections import defaultdict, OrderedDict
from pathlib import Path

import sys
sys.path.append("/data6/liangyingping/tmp/WeatherGS/3DGS")
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

# --------- 这里按你实际文件位置改一下 ----------
# 你给的那段代码里最后有 sceneLoadTypeCallbacks = { "Colmap": ..., "Blender": ... }
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.graphics_utils import getWorld2View2, fov2focal
# --------------------------------------------


# ============================================================
# helper: build intrinsics K from CameraInfo
# ============================================================
def camera_info_to_K(cam_info):
    """
    cam_info has: FovX, FovY, width, height
    We compute fx, fy from fov -> focal.
    """
    w, h = int(cam_info.width), int(cam_info.height)
    fx = float(fov2focal(float(cam_info.FovX), w))
    fy = float(fov2focal(float(cam_info.FovY), h))
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


def caminfo_to_W2C(cam_info):
    """
    Use your own util getWorld2View2(R,T) to avoid ambiguity of R transpose convention.
    Returns W2C 4x4.
    """
    W2C = getWorld2View2(cam_info.R, cam_info.T)
    W2C = np.array(W2C, dtype=np.float64)
    return W2C


def relpose_from_W2C(W2C0, W2C1):
    """
    Given world-to-camera matrices W2C0, W2C1:
      x_c = Rwc x_w + twc
    Relative transform cam0 -> cam1:
      x_c1 = R_rel x_c0 + t_rel
      R_rel = R1 R0^T
      t_rel = t1 - R_rel t0
    """
    R0 = W2C0[:3, :3]
    t0 = W2C0[:3, 3]
    R1 = W2C1[:3, :3]
    t1 = W2C1[:3, 3]

    R_rel = R1 @ R0.T
    t_rel = t1 - R_rel @ t0

    T_0to1 = np.eye(4, dtype=np.float64)
    T_0to1[:3, :3] = R_rel
    T_0to1[:3, 3] = t_rel
    return T_0to1


# ============================================================
# multi-scene loader (Colmap/Blender)
# ============================================================
def discover_scenes(root_dir: Path):
    """
    A 'scene' is a subfolder under root_dir.
    """
    root_dir = Path(root_dir)
    scenes = [p for p in root_dir.iterdir() if p.is_dir()]
    scenes = sorted(scenes, key=lambda p: p.name)
    return scenes


def build_pairs_for_scene(
    args,
    scene_dir: Path,
    scene_type: str,
    images_dir: str = None,
    masks_dir: str = None,
    eval_split: bool = False,
    llffhold: int = 8,
    use_test_cameras: bool = False,
    max_pairs: int = -1,
    pairing: str = "adjacent",  # "adjacent" or "all"
):
    """
    Use sceneLoadTypeCallbacks[scene_type] to read SceneInfo.
    Then build pair list:
      each pair dict includes:
        im0, im1 (absolute path),
        K0, K1, dist0, dist1,
        T_0to1 (GT)
    """
    if scene_type not in sceneLoadTypeCallbacks:
        raise ValueError(f"Unknown scene_type={scene_type}. Must be one of {list(sceneLoadTypeCallbacks.keys())}")

    reader = sceneLoadTypeCallbacks[scene_type]

    if scene_type == "Colmap":
        scene_info = reader(
            path=str(scene_dir),
            images=images_dir,      # None -> default "images"
            eval=eval_split,
            llffhold=llffhold,
            masks=masks_dir,
        )
        cams = scene_info.test_cameras if use_test_cameras else scene_info.train_cameras

        # 统一拿相机列表（已经按 image_name 排序过）
        cam_infos = cams

    elif scene_type == "Blender":
        # 你的 readNerfSyntheticInfo(path, white_background, eval, extension=".png")
        # 这里给一个参数 white_background，默认 False；你需要白底可以加 CLI 参数扩展
        white_background = False
        scene_info = reader(
            path=str(scene_dir),
            white_background=white_background,
            eval=eval_split,
            extension=".png",
        )
        cam_infos = scene_info.test_cameras if use_test_cameras else scene_info.train_cameras
    else:
        raise ValueError(f"Unhandled scene_type={scene_type}")

    if len(cam_infos) < 2:
        return []

    # pairing strategy
    idx_pairs = []
    gaps = args.gaps
    if pairing == "adjacent":
        idx_pairs = [(i, i + gaps) for i in range(len(cam_infos) - gaps)]
    elif pairing == "all":
        for i in range(len(cam_infos)):
            for j in range(i + 1, len(cam_infos)):
                idx_pairs.append((i, j))
    else:
        raise ValueError(f"Unknown pairing={pairing}. Use adjacent|all")

    if max_pairs > 0:
        idx_pairs = idx_pairs[:max_pairs]

    pairs = []
    for i0, i1 in idx_pairs:
        c0 = cam_infos[i0]
        c1 = cam_infos[i1]

        im0 = str(Path(c0.image_path))
        im1 = str(Path(c1.image_path))

        K0 = camera_info_to_K(c0)
        K1 = camera_info_to_K(c1)
        dist0 = np.zeros(5, dtype=float)
        dist1 = np.zeros(5, dtype=float)

        W2C0 = caminfo_to_W2C(c0)
        W2C1 = caminfo_to_W2C(c1)
        T_0to1 = relpose_from_W2C(W2C0, W2C1)

        pairs.append(
            {
                "im0": im0,
                "im1": im1,
                "K0": K0,
                "K1": K1,
                "dist0": dist0,
                "dist1": dist1,
                "T_0to1": T_0to1,
            }
        )

    return pairs


def load_multiscene_pairs(
    args,
    scenes_root: Path,
    scene_type: str,
    images_dir: str = None,
    masks_dir: str = None,
    eval_split: bool = False,
    llffhold: int = 8,
    use_test_cameras: bool = False,
    max_pairs_per_scene: int = -1,
    pairing: str = "adjacent",
):
    """
    Return:
      scene_pairs = { "<scene_name>": [pair, pair, ...], ... }
    """
    scene_pairs = {}
    scene_dirs = discover_scenes(scenes_root)
    if len(scene_dirs) == 0:
        raise ValueError(f"No scene folders found under {scenes_root}")

    for sd in scene_dirs:
        pairs = build_pairs_for_scene(
            args=args,
            scene_dir=sd,
            scene_type=scene_type,
            images_dir=images_dir,
            masks_dir=masks_dir,
            eval_split=eval_split,
            llffhold=llffhold,
            use_test_cameras=use_test_cameras,
            max_pairs=max_pairs_per_scene,
            pairing=pairing,
        )
        if len(pairs) == 0:
            logging.warning(f"[skip] scene {sd.name}: no valid pairs")
            continue
        scene_pairs[sd.name] = pairs

    if len(scene_pairs) == 0:
        raise ValueError("All scenes have 0 pairs. Check paths / scene_type / images folder.")
    return scene_pairs


def calculate_epi_errs(mkpts0, mkpts1, inlier_mask, T_0to1, K0, K1):
    Tx = np.cross(np.eye(3), T_0to1[:3, 3])
    E_mat = Tx @ T_0to1[:3, :3]
    if inlier_mask is not None and len(inlier_mask) != 0:
        mkpts0_inliers = mkpts0[inlier_mask]
        mkpts1_inliers = mkpts1[inlier_mask]
        epi_errs = symmetric_epipolar_distance_numpy(mkpts0_inliers, mkpts1_inliers, E_mat, K0, K1)
    else:
        epi_errs = np.inf
    return epi_errs


def calculate_epi_errs_no_inlier(mkpts0, mkpts1, inlier_mask, T_0to1, K0, K1):
    Tx = np.cross(np.eye(3), T_0to1[:3, 3])
    E_mat = Tx @ T_0to1[:3, :3]
    return symmetric_epipolar_distance_numpy(mkpts0, mkpts1, E_mat, K0, K1)


# ============================================================
# aggregation helpers (keep)
# ============================================================
def aggregiate_scenes(scene_pose_auc, thresholds):
    temp_pose_auc = {}
    for scene_name in scene_pose_auc.keys():
        main_scene = scene_name.split("_scene")[0]
        temp_pose_auc[main_scene] = [np.zeros(len(thresholds), dtype=np.float32), 0]
    for scene_name in scene_pose_auc.keys():
        main_scene = scene_name.split("_scene")[0]
        temp_pose_auc[main_scene][0] += scene_pose_auc[scene_name]
        temp_pose_auc[main_scene][1] += 1

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
# eval loop (minor change: im0/im1 are absolute now)
# ============================================================
def eval_relapose(
    matcher,
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

        logging.info(f"\nStart evaluation on scene ({scene_name})\n")
        for i, pair in tqdm(enumerate(pairs), smoothing=0.1, total=len(pairs)):
            if debug and i > 10:
                break

            T_0to1 = pair["T_0to1"]

            # absolute path
            im0 = pair["im0"]
            im1 = pair["im1"]

            match_res = matcher(im0, im1, pair["K0"], pair["K1"], pair["dist0"], pair["dist1"], read_color=False)
            matches = match_res["matches"]
            new_K0 = match_res["new_K0"]
            new_K1 = match_res["new_K1"]
            mkpts0 = match_res["mkpts0"]
            mkpts1 = match_res["mkpts1"]
            n = len(matches)

            ret = estimate_pose(mkpts0, mkpts1, new_K0, new_K1, thresh=ransac_thres)

            if ret is None:
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
                inliers = np.array([]).astype(bool)
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
        logging.info(f"{scene_name} {pose_auc} {precs[scene_name]} {precs_no_inlier[scene_name]}")

    agg_pose_auc = aggregiate_scenes(scene_pose_auc, thresholds)
    agg_precs, agg_precs_no_inlier = aggregate_precisions(precs, precs_no_inlier)
    return scene_pose_auc, agg_pose_auc, precs, precs_no_inlier, agg_precs, agg_precs_no_inlier


# ============================================================
# entry
# ============================================================
def test_relative_pose_multiscene(
    scenes_root_dir,
    method="xoftr",
    exp_name="MultiScene",
    ransac_thres=1.5,
    print_out=False,
    save_dir=None,
    save_figs=False,
    debug=False,
    args=None,
):
    if method == "roma":
        save_ = "roma" if args.ckpt is None else args.ckpt.split("/")[-1].replace(".ckpt", "")
    elif args.ckpt is None:
        save_ = method
    else:
        save_ = args.ckpt.split("/")[-1].replace(".ckpt", "")

    path_ = osp.join(save_dir, method, save_)
    if args.debug:
        path_ = osp.join(save_dir, method, save_, "debug")
    os.makedirs(path_, exist_ok=True)

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

    scenes_root = Path(scenes_root_dir)

    # Load multi-scene pairs
    scene_pairs = load_multiscene_pairs(
        args=args,
        scenes_root=scenes_root,
        scene_type=args.scene_type,
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        eval_split=args.eval_split,
        llffhold=args.llffhold,
        use_test_cameras=args.use_test_cameras,
        max_pairs_per_scene=args.max_pairs_per_scene,
        pairing=args.pairing,
    )

    # Load matcher
    matcher = load_model(method, args)

    thresholds = [5, 10, 20]

    scene_pose_auc, agg_pose_auc, precs, precs_no_inlier, agg_precs, agg_precs_no_inlier = eval_relapose(
        matcher=matcher,
        scene_pairs=scene_pairs,
        ransac_thres=ransac_thres,
        thresholds=thresholds,
        save_figs=save_figs,
        figures_dir=figures_dir,
        method=method,
        print_out=print_out,
        debug=debug,
        args=args,
    )
    # -------- split by weather keywords in scene name --------
    groups = split_scene_keys_by_weather(scene_pose_auc.keys())

    rain_auc = mean_auc_for_keys(scene_pose_auc, groups["rain"])
    snow_auc = mean_auc_for_keys(scene_pose_auc, groups["snow"])

    rain_prec = mean_prec_for_keys(precs, groups["rain"])
    snow_prec = mean_prec_for_keys(precs, groups["snow"])

    rain_prec_no_inlier = mean_prec_for_keys(precs_no_inlier, groups["rain"])
    snow_prec_no_inlier = mean_prec_for_keys(precs_no_inlier, groups["snow"])
    # --------------------------------------------------------

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

    # overall average (keep same style)
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
    avg_auc_over_all_scenes = _mean_auc_dict(scene_pose_auc)
    results["avg_auc_over_agg_scenes"] = avg_auc_over_agg_scenes.tolist()
    results["avg_auc_over_all_scenes"] = avg_auc_over_all_scenes.tolist()

    results["avg_precs_over_agg_scenes"] = _mean_prec_dict(agg_precs)
    results["avg_precs_no_inlier_over_agg_scenes"] = _mean_prec_dict(agg_precs_no_inlier)
    results["avg_precs_over_all_scenes"] = _mean_prec_dict(precs)
    results["avg_precs_no_inlier_over_all_scenes"] = _mean_prec_dict(precs_no_inlier)

    results.update({f"precs_{k}": v for k, v in precs.items()})
    results.update({f"precs_no_inlier_{k}": v for k, v in precs_no_inlier.items()})
    results.update({f"agg_precs_{k}": v for k, v in agg_precs.items()})
    results.update({f"agg_precs_no_inlier_{k}": v for k, v in agg_precs_no_inlier.items()})

    logging.info(f"Results: {json.dumps(results, indent=4)}")
    with open(results_file, "w") as outfile:
        json.dump(results, outfile, indent=4)
    logging.info(f"Results saved to {results_file}")

    print("avg_auc_over_agg_scenes:", results["avg_auc_over_agg_scenes"])
    print("avg_auc_over_all_scenes:", results["avg_auc_over_all_scenes"])

    print("== split results by keyword ==")
    print(f"rain scenes ({len(groups['rain'])}):", groups["rain"][:5], "..." if len(groups["rain"]) > 5 else "")
    print(f"snow scenes ({len(groups['snow'])}):", groups["snow"][:5], "..." if len(groups["snow"]) > 5 else "")
    if len(groups["other"]) > 0:
        print(f"other scenes ({len(groups['other'])}):", groups["other"][:5], "..." if len(groups["other"]) > 5 else "")

    print("avg_auc_rain:", rain_auc.tolist())
    print("avg_auc_snow:", snow_auc.tolist())
    print("avg_prec_rain:", rain_prec)
    print("avg_prec_snow:", snow_prec)
    print("avg_prec_no_inlier_rain:", rain_prec_no_inlier)
    print("avg_prec_no_inlier_snow:", snow_prec_no_inlier)


def split_scene_keys_by_weather(scene_keys):
    """
    scene name contains keywords like 'rain' / 'snow' (case-insensitive).
    Returns dict: {'rain': [...], 'snow': [...], 'other': [...]}
    """
    groups = {"rain": [], "snow": [], "other": []}
    for k in scene_keys:
        lk = k.lower()
        if "rain" in lk:
            groups["rain"].append(k)
        elif "snow" in lk:
            groups["snow"].append(k)
        else:
            groups["other"].append(k)
    return groups


def mean_auc_for_keys(scene_auc_dict, keys):
    """
    scene_auc_dict: {scene_name: np.array([auc@5, auc@10, auc@20])}
    keys: list of scene_names
    """
    if len(keys) == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    return np.mean(np.stack([scene_auc_dict[k] for k in keys], axis=0), axis=0)


def mean_prec_for_keys(precs_dict, keys):
    """
    precs_dict: {scene_name: {thr: val, ...}}
    returns: {thr: mean_val}
    """
    if len(keys) == 0:
        return {}
    tmp = defaultdict(list)
    for k in keys:
        thr_map = precs_dict.get(k, {})
        for thr, val in thr_map.items():
            if val is None:
                continue
            tmp[thr].append(float(val))
    return {thr: float(np.mean(vals)) for thr, vals in tmp.items() if len(vals) > 0}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Relative Pose (Multi-scene Colmap/Blender dataset)")

    choose_method_arguments(parser)

    # ---- dataset root with multiple scenes ----
    parser.add_argument("--exp_name", type=str, default="MultiScene")
    parser.add_argument("--scenes_root_dir", type=str, default="/data6/liangyingping/tmp/dataset/WeatherGS/test_scenes",
                        help="root folder that contains multiple scene subfolders")

    parser.add_argument("--scene_type", type=str, default="Colmap", choices=["Colmap", "Blender"])
    parser.add_argument("--images_dir", type=str, default="weather_images",
                        help="(Colmap) images folder name under each scene (default: images)")
    parser.add_argument("--masks_dir", type=str, default="",
                        help="(Colmap) masks folder under each scene (optional)")
    parser.add_argument("--eval_split", action="store_true",
                        help="use llffhold split like your loader (train/test)")
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--use_test_cameras", action="store_true",
                        help="evaluate on test_cameras instead of train_cameras (only meaningful if --eval_split)")
    parser.add_argument("--pairing", type=str, default="adjacent", choices=["adjacent", "all"],
                        help="how to form pairs inside each scene")
    parser.add_argument("--max_pairs_per_scene", type=int, default=-1,
                        help="cap number of pairs per scene (-1 means all pairs)")

    # ---- output / eval ----
    parser.add_argument("--save_dir", type=str, default="./results_weather_pose/")
    parser.add_argument("--ransac_thres", type=float, default=1.5)
    parser.add_argument("--gaps", type=int, default=3)
    parser.add_argument("--e_name", type=str, default=None)
    parser.add_argument("--print_out", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_figs", action="store_true")
    parser.add_argument("--svg", action="store_true")

    args, remaining_args = parser.parse_known_args()
    add_method_arguments(parser, args.method)
    args = parser.parse_args()

    if args.e_name is not None:
        save_dir = osp.join(args.save_dir, args.e_name)
    else:
        save_dir = args.save_dir

    tt = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_relative_pose_multiscene(
            scenes_root_dir=Path(args.scenes_root_dir),
            method=args.method,
            exp_name=args.exp_name,
            ransac_thres=args.ransac_thres,
            print_out=args.print_out,
            save_dir=save_dir,
            save_figs=args.save_figs,
            debug=args.debug,
            args=args,
        )
    print(f"Elapsed time: {time.time() - tt}")
