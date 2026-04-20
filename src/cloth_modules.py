import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def load_rgb_and_mask(path: str):
    # 统一读取 RGBA/ RGB，并尽量从 alpha 得到可靠的衣服 mask
    im = Image.open(path)
    arr = np.array(im)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        rgb = arr[..., :3].astype(np.float32)
        alpha = arr[..., 3].astype(np.float32) / 255.0
        mask = alpha > 0.2
    else:
        rgb = arr[..., :3].astype(np.float32)
        mask = np.any(rgb < 250.0, axis=-1)
        alpha = mask.astype(np.float32)
    return rgb, alpha, mask


def composite_on_white(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = np.clip(alpha[..., None], 0.0, 1.0).astype(np.float32)
    out = rgb * a + 255.0 * (1.0 - a)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def estimate_bg_color(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    bg_mask = alpha < 0.05
    if bg_mask.sum() < 200:
        return np.array([255.0, 255.0, 255.0], dtype=np.float32)
    return np.median(rgb[bg_mask], axis=0).astype(np.float32)


def decontaminate_edges(rgb: np.ndarray, alpha: np.ndarray, bg_rgb: np.ndarray) -> np.ndarray:
    # 抠图常见问题：半透明边缘会混入背景（白边/黑边）。
    # 这里对 0<alpha<1 的像素做一次“反混合”近似，减少接缝处的脏边伪影。
    a = np.clip(alpha[..., None], 0.0, 1.0).astype(np.float32)
    m = (alpha > 0.05) & (alpha < 0.95)
    if m.sum() < 200:
        return rgb
    den = np.maximum(a, 1e-6)
    corrected = (rgb - bg_rgb[None, None, :] * (1.0 - a)) / den
    corrected = np.clip(corrected, 0.0, 255.0)
    out = rgb.copy()
    out[m] = corrected[m]
    return out


def erode_mask(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    m = mask.astype(bool)
    for _ in range(max(0, int(iterations))):
        up = np.zeros_like(m)
        down = np.zeros_like(m)
        left = np.zeros_like(m)
        right = np.zeros_like(m)
        up[1:] = m[:-1]
        down[:-1] = m[1:]
        left[:, 1:] = m[:, :-1]
        right[:, :-1] = m[:, 1:]
        ul = np.zeros_like(m)
        ur = np.zeros_like(m)
        dl = np.zeros_like(m)
        dr = np.zeros_like(m)
        ul[1:, 1:] = m[:-1, :-1]
        ur[1:, :-1] = m[:-1, 1:]
        dl[:-1, 1:] = m[1:, :-1]
        dr[:-1, :-1] = m[1:, 1:]
        m = m & up & down & left & right & ul & ur & dl & dr
    return m


def collar_cut_top_bump(rgb: np.ndarray, alpha: np.ndarray, mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) < 200:
        return rgb, alpha, mask
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    box = mask[y0 : y1 + 1, x0 : x1 + 1]
    h, w = box.shape
    if h < 32 or w < 32:
        return rgb, alpha, mask

    top_y = np.full((w,), h, dtype=np.int32)
    for x in range(w):
        col = box[:, x]
        if col.any():
            top_y[x] = int(np.argmax(col))

    edge = max(1, int(0.2 * w))
    ref = np.concatenate([top_y[:edge], top_y[w - edge :]])
    ref = ref[ref < h]
    if ref.size < 10:
        ref = top_y[top_y < h]
    if ref.size < 10:
        return rgb, alpha, mask

    baseline = float(np.median(ref))
    cutoff = int(max(0.0, baseline - 0.06 * h))
    if cutoff <= 0:
        return rgb, alpha, mask

    x_c0 = int(0.25 * w)
    x_c1 = int(0.75 * w)
    rm = np.zeros_like(box, dtype=bool)
    rm[:cutoff, x_c0:x_c1] = box[:cutoff, x_c0:x_c1]
    if rm.sum() < 50:
        return rgb, alpha, mask

    alpha_box = alpha[y0 : y1 + 1, x0 : x1 + 1]
    alpha_box[rm] = 0.0
    alpha[y0 : y1 + 1, x0 : x1 + 1] = alpha_box
    mask[y0 : y1 + 1, x0 : x1 + 1][rm] = False
    return rgb, alpha, mask


def collar_neckline_cut(rgb: np.ndarray, alpha: np.ndarray, mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if xs.size < 500:
        return rgb, alpha, mask

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    box = mask[y0 : y1 + 1, x0 : x1 + 1]
    h, w = box.shape
    if h < 64 or w < 64:
        return rgb, alpha, mask

    top_y = np.full((w,), h, dtype=np.int32)
    for x in range(w):
        col = box[:, x]
        if col.any():
            top_y[x] = int(np.argmax(col))

    valid = top_y < h
    if valid.sum() < max(20, int(0.25 * w)):
        return rgb, alpha, mask

    win = max(5, (w // 60) * 2 + 1)
    pad = win // 2
    top_y_pad = np.pad(top_y.astype(np.float32), (pad, pad), mode="edge")
    top_y_s = np.empty_like(top_y_pad)
    for i in range(top_y_pad.size - win + 1):
        top_y_s[i + pad] = np.median(top_y_pad[i : i + win])
    top_y_s = top_y_s[pad:-pad].astype(np.float32)

    left_band = top_y_s[: max(1, int(0.20 * w))]
    right_band = top_y_s[w - max(1, int(0.20 * w)) :]
    ref = np.concatenate([left_band[left_band < h], right_band[right_band < h]])
    if ref.size < 10:
        ref = top_y_s[top_y_s < h]
    if ref.size < 10:
        return rgb, alpha, mask

    shoulder_y = float(np.percentile(ref, 35.0))
    center_y = float(np.percentile(top_y_s[int(0.40 * w) : int(0.60 * w)], 25.0))
    if center_y > shoulder_y + 0.06 * h:
        return rgb, alpha, mask

    xL = int(0.22 * w)
    xR = int(0.78 * w)
    yL = float(np.percentile(top_y_s[max(0, xL - 5) : min(w, xL + 6)], 35.0))
    yR = float(np.percentile(top_y_s[max(0, xR - 5) : min(w, xR + 6)], 35.0))
    yS = float(0.5 * (yL + yR))

    neck_drop = float(np.clip(0.10 * h, 8.0, 0.22 * h))
    yC = float(min(h - 1, yS + neck_drop))
    c = 0.5 * (xL + xR)
    denom = (xL - c) ** 2 + 1e-6
    a = (yL - yC) / denom

    xs_idx = np.arange(w, dtype=np.float32)
    curve = a * (xs_idx - c) ** 2 + yC
    curve = np.clip(curve, 0.0, float(h - 1))

    x0c = int(0.28 * w)
    x1c = int(0.72 * w)
    feather = int(np.clip(0.03 * h, 4.0, 16.0))

    alpha_box = alpha[y0 : y1 + 1, x0 : x1 + 1].copy()
    for xi in range(x0c, x1c):
        y_cut = int(curve[xi])
        if y_cut <= 0:
            continue
        col_mask = box[:, xi]
        if not col_mask.any():
            continue
        alpha_box[:y_cut, xi] = 0.0
        if feather > 1 and y_cut + feather < h:
            ramp = np.linspace(0.0, 1.0, feather, dtype=np.float32)
            alpha_box[y_cut : y_cut + feather, xi] *= ramp

    alpha[y0 : y1 + 1, x0 : x1 + 1] = alpha_box
    mask[y0 : y1 + 1, x0 : x1 + 1] = alpha_box > 0.2
    return rgb, alpha, mask


def collar_neckline_edge(
    rgb: np.ndarray,
    alpha: np.ndarray,
    mask: np.ndarray,
    ymax_scale: float,
    depth_bonus: float,
    depth_penalty: float,
    slope_strength: float,
    slope_power: float,
):
    # 领口裁剪（neckline_edge）：在衣服上半部分的 ROI 内做 Sobel 边缘 + 动态规划找一条领口曲线。
    # 这个方法比纯 mask 拟合更鲁棒：即使 mask 被后领“填平”，RGB 边缘仍可能存在。
    ys, xs = np.nonzero(mask)
    if xs.size < 800:
        return rgb, alpha, mask, None

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    box_m = mask[y0 : y1 + 1, x0 : x1 + 1]
    box_a = alpha[y0 : y1 + 1, x0 : x1 + 1].copy()
    box_rgb = rgb[y0 : y1 + 1, x0 : x1 + 1]
    h, w = box_m.shape
    if h < 96 or w < 96:
        return rgb, alpha, mask, None

    top_y = np.full((w,), h, dtype=np.int32)
    for x in range(w):
        col = box_m[:, x]
        if col.any():
            top_y[x] = int(np.argmax(col))
    valid = top_y < h
    if valid.sum() < max(40, int(0.35 * w)):
        return rgb, alpha, mask, None

    win = max(7, (w // 50) * 2 + 1)
    pad = win // 2
    top_y_pad = np.pad(top_y.astype(np.float32), (pad, pad), mode="edge")
    top_y_s = np.empty_like(top_y_pad)
    for i in range(top_y_pad.size - win + 1):
        top_y_s[i + pad] = np.median(top_y_pad[i : i + win])
    top_y_s = top_y_s[pad:-pad]

    # 领口起点：用衣服上边界 top_y(x) 的两个“最高点”（y 最小）
    # 作为路径起点/终点。这样比固定比例 xL/xR 更贴合衣服实际轮廓。
    x_search_l0 = int(0.05 * w)
    x_search_l1 = int(0.45 * w)
    x_search_r0 = int(0.55 * w)
    x_search_r1 = int(0.95 * w)
    if x_search_r0 <= x_search_l1 + 10:
        return rgb, alpha, mask, None

    top_valid = top_y_s.copy()
    top_valid[top_valid >= h] = np.nan
    left_seg = top_valid[x_search_l0:x_search_l1]
    right_seg = top_valid[x_search_r0:x_search_r1]
    if np.all(np.isnan(left_seg)) or np.all(np.isnan(right_seg)):
        return rgb, alpha, mask, None

    peak_xL = int(x_search_l0 + int(np.nanargmin(left_seg)))
    peak_xR = int(x_search_r0 + int(np.nanargmin(right_seg)))
    if peak_xR <= peak_xL + 20:
        return rgb, alpha, mask, None
    peak_yL = float(top_y_s[peak_xL])
    peak_yR = float(top_y_s[peak_xR])

    left_ref = top_y_s[: max(1, int(0.20 * w))]
    right_ref = top_y_s[w - max(1, int(0.20 * w)) :]
    ref = np.concatenate([left_ref[left_ref < h], right_ref[right_ref < h]])
    if ref.size < 10:
        ref = top_y_s[top_y_s < h]
    if ref.size < 10:
        return rgb, alpha, mask, None

    shoulder_y = float(np.percentile(ref, 35.0))
    # 收紧纵向 ROI，避免路径下探到胸前花纹区域
    # y_min 至少要覆盖两侧峰值点
    y_min = int(max(0.0, min(peak_yL, peak_yR, shoulder_y) - 0.02 * h))
    # 纵向范围放宽一些，否则容易把“后领口那条更高更平的弧线”当成最优路径
    y_max = int(min(h - 1, shoulder_y + float(ymax_scale) * h))
    dx = float(peak_xR - peak_xL)
    dy = float(peak_yR - peak_yL)
    dist = float(np.sqrt(dx * dx + dy * dy))
    circle_mid_y = 0.5 * float(peak_yL + peak_yR)
    circle_r = 0.5 * dist
    # 硬约束：前领最低点（在两端点中点处）必须落在“以两端点为直径的圆”外侧。
    # 在图像坐标里等价于：中点列的 y 必须 >= mid_y + r（向下为正）。
    required_mid_y = float(circle_mid_y + circle_r)
    y_max = int(min(h - 1, max(float(y_max), required_mid_y + 0.10 * h)))
    if y_max <= y_min + 10:
        return rgb, alpha, mask, None

    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gray = (0.299 * box_rgb[..., 0] + 0.587 * box_rgb[..., 1] + 0.114 * box_rgb[..., 2]).astype(np.float32)
    gray = gray / 255.0

    p = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
    sx = (
        gx[0, 0] * p[:-2, :-2]
        + gx[0, 1] * p[:-2, 1:-1]
        + gx[0, 2] * p[:-2, 2:]
        + gx[1, 0] * p[1:-1, :-2]
        + gx[1, 1] * p[1:-1, 1:-1]
        + gx[1, 2] * p[1:-1, 2:]
        + gx[2, 0] * p[2:, :-2]
        + gx[2, 1] * p[2:, 1:-1]
        + gx[2, 2] * p[2:, 2:]
    )
    sy = (
        gy[0, 0] * p[:-2, :-2]
        + gy[0, 1] * p[:-2, 1:-1]
        + gy[0, 2] * p[:-2, 2:]
        + gy[1, 0] * p[1:-1, :-2]
        + gy[1, 1] * p[1:-1, 1:-1]
        + gy[1, 2] * p[1:-1, 2:]
        + gy[2, 0] * p[2:, :-2]
        + gy[2, 1] * p[2:, 1:-1]
        + gy[2, 2] * p[2:, 2:]
    )
    edge = np.sqrt(sx * sx + sy * sy)

    roi = np.zeros_like(edge, dtype=bool)
    roi[y_min : y_max + 1, peak_xL : peak_xR + 1] = True
    roi &= box_a > 0.05
    if roi.sum() < 200:
        return rgb, alpha, mask, None

    e = edge.copy()
    e[~roi] = 0.0
    e = e / (e.max() + 1e-6)

    # DP 路径只在两侧峰值之间搜索，并强制起点/终点落在峰值 y
    cols = np.arange(peak_xL, peak_xR + 1, dtype=np.int32)
    H = y_max - y_min + 1
    W = cols.size
    if W < 30:
        return rgb, alpha, mask, None

    y_grid = np.arange(y_min, y_max + 1, dtype=np.int32)
    score = e[y_grid[:, None], cols[None, :]].astype(np.float32)
    # 让路径在中间段更倾向于下探（更像“前领口”），而不是贴着上边缘走成“后领口”
    depth = np.maximum(0.0, (y_grid.astype(np.float32) - shoulder_y) / max(1.0, float(ymax_scale) * h))
    depth = np.clip(depth, 0.0, 1.0)
    col_idx = np.arange(W, dtype=np.float32)
    center = 0.5 * float(W - 1)
    center_w = 1.0 - np.abs(col_idx - center) / (center + 1e-6)
    center_w = np.clip(center_w, 0.0, 1.0)
    score = score + float(depth_bonus) * depth[:, None] * center_w[None, :] - float(depth_penalty) * depth[:, None]

    lam = 0.22
    max_step = max(4, int(0.04 * h))
    cost = np.full((H,), -1e9, dtype=np.float32)
    back = np.full((H, W), -1, dtype=np.int32)
    # 起点强约束：只允许从左峰值 y 开始
    start_yi = int(np.clip(round(peak_yL) - y_min, 0, H - 1))
    cost[start_yi] = score[start_yi, 0]
    mid_j = W // 2
    required_mid_yi = int(np.ceil(required_mid_y) - y_min)
    required_mid_yi = max(0, min(required_mid_yi, H - 1))

    center0 = int(0.40 * W)
    center1 = int(0.60 * W)
    center0 = max(0, min(center0, W - 1))
    center1 = max(center0 + 1, min(center1, W))
    center_best = np.argmax(score[:, center0:center1], axis=0) + y_min
    target_mid_y = float(np.percentile(center_best, 85.0))
    target_mid_y = max(target_mid_y, shoulder_y + 0.10 * h)
    target_mid_y = min(target_mid_y, float(y_max))

    drop_left = max(0.0, target_mid_y - peak_yL)
    drop_right = max(0.0, target_mid_y - peak_yR)

    def build_targets(total_drop: float, steps: int, sign: float) -> np.ndarray:
        if steps <= 0:
            return np.zeros((0,), dtype=np.float32)
        t = np.linspace(0.0, 1.0, steps, dtype=np.float32)
        weights = (1.0 - t) ** float(slope_power)
        weights = np.clip(weights, 1e-3, None)
        weights = weights / weights.sum()
        return sign * (total_drop * weights)

    left_targets = build_targets(drop_left, mid_j, sign=1.0)
    right_targets = build_targets(drop_right, W - 1 - mid_j, sign=-1.0)

    for j in range(1, W):
        new_cost = np.full((H,), -1e9, dtype=np.float32)
        if j <= mid_j:
            target_step = float(left_targets[j - 1]) if left_targets.size >= j else 0.0
        else:
            jj = j - (mid_j + 1)
            target_step = float(right_targets[jj]) if right_targets.size > jj else 0.0
        for yi in range(H):
            if j == mid_j and yi < required_mid_yi:
                continue
            lo = max(0, yi - max_step)
            hi = min(H, yi + max_step + 1)
            prev_ids = np.arange(lo, hi, dtype=np.int32)
            # 关键约束：
            # 左半段（到中点前）只能“向右下”（y 递增或不变）
            # 右半段（中点后）只能“向右上”（y 递减或不变）
            if j <= mid_j:
                prev_ids = prev_ids[prev_ids <= yi]
            else:
                prev_ids = prev_ids[prev_ids >= yi]
            if prev_ids.size == 0:
                continue
            prev = cost[prev_ids]
            step = float(yi) - prev_ids.astype(np.float32)
            smooth = prev - lam * (step * step)
            slope = -float(slope_strength) * ((step - target_step) ** 2)
            cand = smooth + slope
            k = int(np.argmax(cand))
            best_prev = int(prev_ids[k])
            new_cost[yi] = score[yi, j] + cand[k]
            back[yi, j] = best_prev
        cost = new_cost

    # 终点强约束：尽量在右峰值 y 结束（允许 1-2 像素误差）
    end_target = int(np.clip(round(peak_yR) - y_min, 0, H - 1))
    end_lo = max(0, end_target - 2)
    end_hi = min(H, end_target + 3)
    yi = int(end_lo + int(np.argmax(cost[end_lo:end_hi])))
    if cost[yi] < 0.05 * W:
        return rgb, alpha, mask, None

    path_y = np.zeros((W,), dtype=np.int32)
    path_y[-1] = yi
    for j in range(W - 1, 0, -1):
        yi = back[yi, j]
        if yi < 0:
            return rgb, alpha, mask, None
        path_y[j - 1] = yi
    curve_y = (path_y + y_min).astype(np.int32)

    # 水平扣除范围：主要在两个峰值之间（略微留边，避免肩部被误切）
    margin = int(max(6.0, 0.04 * (peak_xR - peak_xL)))
    mid_band = slice(int(peak_xL + margin), int(peak_xR - margin))

    feather = int(np.clip(0.04 * h, 6.0, 20.0))
    cutline_points: list[tuple[int, int]] = []
    for k, x in enumerate(cols):
        y_cut = int(curve_y[k])
        if y_cut <= 0:
            continue
        # 只在 mid_band 内真正扣除
        if x < mid_band.start or x >= mid_band.stop:
            continue
        cutline_points.append((int(x0 + x), int(y0 + y_cut)))
        box_a[:y_cut, x] = 0.0
        if feather > 1 and y_cut + feather < h:
            ramp = np.linspace(0.0, 1.0, feather, dtype=np.float32)
            box_a[y_cut : y_cut + feather, x] *= ramp

    alpha[y0 : y1 + 1, x0 : x1 + 1] = box_a
    mask[y0 : y1 + 1, x0 : x1 + 1] = box_a > 0.2
    debug = {
        "neckline_cutline": cutline_points,
        "neckline_peaks": [(int(x0 + peak_xL), int(y0 + round(peak_yL))), (int(x0 + peak_xR), int(y0 + round(peak_yR)))],
        "neckline_required_mid_y": int(y0 + round(required_mid_y)),
    }
    return rgb, alpha, mask, debug


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0, mask.shape[1] - 1
    return int(xs.min()), int(xs.max())


def seam_fix_feather_stats(
    front_rgb: np.ndarray,
    front_mask: np.ndarray,
    front_alpha: np.ndarray,
    back_rgb: np.ndarray,
    back_mask: np.ndarray,
    back_alpha: np.ndarray,
    band_w: int,
):
    eps = 1e-6
    h, w = front_mask.shape
    x0_f, x1_f = bbox_from_mask(front_mask)
    x0_b, x1_b = bbox_from_mask(back_mask)
    x0 = max(0, min(x0_f, x0_b))
    x1 = min(w - 1, max(x1_f, x1_b))
    band_w = max(4, min(band_w, (x1 - x0 + 1) // 3 if (x1 - x0 + 1) >= 12 else 4))

    def robust_mean_std(pixels: np.ndarray):
        lo = np.percentile(pixels, 5.0, axis=0)
        hi = np.percentile(pixels, 95.0, axis=0)
        clipped = np.clip(pixels, lo, hi)
        mean = clipped.mean(axis=0)
        std = clipped.std(axis=0) + eps
        return mean, std

    def stat_match(src_pixels: np.ndarray, tgt_pixels: np.ndarray):
        src_mean, src_std = robust_mean_std(src_pixels)
        tgt_mean, tgt_std = robust_mean_std(tgt_pixels)
        return src_mean, src_std, tgt_mean, tgt_std

    def apply_match(region_rgb: np.ndarray, region_mask: np.ndarray, src_mean, src_std, tgt_mean, tgt_std):
        out = region_rgb.copy()
        if region_mask.sum() == 0:
            return out
        adj = (region_rgb - src_mean) * (tgt_std / src_std) + tgt_mean
        adj = np.clip(adj, 0.0, 255.0)
        out[region_mask] = adj[region_mask]
        return out

    def fix_side(xs: slice, weight_1d: np.ndarray):
        nonlocal back_rgb
        f_band = front_rgb[:, xs, :]
        b_band = back_rgb[:, xs, :]
        f_m = front_mask[:, xs] & (front_alpha[:, xs] > 0.85)
        b_m = back_mask[:, xs] & (back_alpha[:, xs] > 0.85)
        common = erode_mask(f_m & b_m, iterations=1)
        if common.sum() < 150:
            return

        f_px = f_band[common]
        b_px = b_band[common]
        src_mean, src_std, tgt_mean, tgt_std = stat_match(b_px, f_px)
        b_band_adj = apply_match(b_band, back_mask[:, xs], src_mean, src_std, tgt_mean, tgt_std)

        # 只把 back 往 front 靠拢，避免把 front 的颜色也“拉花”。
        ww = np.tile(weight_1d[None, :, None], (h, 1, 1)).astype(np.float32)
        ww = ww * back_alpha[:, xs][:, :, None].astype(np.float32)
        use = back_mask[:, xs]
        back_rgb[:, xs, :][use] = (1.0 - ww[use]) * b_band[use] + ww[use] * b_band_adj[use]

    left_xs = slice(x0, min(w, x0 + band_w))
    left_w = np.linspace(1.0, 0.0, left_xs.stop - left_xs.start, dtype=np.float32)
    fix_side(left_xs, left_w)

    right_xs = slice(max(0, x1 - band_w + 1), x1 + 1)
    right_w = np.linspace(0.0, 1.0, right_xs.stop - right_xs.start, dtype=np.float32)[::-1]
    fix_side(right_xs, right_w)

    return front_rgb, back_rgb


def seam_fix_side_views(
    front_rgb: np.ndarray,
    front_mask: np.ndarray,
    front_alpha: np.ndarray,
    back_rgb: np.ndarray,
    back_mask: np.ndarray,
    back_alpha: np.ndarray,
    left_rgb: np.ndarray,
    left_mask: np.ndarray,
    left_alpha: np.ndarray,
    right_rgb: np.ndarray,
    right_mask: np.ndarray,
    right_alpha: np.ndarray,
    band_w: int,
):
    # 接缝一致性（side_views）：利用 Wonder3D 的 left/right 视图作为侧缝处纹理参考，
    # 把侧视图中心竖条融合到 front/back 的左右边缘条带，减少正背交界散乱花纹。
    eps = 1e-6
    h, w = front_mask.shape
    x0_f, x1_f = bbox_from_mask(front_mask)
    x0_b, x1_b = bbox_from_mask(back_mask)
    x0 = max(0, min(x0_f, x0_b))
    x1 = min(w - 1, max(x1_f, x1_b))
    band_w = max(4, min(band_w, (x1 - x0 + 1) // 3 if (x1 - x0 + 1) >= 12 else 4))

    def robust_mean_std(pixels: np.ndarray):
        lo = np.percentile(pixels, 5.0, axis=0)
        hi = np.percentile(pixels, 95.0, axis=0)
        clipped = np.clip(pixels, lo, hi)
        mean = clipped.mean(axis=0)
        std = clipped.std(axis=0) + eps
        return mean, std

    def apply_match(region_rgb: np.ndarray, region_mask: np.ndarray, src_mean, src_std, tgt_mean, tgt_std):
        out = region_rgb.copy()
        if region_mask.sum() == 0:
            return out
        adj = (region_rgb - src_mean) * (tgt_std / src_std) + tgt_mean
        adj = np.clip(adj, 0.0, 255.0)
        out[region_mask] = adj[region_mask]
        return out

    def center_strip(src_rgb: np.ndarray, src_mask: np.ndarray, src_alpha: np.ndarray):
        sx0, sx1 = bbox_from_mask(src_mask)
        mid = int(0.5 * (sx0 + sx1))
        half = band_w // 2
        s0 = max(0, mid - half)
        s1 = min(w, s0 + band_w)
        s0 = max(0, s1 - band_w)
        strip_rgb = src_rgb[:, s0:s1, :].copy()
        strip_a = src_alpha[:, s0:s1].copy()
        strip_m = src_mask[:, s0:s1].copy()
        return strip_rgb, strip_a, strip_m

    def estimate_torso_rows(union_mask: np.ndarray) -> np.ndarray:
        widths = np.zeros((h,), dtype=np.float32)
        for y in range(h):
            xs = np.nonzero(union_mask[y])[0]
            if xs.size >= 10:
                widths[y] = float(xs.max() - xs.min() + 1)
        valid = widths > 0
        if valid.sum() < 20:
            return np.ones((h,), dtype=bool)
        med = float(np.median(widths[valid]))
        lo = 0.75 * med
        hi = 1.30 * med
        torso = (widths >= lo) & (widths <= hi)
        # 经验过滤：避开最上/最下边缘（领口/下摆/袖口更容易乱）
        torso[: int(0.10 * h)] = False
        torso[int(0.92 * h) :] = False
        return torso

    def transfer_to_band(
        band_slice: slice,
        weight_1d: np.ndarray,
        strip_rgb: np.ndarray,
        strip_a: np.ndarray,
        strip_m: np.ndarray,
    ):
        nonlocal front_rgb, back_rgb

        f_band = front_rgb[:, band_slice, :]
        b_band = back_rgb[:, band_slice, :]
        f_m = front_mask[:, band_slice] & (front_alpha[:, band_slice] > 0.85)
        b_m = back_mask[:, band_slice] & (back_alpha[:, band_slice] > 0.85)
        s_m = strip_m & (strip_a > 0.85)

        torso_rows = estimate_torso_rows(front_mask | back_mask)
        torso_rows_2d = torso_rows[:, None]

        ref_mask = erode_mask((f_m | b_m) & s_m, iterations=1)
        if ref_mask.sum() < 150:
            return

        ref_pixels = np.concatenate([f_band[ref_mask], b_band[ref_mask]], axis=0)
        src_pixels = strip_rgb[ref_mask]
        if ref_pixels.shape[0] < 50 or src_pixels.shape[0] < 50:
            return

        src_mean, src_std = robust_mean_std(src_pixels)
        tgt_mean, tgt_std = robust_mean_std(ref_pixels)
        strip_adj = apply_match(strip_rgb, strip_m, src_mean, src_std, tgt_mean, tgt_std)

        ww = np.tile(weight_1d[None, :, None], (h, 1, 1)).astype(np.float32)
        ww = ww * strip_a[:, :, None].astype(np.float32)

        # 只在“躯干行”做融合，尽量避免袖口/领口等细节花纹被破坏
        use_f = front_mask[:, band_slice] & strip_m & torso_rows_2d
        use_b = back_mask[:, band_slice] & strip_m & torso_rows_2d
        ww = ww * torso_rows_2d[:, :, None].astype(np.float32)

        front_rgb[:, band_slice, :][use_f] = (1.0 - ww[use_f]) * f_band[use_f] + ww[use_f] * strip_adj[use_f]
        back_rgb[:, band_slice, :][use_b] = (1.0 - ww[use_b]) * b_band[use_b] + ww[use_b] * strip_adj[use_b]

    left_strip_rgb, left_strip_a, left_strip_m = center_strip(left_rgb, left_mask, left_alpha)
    right_strip_rgb, right_strip_a, right_strip_m = center_strip(right_rgb, right_mask, right_alpha)

    left_band = slice(x0, min(w, x0 + band_w))
    left_w = np.linspace(1.0, 0.0, left_band.stop - left_band.start, dtype=np.float32)
    transfer_to_band(left_band, left_w, left_strip_rgb, left_strip_a, left_strip_m)

    right_band = slice(max(0, x1 - band_w + 1), x1 + 1)
    right_w = np.linspace(0.0, 1.0, right_band.stop - right_band.start, dtype=np.float32)[::-1]
    transfer_to_band(right_band, right_w, right_strip_rgb, right_strip_a, right_strip_m)

    return front_rgb, back_rgb


def apply_modules_to_pair(
    sem_front_path: str,
    sem_back_path: str,
    sem_left_path: str,
    sem_right_path: str,
    collar_module: str,
    seam_module: str,
    seam_band_width: int,
    neckline_edge_ymax_scale: float,
    neckline_edge_depth_bonus: float,
    neckline_edge_depth_penalty: float,
    neckline_edge_slope_strength: float,
    neckline_edge_slope_power: float,
    neckline_manual_x: float,
    neckline_manual_y: float,
):
    front_rgb, front_a, front_m = load_rgb_and_mask(sem_front_path)
    back_rgb, back_a, back_m = load_rgb_and_mask(sem_back_path)
    debug: dict[str, object] = {}
    # 无论是否开启接缝模块，都先做边缘去污染：它本身就能显著降低接缝处乱纹
    bg_front = estimate_bg_color(front_rgb, front_a)
    bg_back = estimate_bg_color(back_rgb, back_a)
    front_rgb = decontaminate_edges(front_rgb, front_a, bg_front)
    back_rgb = decontaminate_edges(back_rgb, back_a, bg_back)

    left_rgb = left_a = left_m = None
    right_rgb = right_a = right_m = None
    if sem_left_path:
        left_rgb, left_a, left_m = load_rgb_and_mask(sem_left_path)
        left_rgb = decontaminate_edges(left_rgb, left_a, estimate_bg_color(left_rgb, left_a))
    if sem_right_path:
        right_rgb, right_a, right_m = load_rgb_and_mask(sem_right_path)
        right_rgb = decontaminate_edges(right_rgb, right_a, estimate_bg_color(right_rgb, right_a))

    if collar_module == "cut_top_bump":
        front_rgb, front_a, front_m = collar_cut_top_bump(front_rgb, front_a, front_m)
    if collar_module == "neckline_cut":
        front_rgb, front_a, front_m = collar_neckline_cut(front_rgb, front_a, front_m)
    if collar_module == "neckline_edge":
        front_rgb, front_a, front_m, info = collar_neckline_edge(
            front_rgb,
            front_a,
            front_m,
            neckline_edge_ymax_scale,
            neckline_edge_depth_bonus,
            neckline_edge_depth_penalty,
            neckline_edge_slope_strength,
            neckline_edge_slope_power,
        )
        if isinstance(info, dict):
            debug.update(info)
    if collar_module == "manual_point":
        # 交互式领口：用户在网页上点一个“前领最低点”，与两侧最高点拟合二次曲线后扣除
        ys, xs = np.nonzero(front_m)
        if xs.size >= 800 and 0.0 <= float(neckline_manual_x) <= 1.0 and 0.0 <= float(neckline_manual_y) <= 1.0:
            full_h, full_w = front_m.shape
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            box = front_m[y0 : y1 + 1, x0 : x1 + 1]
            h, w = box.shape
            top_y = np.full((w,), h, dtype=np.int32)
            for x in range(w):
                col = box[:, x]
                if col.any():
                    top_y[x] = int(np.argmax(col))
            top_y = top_y.astype(np.float32)
            top_y[top_y >= h] = np.nan
            left_seg = top_y[: max(1, int(0.45 * w))]
            right_seg = top_y[max(1, int(0.55 * w)) :]
            if not np.all(np.isnan(left_seg)) and not np.all(np.isnan(right_seg)):
                peak_xL = int(np.nanargmin(left_seg))
                peak_xR = int(max(1, int(0.55 * w)) + int(np.nanargmin(right_seg)))
                peak_yL = float(top_y[peak_xL])
                peak_yR = float(top_y[peak_xR])
                if peak_xR > peak_xL + 20 and not np.isnan(peak_yL) and not np.isnan(peak_yR):
                    u = float(neckline_manual_x)
                    v = float(neckline_manual_y)
                    # 用户点位按“整张 semantic front 图”的归一化坐标传入，
                    # 先映射到 full 图，再转成 bbox 局部坐标，避免因为 bbox 裁切导致点位系统性下移。
                    user_abs_x = int(round(u * float(full_w - 1)))
                    user_abs_y = int(round(v * float(full_h - 1)))
                    px = int(np.clip(user_abs_x - x0, peak_xL, peak_xR))
                    py = int(np.clip(user_abs_y - y0, int(min(peak_yL, peak_yR)), h - 1))

                    A = np.array(
                        [
                            [float(peak_xL) ** 2, float(peak_xL), 1.0],
                            [float(px) ** 2, float(px), 1.0],
                            [float(peak_xR) ** 2, float(peak_xR), 1.0],
                        ],
                        dtype=np.float32,
                    )
                    b = np.array([float(peak_yL), float(py), float(peak_yR)], dtype=np.float32)
                    try:
                        coef = np.linalg.solve(A, b)
                    except Exception:
                        coef = None
                    if coef is not None:
                        a, bb, c = float(coef[0]), float(coef[1]), float(coef[2])
                        # manual_point 以“用户点位”作为硬约束，feather 太大时视觉边界会比点位更靠下
                        # 所以这里用更小的 feather，并对 y_cut 做一点补偿，让可见边界更贴近用户点。
                        feather = int(np.clip(0.02 * h, 2.0, 10.0))
                        bias = int(round(0.2 * feather))
                        cutline_points: list[tuple[int, int]] = []
                        alpha_box = front_a[y0 : y1 + 1, x0 : x1 + 1].copy()
                        for x in range(peak_xL, peak_xR + 1):
                            y_cut = int(np.clip(a * x * x + bb * x + c, 0.0, float(h - 1)))
                            y_cut = max(0, y_cut - bias)
                            cutline_points.append((int(x0 + x), int(y0 + y_cut)))
                            alpha_box[:y_cut, x] = 0.0
                            if feather > 1 and y_cut + feather < h:
                                ramp = np.linspace(0.0, 1.0, feather, dtype=np.float32)
                                alpha_box[y_cut : y_cut + feather, x] *= ramp
                        front_a[y0 : y1 + 1, x0 : x1 + 1] = alpha_box
                        front_m[y0 : y1 + 1, x0 : x1 + 1] = alpha_box > 0.2
                        debug["neckline_cutline"] = cutline_points
                        debug["neckline_peaks"] = [(int(x0 + peak_xL), int(y0 + round(peak_yL))), (int(x0 + peak_xR), int(y0 + round(peak_yR)))]
                        debug["neckline_user_point"] = (int(x0 + px), int(y0 + py))

    if seam_module == "side_views":
        if left_rgb is not None and right_rgb is not None:
            front_rgb, back_rgb = seam_fix_side_views(
                front_rgb,
                front_m,
                front_a,
                back_rgb,
                back_m,
                back_a,
                left_rgb,
                left_m,
                left_a,
                right_rgb,
                right_m,
                right_a,
                seam_band_width,
            )
        else:
            seam_module = "feather_stats"

    if seam_module == "feather_stats":
        front_rgb, back_rgb = seam_fix_feather_stats(
            front_rgb,
            front_m,
            front_a,
            back_rgb,
            back_m,
            back_a,
            seam_band_width,
        )

    return (front_rgb, front_a), (back_rgb, back_a), debug


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items-json", required=True)
    parser.add_argument("--collar-module", choices=["none", "cut_top_bump", "neckline_cut", "neckline_edge", "manual_point"], default="none")
    parser.add_argument("--seam-module", choices=["none", "feather_stats", "side_views"], default="feather_stats")
    parser.add_argument("--seam-band-width", type=int, default=24)
    parser.add_argument("--invert-vton-cloth-names", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--neckline-edge-ymax-scale", type=float, default=0.40)
    parser.add_argument("--neckline-edge-depth-bonus", type=float, default=0.28)
    parser.add_argument("--neckline-edge-depth-penalty", type=float, default=0.04)
    parser.add_argument("--neckline-edge-slope-strength", type=float, default=0.12)
    parser.add_argument("--neckline-edge-slope-power", type=float, default=1.6)
    parser.add_argument("--neckline-manual-x", type=float, default=-1.0)
    parser.add_argument("--neckline-manual-y", type=float, default=-1.0)
    parser.add_argument("--debug-dir", default="")
    args = parser.parse_args()

    items = json.loads(Path(args.items_json).read_text(encoding="utf-8"))

    for it in items:
        (f_rgb, f_a), (b_rgb, b_a), debug = apply_modules_to_pair(
            it["src_sem_front"],
            it["src_sem_back"],
            it.get("src_sem_left", ""),
            it.get("src_sem_right", ""),
            args.collar_module,
            args.seam_module,
            int(args.seam_band_width),
            float(args.neckline_edge_ymax_scale),
            float(args.neckline_edge_depth_bonus),
            float(args.neckline_edge_depth_penalty),
            float(args.neckline_edge_slope_strength),
            float(args.neckline_edge_slope_power),
            float(args.neckline_manual_x),
            float(args.neckline_manual_y),
        )

        dst_front = it["dst_front"]
        dst_back = it["dst_back"]
        os.makedirs(os.path.dirname(dst_front), exist_ok=True)

        if args.invert_vton_cloth_names:
            Image.fromarray(composite_on_white(b_rgb, b_a)).save(dst_front, quality=95)
            Image.fromarray(composite_on_white(f_rgb, f_a)).save(dst_back, quality=95)
        else:
            Image.fromarray(composite_on_white(f_rgb, f_a)).save(dst_front, quality=95)
            Image.fromarray(composite_on_white(b_rgb, b_a)).save(dst_back, quality=95)

        if args.debug_dir:
            debug_dir = Path(args.debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
            base = Path(dst_front).stem.replace("_front", "").replace("_back", "")
            out_front = debug_dir / f"{base}_vton_input_front.jpg"
            out_back = debug_dir / f"{base}_vton_input_back.jpg"
            Image.fromarray(composite_on_white(f_rgb, f_a)).save(out_front, quality=95)
            Image.fromarray(composite_on_white(b_rgb, b_a)).save(out_back, quality=95)
            if "neckline_cutline" in debug:
                line = debug["neckline_cutline"]
                if isinstance(line, list) and len(line) >= 2:
                    im = Image.fromarray(composite_on_white(f_rgb, f_a))
                    draw = ImageDraw.Draw(im)
                    draw.line(line, fill=(255, 0, 0), width=3, joint="curve")
                    peaks = debug.get("neckline_peaks")
                    if isinstance(peaks, list) and len(peaks) == 2:
                        for (px, py) in peaks:
                            r = 6
                            draw.ellipse((px - r, py - r, px + r, py + r), outline=(0, 255, 0), width=3)
                    up = debug.get("neckline_user_point")
                    if isinstance(up, tuple) and len(up) == 2:
                        px, py = int(up[0]), int(up[1])
                        r = 6
                        draw.ellipse((px - r, py - r, px + r, py + r), outline=(255, 255, 0), width=3)
                    req = debug.get("neckline_required_mid_y")
                    if isinstance(req, int):
                        y = req
                        draw.line([(0, y), (im.size[0] - 1, y)], fill=(0, 0, 255), width=2)
                    im.save(debug_dir / f"{base}_vton_input_front_cutline.jpg", quality=95)


if __name__ == "__main__":
    main()
