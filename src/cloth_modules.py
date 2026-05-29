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


def _pick_plateau_edge_x(top_y: np.ndarray, x_start: int, x_end: int, h: int, side: str) -> int | None:
    seg = top_y[x_start:x_end].astype(np.float32)
    if seg.size == 0:
        return None
    ymin = float(np.nanmin(seg))
    if not np.isfinite(ymin):
        return None
    tol = float(max(2.0, 0.01 * h))
    ids = np.where(np.isfinite(seg) & (seg <= ymin + tol))[0]
    if ids.size == 0:
        return None
    splits = np.where(np.diff(ids) > 1)[0]
    runs = []
    s = int(ids[0])
    for k in splits:
        e = int(ids[k])
        runs.append((s, e))
        s = int(ids[k + 1])
    runs.append((s, int(ids[-1])))
    min_len = max(3, int(round(0.03 * float(seg.size))))
    good = [r for r in runs if (r[1] - r[0] + 1) >= min_len]
    if not good:
        good = runs
    if side == "left":
        r = min(good, key=lambda t: t[0])
        return int(x_start + r[0])
    if side == "right":
        r = max(good, key=lambda t: t[1])
        return int(x_start + r[1])
    return None


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

    def mid_patch(src_rgb: np.ndarray, src_mask: np.ndarray, src_alpha: np.ndarray, patch_w: int):
        sx0, sx1 = bbox_from_mask(src_mask)
        mid = int(0.5 * (sx0 + sx1))
        half = patch_w // 2
        s0 = max(0, mid - half)
        s1 = min(w, s0 + patch_w)
        s0 = max(0, s1 - patch_w)
        patch_rgb = src_rgb[:, s0:s1, :].copy()
        patch_a = src_alpha[:, s0:s1].copy()
        patch_m = src_mask[:, s0:s1].copy()
        return patch_rgb, patch_a, patch_m

    def estimate_torso_rows(union_mask: np.ndarray) -> np.ndarray:
        widths = np.zeros((h,), dtype=np.float32)
        for y in range(h):
            xs = np.nonzero(union_mask[y])[0]
            if xs.size >= 10:
                widths[y] = float(xs.max() - xs.min() + 1)
        valid = widths > 0
        if valid.sum() < 20:
            return np.ones((h,), dtype=bool)

        y0 = int(0.25 * h)
        y1 = int(0.75 * h)
        core = valid.copy()
        core[:y0] = False
        core[y1:] = False
        base = widths[core] if core.sum() >= 10 else widths[valid]
        med = float(np.median(base))

        lo = 0.70 * med
        hi = 1.15 * med
        torso = (widths >= lo) & (widths <= hi)
        # 经验过滤：避开更容易出现异常的区域（领口/袖口/下摆）
        torso[: int(0.15 * h)] = False
        torso[int(0.88 * h) :] = False
        return torso

    def _gray(rgb: np.ndarray) -> np.ndarray:
        return (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]).astype(np.float32)

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float32).reshape(-1)
        b = b.astype(np.float32).reshape(-1)
        if a.size < 80 or b.size != a.size:
            return -1e9
        a = a - float(a.mean())
        b = b - float(b.mean())
        sa = float(a.std()) + 1e-6
        sb = float(b.std()) + 1e-6
        return float(np.mean((a / sa) * (b / sb)))

    def _best_offsets_for_blocks(
        band_slice: slice,
        patch_rgb: np.ndarray,
        patch_a: np.ndarray,
        patch_m: np.ndarray,
        torso_rows: np.ndarray,
        block_h: int,
    ) -> np.ndarray:
        patch_w = patch_rgb.shape[1]
        max_off = max(0, patch_w - band_w)
        if max_off <= 0:
            return np.zeros((int(np.ceil(h / float(block_h))),), dtype=np.int32)

        f_band = front_rgb[:, band_slice, :]
        b_band = back_rgb[:, band_slice, :]
        f_m = front_mask[:, band_slice] & (front_alpha[:, band_slice] > 0.85)
        b_m = back_mask[:, band_slice] & (back_alpha[:, band_slice] > 0.85)
        p_m = patch_m & (patch_a > 0.85)

        offsets = np.zeros((int(np.ceil(h / float(block_h))),), dtype=np.int32)
        f_gray = _gray(f_band)
        b_gray = _gray(b_band)
        p_gray = _gray(patch_rgb)

        # 全局对齐偏移：用所有躯干行估计一个稳定的相位偏移，作为后续分块搜索的先验与回退。
        torso2d_all = torso_rows[:, None]
        fm_all = f_m & torso2d_all
        bm_all = b_m & torso2d_all
        pm_all = p_m
        global_best_o = int(max_off // 2)
        if ((fm_all | bm_all).sum() >= 400) and (pm_all.sum() >= 400):
            best_s = -1e9
            for o in range(0, max_off + 1):
                pw = p_gray[:, o : o + band_w]
                pmw = pm_all[:, o : o + band_w]
                s = 0.0
                if fm_all.sum() >= 120:
                    mm = fm_all & pmw
                    if mm.sum() >= 120:
                        s += _corr(pw[mm], f_gray[mm])
                if bm_all.sum() >= 120:
                    mm = bm_all & pmw
                    if mm.sum() >= 120:
                        s += _corr(pw[mm], b_gray[mm])
                if s > best_s:
                    best_s = s
                    global_best_o = int(o)

        for bi, y0 in enumerate(range(0, h, block_h)):
            y1 = min(h, y0 + block_h)
            torso2d = torso_rows[y0:y1, None]
            fm = f_m[y0:y1] & torso2d
            bm = b_m[y0:y1] & torso2d
            pm = p_m[y0:y1]
            if (fm | bm).sum() < 120 or pm.sum() < 120:
                offsets[bi] = int(max_off // 2)
                continue

            best_o = int(global_best_o)
            best_s = -1e9
            # 偏移正则：避免块与块之间出现跳变，降低“方块纹理”伪影风险
            dev_lambda = 0.015
            for o in range(0, max_off + 1):
                pw = p_gray[y0:y1, o : o + band_w]
                pmw = pm[:, o : o + band_w]
                s = 0.0
                if fm.sum() >= 80:
                    mm = fm & pmw
                    if mm.sum() >= 80:
                        s += _corr(pw[mm], f_gray[y0:y1][mm])
                if bm.sum() >= 80:
                    mm = bm & pmw
                    if mm.sum() >= 80:
                        s += _corr(pw[mm], b_gray[y0:y1][mm])
                s = float(s) - float(dev_lambda) * float((o - global_best_o) ** 2)
                if s > best_s:
                    best_s = s
                    best_o = o
            offsets[bi] = int(best_o)

        # 若分块偏移波动过大，直接回退为全局偏移，避免在边缘带出现可见的块状分段。
        if offsets.size >= 3 and max_off >= 8:
            if float(np.std(offsets.astype(np.float32))) > max(3.0, 0.18 * float(max_off)):
                offsets[:] = int(global_best_o)

        return offsets

    def _build_ref_band(
        band_slice: slice,
        patch_rgb: np.ndarray,
        patch_a: np.ndarray,
        patch_m: np.ndarray,
        torso_rows: np.ndarray,
    ):
        block_h = int(np.clip(0.07 * h, 20, 56))
        offsets = _best_offsets_for_blocks(band_slice, patch_rgb, patch_a, patch_m, torso_rows, block_h)
        patch_w = int(patch_rgb.shape[1])
        max_off = max(0, patch_w - band_w)

        block_centers = (np.arange(offsets.size, dtype=np.float32) * float(block_h) + 0.5 * float(block_h)).astype(np.float32)
        block_centers = np.clip(block_centers, 0.0, float(h - 1))
        rows = np.arange(h, dtype=np.float32)
        row_off = np.interp(rows, block_centers, offsets.astype(np.float32))
        k = int(np.clip(0.06 * h, 5, 19))
        if k % 2 == 0:
            k += 1
        pad = k // 2
        row_pad = np.pad(row_off, (pad, pad), mode="edge")
        kernel = np.ones((k,), dtype=np.float32) / float(k)
        row_off = np.convolve(row_pad, kernel, mode="valid")
        row_off = np.clip(np.round(row_off), 0.0, float(max_off)).astype(np.int32)

        ref_rgb = np.zeros((h, band_w, 3), dtype=np.float32)
        ref_a = np.zeros((h, band_w), dtype=np.float32)
        ref_m = np.zeros((h, band_w), dtype=bool)
        for y in range(h):
            o = int(row_off[y])
            ref_rgb[y] = patch_rgb[y, o : o + band_w]
            ref_a[y] = patch_a[y, o : o + band_w]
            ref_m[y] = patch_m[y, o : o + band_w]
        return ref_rgb, ref_a, ref_m

    def transfer_to_band(
        band_slice: slice,
        weight_1d: np.ndarray,
        patch_rgb: np.ndarray,
        patch_a: np.ndarray,
        patch_m: np.ndarray,
    ):
        nonlocal front_rgb, back_rgb

        f_band = front_rgb[:, band_slice, :]
        b_band = back_rgb[:, band_slice, :]
        f_m = front_mask[:, band_slice] & (front_alpha[:, band_slice] > 0.85)
        b_m = back_mask[:, band_slice] & (back_alpha[:, band_slice] > 0.85)
        torso_rows = estimate_torso_rows(front_mask | back_mask)
        torso_rows_2d = torso_rows[:, None]

        patch_w = int(patch_rgb.shape[1])
        if patch_w < band_w:
            return

        ref_rgb, ref_a, ref_m = _build_ref_band(band_slice, patch_rgb, patch_a, patch_m, torso_rows)
        s_m = ref_m & (ref_a > 0.85)

        ref_mask = erode_mask((f_m | b_m) & s_m & torso_rows_2d, iterations=1)
        if ref_mask.sum() < 180:
            return

        ref_pixels = np.concatenate([f_band[ref_mask], b_band[ref_mask]], axis=0)
        src_pixels = ref_rgb[ref_mask]
        if ref_pixels.shape[0] < 80 or src_pixels.shape[0] < 80:
            return

        src_mean, src_std = robust_mean_std(src_pixels)
        tgt_mean, tgt_std = robust_mean_std(ref_pixels)
        ref_adj = apply_match(ref_rgb, ref_m, src_mean, src_std, tgt_mean, tgt_std)

        ww = np.tile(weight_1d[None, :, None], (h, 1, 1)).astype(np.float32)
        ww = ww * ref_a[:, :, None].astype(np.float32)
        ww = ww * torso_rows_2d[:, :, None].astype(np.float32)

        use_f = front_mask[:, band_slice] & ref_m & torso_rows_2d
        use_b = back_mask[:, band_slice] & ref_m & torso_rows_2d

        front_rgb[:, band_slice, :][use_f] = (1.0 - ww[use_f]) * f_band[use_f] + ww[use_f] * ref_adj[use_f]
        back_rgb[:, band_slice, :][use_b] = (1.0 - ww[use_b]) * b_band[use_b] + ww[use_b] * ref_adj[use_b]

    patch_w = int(np.clip(3 * band_w, band_w + 8, int(0.60 * w)))
    patch_w = max(patch_w, band_w)
    left_patch_rgb, left_patch_a, left_patch_m = mid_patch(left_rgb, left_mask, left_alpha, patch_w)
    right_patch_rgb, right_patch_a, right_patch_m = mid_patch(right_rgb, right_mask, right_alpha, patch_w)

    left_band = slice(x0, min(w, x0 + band_w))
    left_w = np.linspace(1.0, 0.0, left_band.stop - left_band.start, dtype=np.float32)
    transfer_to_band(left_band, left_w, left_patch_rgb, left_patch_a, left_patch_m)

    right_band = slice(max(0, x1 - band_w + 1), x1 + 1)
    right_w = np.linspace(0.0, 1.0, right_band.stop - right_band.start, dtype=np.float32)[::-1]
    transfer_to_band(right_band, right_w, right_patch_rgb, right_patch_a, right_patch_m)

    return front_rgb, back_rgb


def apply_modules_to_pair(
    sem_front_path: str,
    sem_back_path: str,
    sem_left_path: str,
    sem_right_path: str,
    seam_module: str,
    seam_band_width: int,
    neckline_manual_x: float,
    neckline_manual_y: float,
    neckline_manual_shape: float,
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

    # 交互式领口：用户在网页上点一个“前领最低点”，与两侧最高点拟合曲线后扣除。
    # 若未提供有效点位（x/y 不在 [0,1]），则不做任何裁剪。
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
        x_search_l0 = 0
        x_search_l1 = max(1, int(0.45 * w))
        x_search_r0 = max(1, int(0.55 * w))
        x_search_r1 = w
        peak_xL = _pick_plateau_edge_x(top_y, x_search_l0, x_search_l1, h, side="left")
        peak_xR = _pick_plateau_edge_x(top_y, x_search_r0, x_search_r1, h, side="right")
        if peak_xL is not None and peak_xR is not None:
            peak_yL = float(top_y[peak_xL])
            peak_yR = float(top_y[peak_xR])
            if peak_xR > peak_xL + 20 and not np.isnan(peak_yL) and not np.isnan(peak_yR):
                u = float(neckline_manual_x)
                v = float(neckline_manual_y)
                user_abs_x = int(round(u * float(full_w - 1)))
                user_abs_y = int(round(v * float(full_h - 1)))
                px = int(np.clip(user_abs_x - x0, peak_xL, peak_xR))
                py = int(np.clip(user_abs_y - y0, int(min(peak_yL, peak_yR)), h - 1))

                t01 = float(np.clip((float(neckline_manual_shape) - 0.50) / 2.00, 0.0, 1.0))
                p_shape = float(1.0 + (1.0 - t01) * 5.0)
                feather = int(np.clip(0.02 * h, 2.0, 10.0))
                bias = int(round(0.2 * feather))
                cutline_points: list[tuple[int, int]] = []
                alpha_box = front_a[y0 : y1 + 1, x0 : x1 + 1].copy()
                for x in range(peak_xL, peak_xR + 1):
                    if x <= px:
                        denom = float(max(1, px - peak_xL))
                        t = float(x - peak_xL) / denom
                        f = 1.0 - (1.0 - t) ** p_shape
                        y_cut_f = float(peak_yL) + (float(py) - float(peak_yL)) * f
                    else:
                        denom = float(max(1, peak_xR - px))
                        t = float(peak_xR - x) / denom
                        f = 1.0 - (1.0 - t) ** p_shape
                        y_cut_f = float(peak_yR) + (float(py) - float(peak_yR)) * f
                    y_cut = int(np.clip(y_cut_f, 0.0, float(h - 1)))
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
                debug["neckline_manual_shape"] = float(neckline_manual_shape)
                debug["neckline_manual_p"] = float(p_shape)

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
    parser.add_argument("--seam-module", choices=["none", "feather_stats", "side_views"], default="feather_stats")
    parser.add_argument("--seam-band-width", type=int, default=24)
    parser.add_argument("--invert-vton-cloth-names", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--neckline-manual-x", type=float, default=-1.0)
    parser.add_argument("--neckline-manual-y", type=float, default=-1.0)
    parser.add_argument("--neckline-manual-shape", type=float, default=1.0)
    parser.add_argument("--debug-dir", default="")
    args = parser.parse_args()

    items = json.loads(Path(args.items_json).read_text(encoding="utf-8"))

    for it in items:
        (f_rgb, f_a), (b_rgb, b_a), debug = apply_modules_to_pair(
            it["src_sem_front"],
            it["src_sem_back"],
            it.get("src_sem_left", ""),
            it.get("src_sem_right", ""),
            args.seam_module,
            int(args.seam_band_width),
            float(args.neckline_manual_x),
            float(args.neckline_manual_y),
            float(args.neckline_manual_shape),
        )

        dst_front = it["dst_front"]
        dst_back = it["dst_back"]
        os.makedirs(os.path.dirname(dst_front), exist_ok=True)

        if args.invert_vton_cloth_names:
            Image.fromarray(composite_on_white(b_rgb, b_a)).save(dst_front, quality=98, subsampling=0, optimize=True)
            Image.fromarray(composite_on_white(f_rgb, f_a)).save(dst_back, quality=98, subsampling=0, optimize=True)
        else:
            Image.fromarray(composite_on_white(f_rgb, f_a)).save(dst_front, quality=98, subsampling=0, optimize=True)
            Image.fromarray(composite_on_white(b_rgb, b_a)).save(dst_back, quality=98, subsampling=0, optimize=True)

        if args.debug_dir:
            debug_dir = Path(args.debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
            base = Path(dst_front).stem.replace("_front", "").replace("_back", "")
            out_front = debug_dir / f"{base}_vton_input_front.jpg"
            out_back = debug_dir / f"{base}_vton_input_back.jpg"
            Image.fromarray(composite_on_white(f_rgb, f_a)).save(out_front, quality=98, subsampling=0, optimize=True)
            Image.fromarray(composite_on_white(b_rgb, b_a)).save(out_back, quality=98, subsampling=0, optimize=True)
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
                    im.save(debug_dir / f"{base}_vton_input_front_cutline.jpg", quality=98, subsampling=0, optimize=True)


if __name__ == "__main__":
    main()
