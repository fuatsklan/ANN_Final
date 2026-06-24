import argparse
import csv
import json
import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from dataset import MVTecTrainDataset, MVTecTestDataset, batch_loader
from model import ConvAutoencoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import ndimage
except ImportError:
    ndimage = None


EPS = 1e-8
NORMALIZED_SCORE_TYPES = {
    "combined_global_patch",
    "combined_global_patch_globalnorm",
    "max_patch_globalnorm",
}


def _require_pixel_threshold(score_type, pixel_threshold):
    """Fail early when a score needs a train-derived pixel threshold."""
    if pixel_threshold is None:
        raise ValueError(f"{score_type} needs pixel_threshold")


def _require_train_stats(score_type, stats):
    """Fail early when a score needs train-set normalization stats."""
    if stats is None:
        raise ValueError(f"{score_type} needs training normalization stats")


def load_model(model, path):
    """Load checkpoint arrays into an existing model."""
    with open(path, "rb") as f:
        saved_params = pickle.load(f)

    for saved, (p, g) in zip(saved_params, model.params()):
        p[...] = saved


def reconstruction_error(x, x_hat):
    """Return one MSE value per image in a batch."""
    return np.mean((x - x_hat) ** 2, axis=(1, 2, 3))


def transform_scores(train_errors, test_errors, mode):
    """Optionally flip scores when lower error should count as more anomalous."""
    if mode == "mse":
        return train_errors, test_errors, "mse"
    if mode == "inverse_mse":
        return -train_errors, -test_errors, "inverse_mse"
    if mode != "auto":
        raise ValueError(f"Unknown score mode: {mode}")

    return train_errors, test_errors, "mse"


def per_pixel_error(x, x_hat):
    """Mean squared error map for a single image (H, W)."""
    diff = (np.asarray(x) - np.asarray(x_hat)) ** 2
    if diff.ndim < 2:
        raise ValueError(f"Expected at least 2D arrays, got shape {diff.shape}")
    reduce_axes = tuple(range(diff.ndim - 2))
    return diff.mean(axis=reduce_axes)


def topk_mean(values, percent):
    """Average the largest percent of values."""
    flat = np.asarray(values).reshape(-1)
    if flat.size == 0:
        return 0.0
    k = max(1, int(np.ceil(flat.size * percent / 100.0)))
    top = np.partition(flat, flat.size - k)[-k:]
    return float(np.mean(top))


def patch_means(error_map, patch_size=8, stride=4):
    """Scan an error map and return the mean error of each patch."""
    error_map = np.asarray(error_map)
    h, w = error_map.shape
    patch_size = max(1, min(patch_size, h, w))
    stride = max(1, stride)
    values = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = error_map[y:y + patch_size, x:x + patch_size]
            values.append(float(np.mean(patch)))

    if not values:
        return np.array([float(np.mean(error_map))])
    return np.array(values)


def smooth_error_map(error_map, kernel_size=1):
    """Apply a small mean filter to make masks less noisy."""
    error_map = np.asarray(error_map)
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return error_map
    if kernel_size % 2 == 0:
        kernel_size += 1

    pad = kernel_size // 2
    padded = np.pad(error_map, ((pad, pad), (pad, pad)), mode="edge")
    smoothed = np.zeros_like(error_map, dtype=np.float64)

    for y in range(kernel_size):
        for x in range(kernel_size):
            smoothed += padded[y:y + error_map.shape[0], x:x + error_map.shape[1]]

    return smoothed / (kernel_size * kernel_size)


def connected_components(binary_mask):
    """Find 8-connected foreground components in a binary mask."""
    binary_mask = np.asarray(binary_mask, dtype=bool)

    if ndimage is not None:
        structure = np.ones((3, 3), dtype=int)
        labels, num_labels = ndimage.label(binary_mask, structure=structure)
        return [
            list(zip(*np.where(labels == label_id)))
            for label_id in range(1, num_labels + 1)
        ]

    h, w = binary_mask.shape
    visited = np.zeros_like(binary_mask, dtype=bool)
    components = []

    for y0 in range(h):
        for x0 in range(w):
            if not binary_mask[y0, x0] or visited[y0, x0]:
                continue

            stack = [(y0, x0)]
            visited[y0, x0] = True
            coords = []

            while stack:
                y, x = stack.pop()
                coords.append((y, x))

                for ny in range(y - 1, y + 2):
                    for nx in range(x - 1, x + 2):
                        if ny == y and nx == x:
                            continue
                        if (
                            0 <= ny < h
                            and 0 <= nx < w
                            and binary_mask[ny, nx]
                            and not visited[ny, nx]
                        ):
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            components.append(coords)

    return components


def filter_small_components(binary_mask, min_component_size=1):
    """Remove tiny connected components from a binary mask."""
    min_component_size = max(1, int(min_component_size))
    if min_component_size <= 1:
        return np.asarray(binary_mask, dtype=bool)

    filtered = np.zeros_like(binary_mask, dtype=bool)
    for component in connected_components(binary_mask):
        if len(component) >= min_component_size:
            ys, xs = zip(*component)
            filtered[np.array(ys), np.array(xs)] = True
    return filtered


def blob_score(
    error_map,
    pixel_threshold,
    smooth_kernel=1,
    min_component_size=1,
):
    """Score the strongest connected high-error region."""
    _require_pixel_threshold("blob_score", pixel_threshold)

    processed = smooth_error_map(error_map, smooth_kernel)
    blob_mask = filter_small_components(processed > pixel_threshold, min_component_size)
    best_score = 0.0

    for component in connected_components(blob_mask):
        ys, xs = zip(*component)
        values = processed[np.array(ys), np.array(xs)]
        excess = np.maximum(values - pixel_threshold, 0.0)
        best_score = max(best_score, float(np.sum(excess)))

    return best_score


def hybrid_blob_score(
    error_map,
    pixel_threshold,
    smooth_kernel=1,
    min_component_size=1,
):
    """Blob score with an extra image-relative contrast term."""
    _require_pixel_threshold("hybrid_blob_score", pixel_threshold)

    processed = smooth_error_map(error_map, smooth_kernel)
    blob_mask = filter_small_components(processed > pixel_threshold, min_component_size)
    image_background = float(np.median(processed))
    image_spread = float(np.median(np.abs(processed - image_background))) + EPS
    best_score = 0.0

    for component in connected_components(blob_mask):
        ys, xs = zip(*component)
        values = processed[np.array(ys), np.array(xs)]
        absolute_strength = float(np.sum(np.maximum(values - pixel_threshold, 0.0)))
        relative_contrast = max(float((np.mean(values) - image_background) / image_spread), 0.0)
        best_score = max(best_score, absolute_strength * np.log1p(relative_contrast))

    return best_score


def combined_global_patch_score(error_map, patch_size, patch_stride, stats):
    """Blend global and local patch z-scores."""
    _require_train_stats("combined_global_patch", stats)

    global_score = float(np.mean(error_map))
    local_score = float(np.max(patch_means(error_map, patch_size, patch_stride)))

    global_z = (global_score - stats["global_mean"]) / stats["global_std"]
    local_z = (local_score - stats["local_mean"]) / stats["local_std"]

    return 0.2 * global_z + 0.8 * local_z


def combined_global_patch_globalnorm_score(error_map, patch_size, patch_stride, stats):
    """Blend global score and local patch score using global train stats."""
    _require_train_stats("combined_global_patch_globalnorm", stats)

    global_score = float(np.mean(error_map))
    local_score = float(np.max(patch_means(error_map, patch_size, patch_stride)))

    global_z = (global_score - stats["global_mean"]) / stats["global_std"]
    local_global_z = (local_score - stats["global_mean"]) / stats["global_std"]

    return 0.5 * global_z + 0.5 * local_global_z


def max_patch_globalnorm_score(error_map, patch_size, patch_stride, stats):
    """Use the strongest patch, normalized by normal-image global MSE stats."""
    _require_train_stats("max_patch_globalnorm", stats)

    local_score = float(np.max(patch_means(error_map, patch_size, patch_stride)))
    return (local_score - stats["global_mean"]) / stats["global_std"]


def anomaly_score_from_map(
    error_map,
    score_type="global_mse",
    patch_size=8,
    patch_stride=4,
    topk_percent=5.0,
    pixel_threshold=None,
    smooth_kernel=1,
    min_component_size=1,
    combined_stats=None,
):
    """Convert a reconstruction error map into one image-level anomaly score."""
    if score_type == "global_mse":
        return float(np.mean(error_map))
    if score_type == "max_pixel":
        return float(np.max(error_map))
    if score_type == "topk_pixels":
        return topk_mean(error_map, topk_percent)
    if score_type == "max_patch":
        return float(np.max(patch_means(error_map, patch_size, patch_stride)))
    if score_type == "topk_patches":
        patches = patch_means(error_map, patch_size, patch_stride)
        return topk_mean(patches, topk_percent)
    if score_type == "blob_score":
        return blob_score(
            error_map,
            pixel_threshold=pixel_threshold,
            smooth_kernel=smooth_kernel,
            min_component_size=min_component_size,
        )
    if score_type == "hybrid_blob_score":
        return hybrid_blob_score(
            error_map,
            pixel_threshold=pixel_threshold,
            smooth_kernel=smooth_kernel,
            min_component_size=min_component_size,
        )
    if score_type == "combined_global_patch":
        return combined_global_patch_score(
            error_map,
            patch_size=patch_size,
            patch_stride=patch_stride,
            stats=combined_stats,
        )
    if score_type == "combined_global_patch_globalnorm":
        return combined_global_patch_globalnorm_score(
            error_map,
            patch_size=patch_size,
            patch_stride=patch_stride,
            stats=combined_stats,
        )
    if score_type == "max_patch_globalnorm":
        return max_patch_globalnorm_score(
            error_map,
            patch_size=patch_size,
            patch_stride=patch_stride,
            stats=combined_stats,
        )
    raise ValueError(f"Unknown score type: {score_type}")


def mask_path_for_image(image_path, data_root, category):
    """Map a defective test image path to its ground-truth mask path."""
    image_path = Path(image_path)
    defect_type = image_path.parent.name
    if defect_type == "good":
        return None
    return (
        Path(data_root)
        / category
        / "ground_truth"
        / defect_type
        / f"{image_path.stem}_mask.png"
    )


def load_mask(image_path, data_root, category, img_size):
    """Load a ground-truth mask, or return an empty mask for normal images."""
    mask_path = mask_path_for_image(image_path, data_root, category)
    if mask_path is None or not mask_path.exists():
        return np.zeros((img_size, img_size), dtype=bool)

    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((img_size, img_size), resample=Image.Resampling.NEAREST)
    return np.asarray(mask) > 0


def pixel_metrics_from_counts(tp, fp, fn, tn):
    """Compute pixel-level metrics from accumulated confusion counts."""
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)
    iou = tp / (tp + fp + fn + EPS)
    accuracy = (tp + tn) / (tp + fp + fn + tn + EPS)
    return {
        "pixel_precision": float(precision),
        "pixel_recall": float(recall),
        "pixel_f1": float(f1),
        "pixel_iou": float(iou),
        "pixel_accuracy": float(accuracy),
    }


def classification_metrics_from_predictions(y_true, y_pred):
    """Compute image-level binary classification metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)

    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def parse_percentiles(value):
    """Parse comma-separated percentile values from the CLI."""
    if not value:
        return []
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def auc_score(y_true, scores):
    """Compute ROC-AUC by pairwise positive/negative ranking."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    pos = scores[y_true == 1]
    neg = scores[y_true == 0]

    if len(pos) == 0 or len(neg) == 0:
        return np.nan

    count = 0
    total = len(pos) * len(neg)

    for p in pos:
        for n in neg:
            if p > n:
                count += 1
            elif p == n:
                count += 0.5

    return count / total


def roc_curve_points(y_true, scores):
    """Build ROC points by sweeping unique score thresholds."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    order = np.argsort(-scores)
    y_sorted = y_true[order]
    s_sorted = scores[order]

    thresholds = np.r_[np.inf, np.unique(s_sorted)[::-1], -np.inf]
    tpr_list = []
    fpr_list = []

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    for t in thresholds:
        pred = scores >= t
        tp = np.sum((y_true == 1) & pred)
        fp = np.sum((y_true == 0) & pred)
        fn = np.sum((y_true == 1) & ~pred)
        tn = np.sum((y_true == 0) & ~pred)

        tpr = tp / (tp + fn + EPS) if n_pos else 0.0
        fpr = fp / (fp + tn + EPS) if n_neg else 0.0
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return np.array(fpr_list), np.array(tpr_list), thresholds


def collect_train_scores(
    model,
    train_dataset,
    batch_size,
    score_type,
    patch_size,
    patch_stride,
    topk_percent,
    pixel_threshold_percentile,
    smooth_kernel,
    min_component_size,
):
    """Collect normal-training scores and thresholds used during evaluation."""
    error_maps = []
    processed_pixel_errors = []
    for x, _ in batch_loader(train_dataset, batch_size, shuffle=False):
        x_hat = model.forward(x)
        for i in range(x.shape[0]):
            error_map = per_pixel_error(x[i], x_hat[i])
            processed_error_map = smooth_error_map(error_map, smooth_kernel)
            error_maps.append(error_map)
            processed_pixel_errors.extend(processed_error_map.reshape(-1))

    processed_pixel_errors = np.array(processed_pixel_errors)
    pixel_threshold = float(
        np.percentile(processed_pixel_errors, pixel_threshold_percentile)
    )
    combined_stats = None

    if score_type in NORMALIZED_SCORE_TYPES:
        global_scores = np.array([float(np.mean(error_map)) for error_map in error_maps])
        local_scores = np.array([
            float(np.max(patch_means(error_map, patch_size, patch_stride)))
            for error_map in error_maps
        ])
        combined_stats = {
            "global_mean": float(np.mean(global_scores)),
            "global_std": float(np.std(global_scores) + EPS),
            "local_mean": float(np.mean(local_scores)),
            "local_std": float(np.std(local_scores) + EPS),
        }

    scores = [
        anomaly_score_from_map(
            error_map,
            score_type=score_type,
            patch_size=patch_size,
            patch_stride=patch_stride,
            topk_percent=topk_percent,
            pixel_threshold=pixel_threshold,
            smooth_kernel=smooth_kernel,
            min_component_size=min_component_size,
            combined_stats=combined_stats,
        )
        for error_map in error_maps
    ]
    return np.array(scores), processed_pixel_errors, pixel_threshold, combined_stats


def run_evaluation(
    model,
    train_dataset,
    test_dataset,
    batch_size,
    threshold_percentile,
    score_mode,
    score_type,
    patch_size,
    patch_stride,
    topk_percent,
    pixel_threshold_percentile,
    smooth_kernel,
    min_component_size,
    data_root,
    category,
    img_size,
):
    """Run the full evaluation loop over train and test splits."""
    train_raw_scores, train_pixel_errors, pixel_threshold, combined_stats = collect_train_scores(
        model,
        train_dataset,
        batch_size,
        score_type,
        patch_size,
        patch_stride,
        topk_percent,
        pixel_threshold_percentile,
        smooth_kernel,
        min_component_size,
    )

    records = []
    for i in range(len(test_dataset)):
        x, label, defect_type, path = test_dataset[i]
        x_batch = x[None, :, :, :]
        x_hat = model.forward(x_batch)
        error_map = per_pixel_error(x, x_hat[0])
        error = float(np.mean(error_map))
        raw_score = anomaly_score_from_map(
            error_map,
            score_type=score_type,
            patch_size=patch_size,
            patch_stride=patch_stride,
            topk_percent=topk_percent,
            pixel_threshold=pixel_threshold,
            smooth_kernel=smooth_kernel,
            min_component_size=min_component_size,
            combined_stats=combined_stats,
        )
        mask = load_mask(path, data_root, category, img_size)
        processed_error_map = smooth_error_map(error_map, smooth_kernel)
        pred_mask = filter_small_components(
            processed_error_map > pixel_threshold,
            min_component_size,
        )
        pixel_tp = int(np.sum(pred_mask & mask))
        pixel_fp = int(np.sum(pred_mask & ~mask))
        pixel_fn = int(np.sum(~pred_mask & mask))
        pixel_tn = int(np.sum(~pred_mask & ~mask))
        per_image_pixel_metrics = pixel_metrics_from_counts(
            pixel_tp, pixel_fp, pixel_fn, pixel_tn
        )

        records.append(
            {
                "error": error,
                "raw_score": float(raw_score),
                "score": float(raw_score),
                "label": label,
                "pred": 0,
                "defect_type": defect_type,
                "path": path,
                "x": x,
                "x_hat": x_hat[0],
                "error_map": error_map,
                "processed_error_map": processed_error_map,
                "mask": mask,
                "pred_mask": pred_mask,
                "pixel_tp": pixel_tp,
                "pixel_fp": pixel_fp,
                "pixel_fn": pixel_fn,
                "pixel_tn": pixel_tn,
                **per_image_pixel_metrics,
            }
        )

    y_true = np.array([r["label"] for r in records])
    raw_scores = np.array([r["raw_score"] for r in records])
    train_scores, scores, resolved_score_mode = transform_scores(
        train_raw_scores, raw_scores, score_mode
    )

    if score_mode == "auto" and auc_score(y_true, -raw_scores) > auc_score(y_true, raw_scores):
        train_scores, scores, resolved_score_mode = transform_scores(
            train_raw_scores, raw_scores, "inverse_mse"
        )

    threshold = np.percentile(train_scores, threshold_percentile)

    for r, score in zip(records, scores):
        r["score"] = float(score)
        r["pred"] = 1 if score > threshold else 0

    y_pred = np.array([r["pred"] for r in records])

    image_metrics = classification_metrics_from_predictions(y_true, y_pred)
    pixel_tp = int(sum(r["pixel_tp"] for r in records))
    pixel_fp = int(sum(r["pixel_fp"] for r in records))
    pixel_fn = int(sum(r["pixel_fn"] for r in records))
    pixel_tn = int(sum(r["pixel_tn"] for r in records))

    auc = auc_score(y_true, scores)
    pixel_metrics = pixel_metrics_from_counts(pixel_tp, pixel_fp, pixel_fn, pixel_tn)

    metrics = {
        "threshold": float(threshold),
        "threshold_percentile": threshold_percentile,
        "score_mode": resolved_score_mode,
        "score_type": score_type,
        "patch_size": patch_size,
        "patch_stride": patch_stride,
        "topk_percent": topk_percent,
        "smooth_kernel": smooth_kernel,
        "min_component_size": min_component_size,
        "combined_stats": combined_stats,
        "pixel_threshold": pixel_threshold,
        "pixel_threshold_percentile": pixel_threshold_percentile,
        "accuracy": image_metrics["accuracy"],
        "precision": image_metrics["precision"],
        "recall": image_metrics["recall"],
        "f1": image_metrics["f1"],
        "auc": float(auc),
        "tp": image_metrics["tp"],
        "tn": image_metrics["tn"],
        "fp": image_metrics["fp"],
        "fn": image_metrics["fn"],
        "pixel_tp": pixel_tp,
        "pixel_fp": pixel_fp,
        "pixel_fn": pixel_fn,
        "pixel_tn": pixel_tn,
        **pixel_metrics,
    }

    return train_scores, train_pixel_errors, records, metrics


def print_metrics(metrics):
    """Print the main image-level and pixel-level results."""
    print("\nScratch CAE Evaluation")
    print("----------------------")
    print(f"Threshold percentile: {metrics['threshold_percentile']}")
    print(f"Score type:           {metrics['score_type']}")
    print(f"Score mode:           {metrics['score_mode']}")
    print(f"Smooth kernel:        {metrics['smooth_kernel']}")
    print(f"Min component size:   {metrics['min_component_size']}")
    print(f"Threshold:            {metrics['threshold']:.8f}")
    print(f"Accuracy:             {metrics['accuracy']:.4f}")
    print(f"Precision:            {metrics['precision']:.4f}")
    print(f"Recall:               {metrics['recall']:.4f}")
    print(f"F1-score:             {metrics['f1']:.4f}")
    print(f"ROC-AUC:              {metrics['auc']:.4f}")
    print()
    print(
        f"TP={metrics['tp']}, TN={metrics['tn']}, "
        f"FP={metrics['fp']}, FN={metrics['fn']}"
    )
    print()
    print("Pixel Localization")
    print("------------------")
    print(f"Pixel threshold percentile: {metrics['pixel_threshold_percentile']}")
    print(f"Pixel threshold:            {metrics['pixel_threshold']:.8f}")
    print(f"Pixel precision:            {metrics['pixel_precision']:.4f}")
    print(f"Pixel recall:               {metrics['pixel_recall']:.4f}")
    print(f"Pixel F1-score:             {metrics['pixel_f1']:.4f}")
    print(f"Pixel IoU:                  {metrics['pixel_iou']:.4f}")
    print(
        f"TPP={metrics['pixel_tp']}, TNP={metrics['pixel_tn']}, "
        f"FPP={metrics['pixel_fp']}, FNP={metrics['pixel_fn']}"
    )


def plot_score_distributions(train_scores, records, threshold, out_path, score_label):
    """Plot train/test score distributions and the selected threshold."""
    test_good = np.array([r["score"] for r in records if r["label"] == 0])
    test_defect = np.array([r["score"] for r in records if r["label"] == 1])
    train_scores = np.asarray(train_scores)
    combined = np.concatenate([train_scores, test_good, test_defect])
    positive = combined[combined > 0]
    use_log = (
        positive.size > 0
        and np.min(combined) >= 0
        and np.max(positive) / (np.median(positive) + EPS) > 50
    )

    if use_log:
        train_plot = np.log1p(train_scores)
        good_plot = np.log1p(test_good)
        defect_plot = np.log1p(test_defect)
        threshold_plot = np.log1p(max(threshold, 0.0))
        x_label = f"log1p({score_label})"
    else:
        train_plot = train_scores
        good_plot = test_good
        defect_plot = test_defect
        threshold_plot = threshold
        x_label = score_label

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = 30

    ax.hist(
        train_plot,
        bins=bins,
        alpha=0.55,
        density=True,
        label=f"Train (normal), n={len(train_plot)}",
        color="#2ecc71",
    )
    ax.hist(
        good_plot,
        bins=bins,
        alpha=0.55,
        density=True,
        label=f"Test good, n={len(good_plot)}",
        color="#3498db",
    )
    ax.hist(
        defect_plot,
        bins=bins,
        alpha=0.55,
        density=True,
        label=f"Test defect, n={len(defect_plot)}",
        color="#e74c3c",
    )
    ax.axvline(
        threshold_plot,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold ({threshold:.2e})",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    ax.set_title("Anomaly score: train normal vs test good vs test defect")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scores_by_defect_type(records, threshold, out_path, score_label):
    """Show how scores differ across test folders."""
    by_type = {}
    for r in records:
        by_type.setdefault(r["defect_type"], []).append(r["score"])

    types = sorted(by_type.keys())
    data = [by_type[t] for t in types]

    fig, ax = plt.subplots(figsize=(max(8, len(types) * 0.9), 5))
    bp = ax.boxplot(data, tick_labels=types, patch_artist=True)

    colors = ["#3498db" if t == "good" else "#e74c3c" for t in types]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Threshold",
    )
    ax.set_ylabel(score_label)
    ax.set_xlabel("Test folder (defect type)")
    ax.set_title("Score by defect type (boxplot)")
    ax.legend(loc="upper right")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc(y_true, scores, auc, out_path):
    """Save the ROC curve for image-level anomaly detection."""
    fpr, tpr, _ = roc_curve_points(y_true, scores)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#8e44ad", linewidth=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (anomaly = higher score)")
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(metrics, out_path):
    """Save a labeled 2x2 confusion matrix."""
    cm = np.array(
        [
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ]
    )
    labels = [["TN", "FP"], ["FN", "TP"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred normal", "Pred defect"])
    ax.set_yticklabels(["True normal", "True defect"])
    ax.set_title("Confusion matrix")

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{labels[i][j]}\n{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=12,
            )

    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metrics_summary(metrics, out_path):
    """Save a compact bar chart of the main image-level metrics."""
    names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    values = [
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["auc"] if not np.isnan(metrics["auc"]) else 0.0,
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, values, color=["#3498db", "#9b59b6", "#e67e22", "#1abc9c", "#34495e"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Evaluation metrics summary")
    ax.axhline(1.0, color="gray", linewidth=0.5, alpha=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _pick_example_records(records, max_per_type=1, seed=42):
    """Pick a small, repeatable set of examples for reconstruction figures."""
    rng = np.random.default_rng(seed)
    by_type = {}
    for r in records:
        by_type.setdefault(r["defect_type"], []).append(r)

    chosen = []
    for defect_type in sorted(by_type.keys()):
        pool = by_type[defect_type]
        k = min(max_per_type, len(pool))
        idx = rng.choice(len(pool), size=k, replace=False)
        chosen.extend(pool[i] for i in idx)

    return chosen


def plot_reconstruction_grid(examples, threshold, out_path):
    """Save input, reconstruction, and error-map examples."""
    n = len(examples)
    fig, axes = plt.subplots(n, 3, figsize=(9, 2.8 * n))
    if n == 1:
        axes = axes[None, :]

    for row, rec in enumerate(examples):
        x = rec["x"][0]
        x_hat = rec["x_hat"][0]
        err_map = rec["error_map"]

        status = "OK" if rec["label"] == rec["pred"] else "WRONG"
        title_suffix = f" | MSE={rec['error']:.2e} | {status}"

        axes[row, 0].imshow(x, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"Input ({rec['defect_type']}){title_suffix}", fontsize=8)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(x_hat, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Reconstruction", fontsize=8)
        axes[row, 1].axis("off")

        im = axes[row, 2].imshow(err_map, cmap="hot")
        axes[row, 2].set_title("Per-pixel MSE", fontsize=8)
        axes[row, 2].axis("off")
        fig.colorbar(im, ax=axes[row, 2], fraction=0.046)

    fig.suptitle(
        f"Reconstructions vs error maps (threshold = {threshold:.2e})",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_misclassified(records, threshold, out_path, max_images=8):
    """Save examples where the image-level decision was wrong."""
    wrong = [r for r in records if r["label"] != r["pred"]]
    if not wrong:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No misclassifications", ha="center", va="center", fontsize=14)
        ax.axis("off")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    wrong = wrong[:max_images]
    n = len(wrong)
    fig, axes = plt.subplots(n, 2, figsize=(6, 2.5 * n))
    if n == 1:
        axes = axes[None, :]

    for row, rec in enumerate(wrong):
        x = rec["x"][0]
        err_map = rec["error_map"]
        kind = "FP (normal flagged)" if rec["label"] == 0 else "FN (defect missed)"

        axes[row, 0].imshow(x, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(
            f"{kind} — {rec['defect_type']}\nMSE={rec['error']:.2e}",
            fontsize=8,
        )
        axes[row, 0].axis("off")

        im = axes[row, 1].imshow(err_map, cmap="hot")
        axes[row, 1].set_title("Error map", fontsize=8)
        axes[row, 1].axis("off")
        fig.colorbar(im, ax=axes[row, 1], fraction=0.046)

    fig.suptitle(f"Misclassified samples (threshold = {threshold:.2e})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_localization_grid(records, pixel_threshold, out_path, max_images=6):
    """Save predicted masks next to ground-truth masks."""
    examples = [r for r in records if r["label"] == 1 and np.any(r["mask"])]
    examples = examples[:max_images]

    if not examples:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No ground-truth masks available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    n = len(examples)
    fig, axes = plt.subplots(n, 4, figsize=(10, 2.5 * n))
    if n == 1:
        axes = axes[None, :]

    for row, rec in enumerate(examples):
        axes[row, 0].imshow(rec["x"][0], cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"Input ({rec['defect_type']})", fontsize=8)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(rec["mask"], cmap="gray")
        axes[row, 1].set_title("Ground truth", fontsize=8)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(rec["pred_mask"], cmap="gray")
        axes[row, 2].set_title(f"Pred mask\nIoU={rec['pixel_iou']:.3f}", fontsize=8)
        axes[row, 2].axis("off")

        im = axes[row, 3].imshow(rec["processed_error_map"], cmap="hot")
        axes[row, 3].set_title("Processed error map", fontsize=8)
        axes[row, 3].axis("off")
        fig.colorbar(im, ax=axes[row, 3], fraction=0.046)

    fig.suptitle(f"Pixel localization (threshold = {pixel_threshold:.2e})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_threshold_sweep(train_scores, records, percentiles):
    """Evaluate image-level metrics at several threshold percentiles."""
    y_true = np.array([r["label"] for r in records])
    scores = np.array([r["score"] for r in records])
    rows = []

    for percentile in percentiles:
        threshold = float(np.percentile(train_scores, percentile))
        y_pred = (scores > threshold).astype(int)
        row = {
            "threshold_percentile": float(percentile),
            "threshold": threshold,
            **classification_metrics_from_predictions(y_true, y_pred),
        }
        rows.append(row)

    return rows


def build_pixel_threshold_sweep(
    train_pixel_errors,
    records,
    percentiles,
    smooth_kernel,
    min_component_size,
):
    """Evaluate localization metrics at several pixel thresholds."""
    rows = []

    for percentile in percentiles:
        threshold = float(np.percentile(train_pixel_errors, percentile))
        tp = fp = fn = tn = 0

        for rec in records:
            pred_mask = filter_small_components(
                rec["processed_error_map"] > threshold,
                min_component_size,
            )
            mask = rec["mask"]
            tp += int(np.sum(pred_mask & mask))
            fp += int(np.sum(pred_mask & ~mask))
            fn += int(np.sum(~pred_mask & mask))
            tn += int(np.sum(~pred_mask & ~mask))

        rows.append(
            {
                "pixel_threshold_percentile": float(percentile),
                "pixel_threshold": threshold,
                "pixel_tp": tp,
                "pixel_fp": fp,
                "pixel_fn": fn,
                "pixel_tn": tn,
                **pixel_metrics_from_counts(tp, fp, fn, tn),
            }
        )

    return rows


def plot_threshold_sweep(rows, out_path):
    """Plot image-level metrics over threshold percentiles."""
    if not rows:
        return

    x = [row["threshold_percentile"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, color in [
        ("precision", "#2c7fb8"),
        ("recall", "#d95f0e"),
        ("f1", "#31a354"),
        ("accuracy", "#756bb1"),
    ]:
        ax.plot(x, [row[name] for row in rows], marker="o", linewidth=2, label=name, color=color)

    ax.invert_xaxis()
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Training score percentile threshold")
    ax.set_ylabel("Image-level metric")
    ax.set_title("Image threshold sweep")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pixel_threshold_sweep(rows, out_path):
    """Plot pixel-level metrics over pixel threshold percentiles."""
    if not rows:
        return

    x = [row["pixel_threshold_percentile"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, color in [
        ("pixel_precision", "#2c7fb8"),
        ("pixel_recall", "#d95f0e"),
        ("pixel_f1", "#31a354"),
        ("pixel_iou", "#756bb1"),
    ]:
        ax.plot(x, [row[name] for row in rows], marker="o", linewidth=2, label=name, color=color)

    ax.invert_xaxis()
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Training pixel-error percentile threshold")
    ax.set_ylabel("Pixel-level metric")
    ax.set_title("Pixel threshold sweep")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_visualizations(
    train_scores,
    records,
    metrics,
    figures_dir,
    threshold_sweep_rows=None,
    pixel_threshold_sweep_rows=None,
    max_recon_per_type=1,
    seed=42,
):
    """Save all standard evaluation plots for one experiment."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.array([r["label"] for r in records])
    scores = np.array([r["score"] for r in records])
    threshold = metrics["threshold"]
    score_label = f"Anomaly score ({metrics['score_type']})"

    plots = {
        "01_score_distributions.png": lambda: plot_score_distributions(
            train_scores,
            records,
            threshold,
            figures_dir / "01_score_distributions.png",
            score_label,
        ),
        "02_scores_by_defect_type.png": lambda: plot_scores_by_defect_type(
            records, threshold, figures_dir / "02_scores_by_defect_type.png", score_label
        ),
        "03_roc_curve.png": lambda: plot_roc(
            y_true, scores, metrics["auc"], figures_dir / "03_roc_curve.png"
        ),
        "04_confusion_matrix.png": lambda: plot_confusion_matrix(
            metrics, figures_dir / "04_confusion_matrix.png"
        ),
        "05_metrics_summary.png": lambda: plot_metrics_summary(
            metrics, figures_dir / "05_metrics_summary.png"
        ),
    }

    examples = _pick_example_records(records, max_per_type=max_recon_per_type, seed=seed)
    plots["06_reconstruction_examples.png"] = lambda: plot_reconstruction_grid(
        examples, threshold, figures_dir / "06_reconstruction_examples.png"
    )
    plots["07_misclassified.png"] = lambda: plot_misclassified(
        records, threshold, figures_dir / "07_misclassified.png"
    )
    plots["08_pixel_localization.png"] = lambda: plot_localization_grid(
        records,
        metrics["pixel_threshold"],
        figures_dir / "08_pixel_localization.png",
    )
    if threshold_sweep_rows:
        plots["09_threshold_sweep.png"] = lambda: plot_threshold_sweep(
            threshold_sweep_rows,
            figures_dir / "09_threshold_sweep.png",
        )
    if pixel_threshold_sweep_rows:
        plots["10_pixel_threshold_sweep.png"] = lambda: plot_pixel_threshold_sweep(
            pixel_threshold_sweep_rows,
            figures_dir / "10_pixel_threshold_sweep.png",
        )

    for name, fn in plots.items():
        fn()

    print(f"\nFigures saved to: {figures_dir.resolve()}")
    for name in plots:
        print(f"  - {name}")


def save_metrics(metrics, records, out_dir, category):
    """Save experiment metrics and per-image records."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / f"{category}_evaluation_metrics.json"
    records_path = out_dir / f"{category}_evaluation_records.csv"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(records_path, "w", newline="") as f:
        fieldnames = [
            "path",
            "defect_type",
            "label",
            "pred",
            "error",
            "raw_score",
            "score",
            "pixel_precision",
            "pixel_recall",
            "pixel_f1",
            "pixel_iou",
            "pixel_accuracy",
            "pixel_tp",
            "pixel_fp",
            "pixel_fn",
            "pixel_tn",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({name: r[name] for name in fieldnames})

    print(f"\nMetrics saved to: {metrics_path.resolve()}")
    print(f"Records saved to: {records_path.resolve()}")


def save_sweep_csv(rows, path):
    """Save threshold sweep rows when a sweep was requested."""
    if not rows:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Sweep saved to: {path.resolve()}")


def evaluate(args):
    """Load data/model, run evaluation, and save results."""
    np.random.seed(args.seed)

    model = ConvAutoencoder()
    load_model(model, args.checkpoint)

    train_dataset = MVTecTrainDataset(
        root_dir=args.data_root,
        category=args.category,
        img_size=args.img_size,
    )

    test_dataset = MVTecTestDataset(
        root_dir=args.data_root,
        category=args.category,
        img_size=args.img_size,
    )

    train_scores, train_pixel_errors, records, metrics = run_evaluation(
        model,
        train_dataset,
        test_dataset,
        args.batch_size,
        args.threshold_percentile,
        args.score_mode,
        args.score_type,
        args.patch_size,
        args.patch_stride,
        args.topk_percent,
        args.pixel_threshold_percentile,
        args.smooth_error_kernel,
        args.min_component_size,
        args.data_root,
        args.category,
        args.img_size,
    )
    threshold_sweep_rows = build_threshold_sweep(
        train_scores,
        records,
        parse_percentiles(args.threshold_sweep_percentiles),
    )
    pixel_threshold_sweep_rows = build_pixel_threshold_sweep(
        train_pixel_errors,
        records,
        parse_percentiles(args.pixel_threshold_sweep_percentiles),
        args.smooth_error_kernel,
        args.min_component_size,
    )

    print_metrics(metrics)
    save_metrics(metrics, records, args.metrics_dir, args.category)
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_sweep_csv(
        threshold_sweep_rows,
        metrics_dir / f"{args.category}_threshold_sweep.csv",
    )
    save_sweep_csv(
        pixel_threshold_sweep_rows,
        metrics_dir / f"{args.category}_pixel_threshold_sweep.csv",
    )

    if not args.no_plots:
        figures_dir = Path(args.figures_dir) / args.category
        save_visualizations(
            train_scores,
            records,
            metrics,
            figures_dir,
            threshold_sweep_rows=threshold_sweep_rows,
            pixel_threshold_sweep_rows=pixel_threshold_sweep_rows,
            max_recon_per_type=args.max_recon_per_type,
            seed=args.seed,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--category", type=str, default="carpet")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "checkpoints" / "carpet_scratch_cae.pkl"),
    )

    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--threshold_percentile",
        type=float,
        default=90.0,
        help="Image threshold percentile from normal train scores. Lower catches more defects.",
    )
    parser.add_argument(
        "--score_type",
        choices=[
            "global_mse",
            "max_pixel",
            "topk_pixels",
            "max_patch",
            "topk_patches",
            "blob_score",
            "hybrid_blob_score",
            "combined_global_patch",
            "combined_global_patch_globalnorm",
            "max_patch_globalnorm",
        ],
        default="max_patch",
        help="Image-level anomaly score computed from the reconstruction error map",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=8,
        help="Patch size for max_patch and topk_patches scores",
    )
    parser.add_argument(
        "--patch_stride",
        type=int,
        default=4,
        help="Patch stride for max_patch and topk_patches scores",
    )
    parser.add_argument(
        "--topk_percent",
        type=float,
        default=5.0,
        help="Percent of largest pixels/patches averaged by top-k scores",
    )
    parser.add_argument(
        "--smooth_error_kernel",
        type=int,
        default=1,
        help="Odd average-filter size for error-map smoothing before blob/mask scoring",
    )
    parser.add_argument(
        "--min_component_size",
        type=int,
        default=1,
        help="Remove predicted connected components smaller than this many pixels",
    )
    parser.add_argument(
        "--pixel_threshold_percentile",
        type=float,
        default=99.0,
        help="Training-pixel percentile used for predicted localization masks",
    )
    parser.add_argument(
        "--threshold_sweep_percentiles",
        type=str,
        default="99,97.5,95,90,85,80,75,70,60,50,40,30",
        help="Comma-separated image-threshold percentiles for sweep CSV/plot",
    )
    parser.add_argument(
        "--pixel_threshold_sweep_percentiles",
        type=str,
        default="99.9,99.5,99,98,97,95,90,85,80",
        help="Comma-separated pixel-threshold percentiles for sweep CSV/plot",
    )

    parser.add_argument(
        "--figures_dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "figures"),
        help="Directory for evaluation plots",
    )
    parser.add_argument(
        "--metrics_dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "metrics"),
        help="Directory for evaluation metrics and per-image records",
    )
    parser.add_argument(
        "--score_mode",
        choices=["mse", "inverse_mse", "auto"],
        default="mse",
        help="Anomaly score. Use auto only as a diagnostic because it uses test labels.",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip saving figures",
    )
    parser.add_argument(
        "--max_recon_per_type",
        type=int,
        default=1,
        help="Reconstruction grid: samples per defect folder",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    evaluate(args)
