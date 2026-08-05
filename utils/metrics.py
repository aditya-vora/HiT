import numpy as np
from typing import Tuple
from scipy.spatial import cKDTree


def segmentation_miou(pred_labels: np.array, gt_labels: np.array) -> float:
    """Computes the per-shape mean part IoU between predicted and ground-truth labels.

    Each predicted label is first matched to whichever ground-truth label it overlaps
    with the most (majority vote), after which IoU is computed per ground-truth part.

    Args:
        pred_labels (np.array): predicted part label per point, [N].
        gt_labels (np.array): ground-truth part label per point, [N].
    Returns:
        float: mean IoU over ground-truth parts.
    """
    pred_ids = np.unique(pred_labels)
    gt_ids = np.unique(gt_labels)

    poll = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.int64)
    for i, pred_id in enumerate(pred_ids):
        pred_mask = pred_labels == pred_id
        for j, gt_id in enumerate(gt_ids):
            poll[i, j] = np.sum(pred_mask & (gt_labels == gt_id))

    matched_gt_ids = gt_ids[np.argmax(poll, axis=-1)]
    mapped_pred_labels = np.zeros_like(pred_labels)
    for i, pred_id in enumerate(pred_ids):
        mapped_pred_labels[pred_labels == pred_id] = matched_gt_ids[i]

    part_ious = []
    for gt_id in gt_ids:
        gt_mask = gt_labels == gt_id
        pred_mask = mapped_pred_labels == gt_id
        union = np.sum(gt_mask | pred_mask)
        if union == 0:
            part_ious.append(1.0)
        else:
            part_ious.append(np.sum(gt_mask & pred_mask) / float(union))

    return float(np.mean(part_ious))


def chamfer_distance_and_fscore(pred_points: np.array, gt_points: np.array, threshold: float = 0.02) -> Tuple[float, float]:
    """Computes the bidirectional Chamfer Distance and F-score between two point sets.

    Args:
        pred_points (np.array): reconstructed surface points, [Np, 3].
        gt_points (np.array): ground-truth surface points, [Ng, 3].
        threshold (float): distance threshold for precision/recall used in the F-score.
    Returns:
        Tuple[float, float]: Chamfer Distance (sum of both directions' mean distance), F-score.
    """
    pred_tree = cKDTree(pred_points)
    gt_to_pred_dist, _ = pred_tree.query(gt_points)

    gt_tree = cKDTree(gt_points)
    pred_to_gt_dist, _ = gt_tree.query(pred_points)

    chamfer_dist = float(np.mean(gt_to_pred_dist) + np.mean(pred_to_gt_dist))

    precision = float(np.mean(pred_to_gt_dist < threshold))
    recall = float(np.mean(gt_to_pred_dist < threshold))
    fscore = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return chamfer_dist, fscore
