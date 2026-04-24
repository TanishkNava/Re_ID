import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

import lap


def bbox_ious(atlbrs, btlbrs):
    """Pure numpy IoU — replaces cython_bbox."""
    atlbrs = np.array(atlbrs)
    btlbrs = np.array(btlbrs)
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if ious.size == 0:
        return ious
    ax1,ay1,ax2,ay2 = atlbrs[:,0],atlbrs[:,1],atlbrs[:,2],atlbrs[:,3]
    bx1,by1,bx2,by2 = btlbrs[:,0],btlbrs[:,1],btlbrs[:,2],btlbrs[:,3]
    for i in range(len(atlbrs)):
        xx1 = np.maximum(ax1[i], bx1)
        yy1 = np.maximum(ay1[i], by1)
        xx2 = np.minimum(ax2[i], bx2)
        yy2 = np.minimum(ay2[i], by2)
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_a = (ax2[i] - ax1[i]) * (ay2[i] - ay1[i])
        area_b = (bx2 - bx1) * (by2 - by1)
        ious[i] = inter / (area_a + area_b - inter + 1e-7)
    return ious


def linear_assignment(cost_matrix, thresh=np.inf):
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    matches = np.array([[y[i], i] for i in x if i >= 0])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    ious_ = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if ious_.size == 0:
        return ious_
    ious_ = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=float),
        np.ascontiguousarray(btlbrs, dtype=float),
    )
    return ious_


def iou_distance(atracks, btracks):
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious
    return cost_matrix


def embedding_distance(tracks, detections, metric="cosine"):
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray(
        [track.curr_feat for track in detections], dtype=float
    )
    track_features = np.asarray(
        [track.smooth_feat for track in tracks], dtype=float
    )
    cost_matrix = np.maximum(
        0.0, cdist(track_features, det_features, metric)
    )
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = KalmanFilter.chi2inv95[gating_dim]
    measurements = np.asarray(
        [det.to_xyah() for det in detections]
    )
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position
        )
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(
        cost_matrix.shape[0], axis=0
    )
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost