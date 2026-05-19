"""
Calculul metricilor clinice pe un fisier EDF cu ground truth.
Per-window si per-event.
"""
import numpy as np


def compute_window_metrics(starts_sec, alerts, ground_truth, window_sec=4):
    """
    Calculeaza metrici la nivel de fereastra.

    Args:
        starts_sec: array cu timestamp-urile de start ale fiecarei ferestre
        alerts: array binar cu alertele
        ground_truth: lista de (start, end) cu crizele reale in secunde
        window_sec: durata unei ferestre

    Returns:
        dict cu sensibility, specificity, precision, f1, fpr_per_hour, tp, fp, tn, fn
    """
    # y_true = 1 daca fereastra se suprapune cu o criza reala
    y_true = np.zeros(len(starts_sec), dtype=int)
    for gt_start, gt_end in ground_truth:
        for i, ws in enumerate(starts_sec):
            we = ws + window_sec
            if ws < gt_end and we > gt_start:
                y_true[i] = 1

    y_pred = alerts.astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    sens = tp / (tp + fn + 1e-10)
    spec = tn / (tn + fp + 1e-10)
    prec = tp / (tp + fp + 1e-10)
    f1 = 2 * prec * sens / (prec + sens + 1e-10)
    fpr_h = fp / (tn / (3600 / window_sec) + 1e-10)

    return {
        'sensitivity': sens,
        'specificity': spec,
        'precision': prec,
        'f1': f1,
        'fpr_per_hour': fpr_h,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    }


def compute_event_metrics(episodes, ground_truth):
    """
    Calculeaza metrici la nivel de eveniment (criza intreaga).

    Un TP este un episod detectat care se suprapune cu o criza reala.
    Un FP este un episod detectat fara suprapunere cu o criza reala.
    Un FN este o criza reala nedetectata (niciun episod se suprapune).

    Returns:
        dict cu n_gt_seizures, n_detected, tp, fp, fn, sensitivity, avg_latency_sec
    """
    n_gt = len(ground_truth)
    n_det = len(episodes)

    # Pentru fiecare criza reala, verificam daca a fost detectata
    gt_detected = [False] * n_gt
    latencies = []

    for i, (gt_start, gt_end) in enumerate(ground_truth):
        for ep in episodes:
            if ep['start_sec'] < gt_end and ep['end_sec'] > gt_start:
                gt_detected[i] = True
                # Latenta: cu cat a fost detectat inainte de debut (pozitiv = anticipare)
                latency = gt_start - ep['start_sec']
                latencies.append(latency)
                break

    # Pentru fiecare episod, verificam daca e TP sau FP
    ep_is_tp = [False] * n_det
    for j, ep in enumerate(episodes):
        for gt_start, gt_end in ground_truth:
            if ep['start_sec'] < gt_end and ep['end_sec'] > gt_start:
                ep_is_tp[j] = True
                break

    tp = sum(gt_detected)
    fn = n_gt - tp
    fp = sum(1 for x in ep_is_tp if not x)

    sens = tp / (n_gt + 1e-10) if n_gt > 0 else None
    avg_latency = np.mean(latencies) if latencies else None

    return {
        'n_gt_seizures': n_gt,
        'n_detected_episodes': n_det,
        'tp': tp, 'fp': fp, 'fn': fn,
        'sensitivity': sens,
        'avg_latency_sec': avg_latency,
        'latencies': latencies,
    }
