"""
Pipeline complet de inferenta: incarca modelul, proceseaza un EDF, intoarce rezultate.
"""
import pickle
import io
import numpy as np
import torch
from pathlib import Path
from scipy.ndimage import uniform_filter1d

from .preprocessing import (load_and_preprocess_edf, segment_into_windows,
                             N_CHANNELS, WINDOW_SAMPLES, FS)
from .features import extract_features_batch
from .models import EEGNet, get_device


# ============================================================================
# CUSTOM UNPICKLER PENTRU CPU
# ============================================================================
class CPU_Unpickler(pickle.Unpickler):
    """
    Forteaza tensorii PyTorch salvati ca byte-strings in interiorul
    dictionarului sa se incarce pe CPU, prevenind erorile pe Mac/masini fara CUDA.
    """
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def load_model(model_path):
    """
    Incarca modelul ensemble din pickle.
    Returneaza dict cu toate componentele necesare pentru inferenta.
    """
    with open(model_path, 'rb') as f:
        # Folosim unpickler-ul custom in loc de torch.load
        ckpt = CPU_Unpickler(f).load()

    # LightGBM
    lgbm_model = ckpt['lgbm_model']
    lgbm_scaler = ckpt['lgbm_scaler']
    selected_features = ckpt.get('selected_features', None)

    # EEGNet
    eegnet = ckpt.get('eegnet_model', None)
    has_eegnet = eegnet is not None

    # Parametri ensemble
    alpha_lgbm = ckpt['alpha_lgbm']
    alpha_eegnet = ckpt['alpha_eegnet']
    default_threshold = ckpt['final_threshold']

    device = get_device()
    if has_eegnet:
        eegnet = eegnet.to(device)
        eegnet.eval()

    return {
        'lgbm_model': lgbm_model,
        'lgbm_scaler': lgbm_scaler,
        'selected_features': selected_features,
        'eegnet_model': eegnet,
        'has_eegnet': has_eegnet,
        'alpha_lgbm': alpha_lgbm,
        'alpha_eegnet': alpha_eegnet,
        'default_threshold': default_threshold,
        'device': device,
        'cv_metrics_lgb': ckpt.get('cv_metrics_lgb', None),
        'cv_metrics_eeg': ckpt.get('cv_metrics_eeg', None),
    }


def predict_lgbm(features, model_dict):
    """Predictie LightGBM pe features tabulare."""
    # 1. Curatam valorile infinite/NaN
    safe_features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # 2. Scalam pe cele 831 features originale
    X_scaled = model_dict['lgbm_scaler'].transform(safe_features)

    # 3. Selectam cele 194 features pe care a fost antrenat modelul LightGBM
    if model_dict['selected_features'] is not None:
        X_scaled = X_scaled[:, model_dict['selected_features']]

    # 4. Returnam predictia
    return model_dict['lgbm_model'].predict_proba(X_scaled)[:, 1]

def predict_eegnet(windows, model_dict, batch_size=64):
    """Predictie EEGNet pe semnal brut."""
    if not model_dict['has_eegnet']:
        return None

    device = model_dict['device']
    eegnet = model_dict['eegnet_model']
    probs = []

    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = windows[i:i + batch_size]
            # Normalizare per-fereastra (identic cu antrenare)
            mu = batch.mean(axis=-1, keepdims=True)
            sigma = batch.std(axis=-1, keepdims=True) + 1e-8
            batch_norm = (batch - mu) / sigma
            x = torch.tensor(batch_norm[:, np.newaxis], dtype=torch.float32).to(device)
            logits = eegnet(x)
            p = torch.sigmoid(logits).cpu().numpy()
            probs.append(p)

    return np.concatenate(probs)


def smooth_predictions(probs, window_size=15, min_consecutive=5, threshold=0.5):
    """
    Smoothing identic cu cel din antrenare.
    Returneaza (prob_smoothed, alerts_binary).
    """
    ps = uniform_filter1d(probs.astype(float), size=window_size)
    alerts = (ps >= threshold).astype(int)
    i = 0
    while i < len(alerts):
        if alerts[i] == 1:
            j = i
            while j < len(alerts) and alerts[j] == 1:
                j += 1
            if j - i < min_consecutive:
                alerts[i:j] = 0
            i = j
        else:
            i += 1
    return ps, alerts


def group_episodes(alerts, starts_sec, probs_smoothed, window_sec=4):
    """
    Grupeaza alertele consecutive in episoade.
    Returneaza lista de dict-uri cu start/end/duration/max_prob.
    """
    episodes = []
    i = 0
    while i < len(alerts):
        if alerts[i] == 1:
            j = i
            while j < len(alerts) and alerts[j] == 1:
                j += 1
            # Episod de la alerts[i] pana la alerts[j-1]
            ep_start = starts_sec[i]
            ep_end = starts_sec[j - 1] + window_sec
            max_prob = float(probs_smoothed[i:j].max())
            episodes.append({
                'start_sec': ep_start,
                'end_sec': ep_end,
                'duration_sec': ep_end - ep_start,
                'max_prob': max_prob,
                'n_windows': j - i,
            })
            i = j
        else:
            i += 1
    return episodes


def run_inference(edf_path, model_dict,
                  threshold=None,
                  smoothing_window=15,
                  min_consecutive=5,
                  progress_callback=None):
    """
    Ruleaza pipeline-ul complet de inferenta pe un EDF.

    Args:
        edf_path: cale catre fisierul EDF
        model_dict: dict-ul returnat de load_model()
        threshold: prag custom (None = foloseste default-ul modelului)
        smoothing_window: marime fereastra smoothing
        min_consecutive: minim alerte consecutive pentru episod
        progress_callback: functie(stage: str, progress: float in [0,1])

    Returns:
        dict cu toate rezultatele inferentei
    """
    if threshold is None:
        threshold = model_dict['default_threshold']

    # 1. Incarcare + preprocesare
    if progress_callback:
        progress_callback('Citire EDF...', 0.05)
    prep = load_and_preprocess_edf(edf_path)
    if 'error' in prep:
        return {'error': prep['error']}

    # 2. Segmentare in ferestre
    if progress_callback:
        progress_callback('Segmentare in ferestre...', 0.15)
    windows, starts_sec = segment_into_windows(prep['data'])
    n_windows = len(windows)

    if n_windows == 0:
        return {'error': 'Fisierul este prea scurt pentru analiza (minim 4 secunde).'}

    # 3. Extractie features
    def feat_progress(i, total):
        if progress_callback:
            # 20% -> 60% pentru features
            progress_callback(f'Extragere features ({i}/{total})...',
                              0.2 + 0.4 * (i / total))

    features, feature_names = extract_features_batch(windows,
                                                      progress_callback=feat_progress)

    # 4. LightGBM
    if progress_callback:
        progress_callback('Rulare LightGBM...', 0.65)
    probs_lgbm = predict_lgbm(features, model_dict)

    # 5. EEGNet
    probs_eegnet = None
    if model_dict['has_eegnet']:
        if progress_callback:
            progress_callback('Rulare EEGNet...', 0.75)
        probs_eegnet = predict_eegnet(windows, model_dict)

    # 6. Ensemble
    if probs_eegnet is not None:
        probs_ensemble = (model_dict['alpha_lgbm'] * probs_lgbm +
                          model_dict['alpha_eegnet'] * probs_eegnet)
    else:
        probs_ensemble = probs_lgbm

    # 7. Smoothing + alertele
    if progress_callback:
        progress_callback('Post-procesare...', 0.92)
    probs_smoothed, alerts = smooth_predictions(
        probs_ensemble, window_size=smoothing_window,
        min_consecutive=min_consecutive, threshold=threshold)

    # 8. Grupare in episoade
    episodes = group_episodes(alerts, starts_sec, probs_smoothed)

    if progress_callback:
        progress_callback('Gata!', 1.0)

    return {
        'file_info': {
            'duration_sec': prep['duration_sec'],
            'fs': prep['fs'],
            'original_fs': prep['original_fs'],
            'was_resampled': prep['was_resampled'],
            'montage_type': prep['montage_type'],
            'n_channels_found': prep['n_channels_found'],
            'missing_channels': prep['missing_channels'],
            'original_channels': prep['original_channels'],
        },
        'inference': {
            'n_windows': n_windows,
            'threshold': threshold,
            'probs_lgbm': probs_lgbm,
            'probs_eegnet': probs_eegnet,
            'probs_ensemble': probs_ensemble,
            'probs_smoothed': probs_smoothed,
            'alerts': alerts,
            'starts_sec': starts_sec,
        },
        'episodes': episodes,
        'raw_data': prep['data'],  # pentru vizualizare EEG brut in iter. 3
        'features': features,       # pentru SHAP in iter. 3
        'feature_names': feature_names,
    }