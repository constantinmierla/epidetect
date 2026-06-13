import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew
import pywt

from .preprocessing import FS

def extract_band_powers(signal_1ch, fs=FS):
    bands = {'delta': (0.5, 4.), 'theta': (4., 8.), 'alpha': (8., 13.),
             'beta': (13., 30.), 'gamma': (30., 45.)}
    nperseg = min(len(signal_1ch), fs * 2)
    freqs, psd = welch(signal_1ch, fs=fs, nperseg=nperseg)
    total_power = np.trapz(psd, freqs) + 1e-10
    feats = {}
    for bn, (flo, fhi) in bands.items():
        idx = (freqs >= flo) & (freqs < fhi)
        bp = np.trapz(psd[idx], freqs[idx])
        feats[f'{bn}_power']     = bp
        feats[f'{bn}_rel_power'] = bp / total_power
    feats['theta_alpha_ratio'] = (feats['theta_power'] + 1e-10) / (feats['alpha_power'] + 1e-10)
    feats['delta_beta_ratio']  = (feats['delta_power'] + 1e-10) / (feats['beta_power']  + 1e-10)
    edge_idx = np.where(np.cumsum(psd) >= 0.95 * np.sum(psd))[0]
    feats['spectral_edge'] = float(freqs[edge_idx[0]]) if len(edge_idx) > 0 else 0.0
    return feats

def extract_wavelet_features(signal_1ch, wavelet='db4', level=5):
    coeffs = pywt.wavedec(signal_1ch, wavelet, level=level)
    feats  = {}
    for i, c in enumerate(coeffs):
        if len(c) == 0:
            continue
        feats[f'wt_energy_{i}']  = float(np.sum(c ** 2))
        feats[f'wt_std_{i}']     = float(np.std(c))
        feats[f'wt_entropy_{i}'] = float(-np.sum(c ** 2 * np.log(c ** 2 + 1e-10)))
    return feats

def extract_statistical_features(signal_1ch):
    return {
        'mean':       float(np.mean(signal_1ch)),
        'std':        float(np.std(signal_1ch)),
        'variance':   float(np.var(signal_1ch)),
        'skewness':   float(skew(signal_1ch)),
        'kurtosis':   float(kurtosis(signal_1ch)),
        'rms':        float(np.sqrt(np.mean(signal_1ch ** 2))),
        'ptp':        float(np.ptp(signal_1ch)),
        'energy':     float(np.sum(signal_1ch ** 2)),
        'zero_cross': float(np.sum(np.diff(np.sign(signal_1ch)) != 0)),
    }

def extract_hjorth_features(signal_1ch):
    activity   = np.var(signal_1ch)
    d1         = np.diff(signal_1ch)
    d2         = np.diff(d1)
    mobility   = np.std(d1) / (np.std(signal_1ch) + 1e-10)
    complexity = (np.std(d2) / (np.std(d1) + 1e-10)) / (mobility + 1e-10)
    return {'hjorth_activity':   float(activity),
            'hjorth_mobility':   float(mobility),
            'hjorth_complexity': float(complexity)}

def extract_nonlinear_features(signal_1ch):
    hist, _ = np.histogram(signal_1ch, bins=20, density=True)
    hist    = hist[hist > 0]
    shannon = -np.sum(hist * np.log2(hist + 1e-10))
    _, psd  = welch(signal_1ch, nperseg=min(len(signal_1ch), 256))
    psd_norm     = psd / (np.sum(psd) + 1e-10)
    spec_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    n = len(signal_1ch)
    hurst_approx = np.log(np.std(signal_1ch[:n // 2]) / (np.std(signal_1ch) + 1e-10) + 1e-10)
    return {'shannon_entropy':  float(shannon),
            'spectral_entropy': float(spec_entropy),
            'hurst_approx':     float(hurst_approx)}

def extract_all_features(window_data):
    all_feats = {}
    for ch_idx in range(window_data.shape[0]):
        sig    = window_data[ch_idx]
        prefix = f'ch{ch_idx:02d}'
        ch_feats = {}
        ch_feats.update(extract_statistical_features(sig))
        ch_feats.update(extract_hjorth_features(sig))
        ch_feats.update(extract_band_powers(sig))
        ch_feats.update(extract_wavelet_features(sig))
        ch_feats.update(extract_nonlinear_features(sig))
        for k, v in ch_feats.items():
            all_feats[f'{prefix}_{k}'] = v

    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.corrcoef(window_data)

    upper_tri = corr[np.triu_indices_from(corr, k=1)]
    all_feats['mean_cross_corr'] = float(np.mean(np.abs(upper_tri)))
    all_feats['std_cross_corr']  = float(np.std(upper_tri))
    all_feats['max_cross_corr']  = float(np.max(np.abs(upper_tri)))

    fvec = np.array(list(all_feats.values()), dtype=np.float32)
    return np.nan_to_num(fvec, nan=0., posinf=0., neginf=0.), list(all_feats.keys())

def extract_features_batch(windows, progress_callback=None):
    n_windows = len(windows)
    all_features = []
    feature_names = None

    for i, win in enumerate(windows):
        fvec, fnames = extract_all_features(win)
        all_features.append(fvec)
        if feature_names is None:
            feature_names = fnames
        if progress_callback is not None and i % 50 == 0:
            progress_callback(i, n_windows)

    if progress_callback is not None:
        progress_callback(n_windows, n_windows)

    return np.array(all_features, dtype=np.float32), feature_names