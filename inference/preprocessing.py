"""
Preprocesare semnale EEG - identica cu cea din NB1.
Include si re-referentierea monopolar->bipolar pentru dataset-uri ca Siena.
"""
import numpy as np
import mne
from scipy.signal import butter, filtfilt, iirnotch, resample as scipy_resample


# Configuratie - trebuie sa fie identica cu cea din antrenare
FS = 256
WINDOW_SEC = 4
OVERLAP = 0.75
WINDOW_SAMPLES = int(WINDOW_SEC * FS)      # 1024
STEP_SAMPLES = int(WINDOW_SAMPLES * (1 - OVERLAP))  # 256

COMMON_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ',  'CZ-PZ'
]
N_CHANNELS = len(COMMON_CHANNELS)  # 18

# Perechi pentru re-referentiere monopolar->bipolar (pentru Siena)
SIENA_BIPOLAR_PAIRS = [
    ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
    ('FP1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    ('FP2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
    ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
    ('FZ', 'CZ'),  ('CZ', 'PZ'),
]


def bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=FS, order=4):
    """Filtru Butterworth bandpass cu faza zero (filtfilt)."""
    nyq = fs / 2.0
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)


def notch_filter(data, freq=50.0, fs=FS, Q=30.0):
    """Filtru notch pentru interferenta retelei electrice."""
    b, a = iirnotch(freq / (fs / 2), Q)
    return filtfilt(b, a, data, axis=-1)


def normalize_signal(data):
    """Z-score normalization per canal."""
    mean = data.mean(axis=-1, keepdims=True)
    std = data.std(axis=-1, keepdims=True)
    std = np.where(std < 1e-8, 1e-8, std)
    return (data - mean) / std


def clip_artifacts(data, threshold=5.0):
    """Clipare la +/- threshold*std pentru atenuarea artefactelor."""
    std = data.std(axis=-1, keepdims=True)
    return np.clip(data, -threshold * std, threshold * std)


def rereference_monopolar_to_bipolar(raw):
    """
    Converteste montajul monopolar (Siena) la bipolar (CHB-MIT).
    Returneaza array de shape (18, n_samples) si lista de canale lipsa.
    """
    data = raw.get_data()
    ch_map = {}
    for i, name in enumerate(raw.ch_names):
        clean_name = name.upper().replace('EEG ', '').strip()
        ch_map[clean_name] = i

    bipolar = np.zeros((N_CHANNELS, data.shape[1]), dtype=np.float32)
    missing = []
    for i, (a, b) in enumerate(SIENA_BIPOLAR_PAIRS):
        if a in ch_map and b in ch_map:
            bipolar[i] = data[ch_map[a]] - data[ch_map[b]]
        else:
            missing.append(f'{a}-{b}')
    return bipolar, missing


def load_and_preprocess_edf(edf_path, fs_target=FS):
    """
    Incarca un EDF, detecteaza montajul, aplica preprocessing complet.

    Returns:
        dict cu: data (np.array 18xN), fs, n_samples, duration_sec,
                 montage_type, channel_info, missing_channels
        sau None daca fisierul nu poate fi procesat.
    """
    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    except Exception as e:
        return {'error': f'MNE nu poate citi fisierul: {e}'}

    fs = int(raw.info['sfreq'])
    original_channels = list(raw.ch_names)

    # Detectam tipul de montaj (bipolar are '-' in numele canalelor)
    has_bipolar = any('-' in ch for ch in raw.ch_names)

    missing = []
    if has_bipolar:
        # CHB-MIT flow
        ch_map = {ch.upper().strip(): ch for ch in raw.ch_names}
        avail = [ch_map[t.upper().strip()] for t in COMMON_CHANNELS
                 if t.upper().strip() in ch_map]
        if len(avail) < 10:
            return {'error': f'Prea putine canale comune ({len(avail)}/18). '
                             f'Verifica ca fisierul respecta standardul CHB-MIT.'}
        raw.pick_channels(avail)
        data = raw.get_data() * 1e6  # microVolti
        montage_type = 'bipolar (CHB-MIT)'
        # Calculam canalele lipsa
        found_set = set([c.upper() for c in avail])
        missing = [c for c in COMMON_CHANNELS if c.upper() not in found_set]
    else:
        # Siena/monopolar -> rereferentiere
        data, missing = rereference_monopolar_to_bipolar(raw)
        data = data * 1e6
        montage_type = 'monopolar -> bipolar (Siena)'

    # Padding cu zero daca avem mai putin de 18 canale
    if data.shape[0] < N_CHANNELS:
        pad = np.zeros((N_CHANNELS - data.shape[0], data.shape[1]), dtype=np.float32)
        data = np.vstack([data, pad])

    # Resampling la fs_target
    was_resampled = False
    if fs != fs_target:
        new_len = int(data.shape[1] * fs_target / fs)
        data = scipy_resample(data, new_len, axis=1).astype(np.float32)
        was_resampled = True
        original_fs = fs
        fs = fs_target
    else:
        original_fs = fs

    # Aplicam filtrarea si normalizarea
    data = bandpass_filter(data, fs=fs)
    data = notch_filter(data, fs=fs)
    data = clip_artifacts(data)
    data = normalize_signal(data)

    return {
        'data': data.astype(np.float32),
        'fs': fs,
        'original_fs': original_fs,
        'was_resampled': was_resampled,
        'n_samples': data.shape[1],
        'duration_sec': data.shape[1] / fs,
        'montage_type': montage_type,
        'original_channels': original_channels,
        'n_channels_found': sum(1 for i in range(N_CHANNELS) if np.any(data[i] != 0)),
        'missing_channels': missing,
    }


def segment_into_windows(data, window_samples=WINDOW_SAMPLES, step_samples=STEP_SAMPLES):
    """
    Segmenteaza semnalul in ferestre cu suprapunere.
    Returneaza array de shape (n_windows, n_channels, window_samples)
    si array cu timestamp-urile de start (in secunde) pentru fiecare fereastra.
    """
    n_samples = data.shape[1]
    windows = []
    starts_sec = []
    start = 0
    while start + window_samples <= n_samples:
        windows.append(data[:, start:start + window_samples])
        starts_sec.append(start / FS)
        start += step_samples
    return np.array(windows, dtype=np.float32), np.array(starts_sec)
