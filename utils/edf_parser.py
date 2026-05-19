"""
Parser pentru fisierele de adnotare CHB-MIT (.seizures si summary.txt).
Permite overlay-ul automat al ground truth in UI.
"""
import re
import struct
from pathlib import Path
from collections import defaultdict


def parse_summary_file(summary_path):
    """
    Parseaza un fisier chbXX-summary.txt si intoarce un dict cu:
    { 'chbXX_YY.edf': [(start_sec, end_sec), ...], ... }
    """
    content = Path(summary_path).read_text(encoding='utf-8', errors='ignore')
    seizures = defaultdict(list)
    file_blocks = re.split(r'(?=File Name:)', content)

    for block in file_blocks:
        if not block.strip():
            continue
        fname_match = re.search(r'File Name:\s*(\S+\.edf)', block, re.IGNORECASE)
        if not fname_match:
            continue
        fname = fname_match.group(1).strip()

        n_sz_match = re.search(r'Number of Seizures in File:\s*(\d+)', block)
        if not n_sz_match:
            continue
        n_seizures = int(n_sz_match.group(1))

        if n_seizures == 0:
            seizures[fname] = []
            continue

        starts = re.findall(r'Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)', block)
        ends = re.findall(r'Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)', block)
        for s, e in zip(starts, ends):
            seizures[fname].append((int(s), int(e)))
    return dict(seizures)


def parse_seizures_binary_file(seizures_path):
    """
    Parseaza un fisier .seizures (format binar CHB-MIT).
    Este o metoda de fallback cand nu avem summary.
    Din pacate formatul .seizures nu e documentat public si poate varia.
    Pentru siguranta, utilizatorul ar trebui sa foloseasca parse_summary_file.

    Returneaza lista de (start_sec, end_sec) sau lista goala daca nu poate parsea.
    """
    try:
        data = Path(seizures_path).read_bytes()
        # Format euristic: cautam perechi de timestamps ca int32 big-endian
        # Nu e garantat corect - folosit ca fallback
        if len(data) < 16:
            return []
        # Heuristic basic: parsam pachete de 4 bytes ca timestamps
        # In practica utilizatorul trebuie sa foloseasca summary.txt
        return []
    except Exception:
        return []


def find_ground_truth_for_edf(edf_path):
    """
    Pentru un EDF dat, incearca sa gaseasca intervalele de crize din:
    1. Fisierul chbXX-summary.txt din acelasi director
    2. (viitor) fisierul .seizures binar

    Returneaza lista de (start_sec, end_sec) sau None daca nu gaseste nimic.
    """
    edf_path = Path(edf_path)
    edf_name = edf_path.name
    parent = edf_path.parent

    # Incercam summary din directorul curent sau din parent
    for summary_dir in [parent, parent.parent]:
        # Pattern: chb05-summary.txt pentru chb05_16.edf
        match = re.match(r'(chb\d+)', edf_name, re.IGNORECASE)
        if match:
            patient_id = match.group(1).lower()
            summary_candidates = [
                summary_dir / f'{patient_id}-summary.txt',
                summary_dir / f'{patient_id.upper()}-summary.txt',
                summary_dir / patient_id / f'{patient_id}-summary.txt',
            ]
            for cand in summary_candidates:
                if cand.exists():
                    seizures_dict = parse_summary_file(cand)
                    if edf_name in seizures_dict:
                        return seizures_dict[edf_name]

    return None


def parse_manual_ground_truth(text):
    """
    Parseaza input manual de la utilizator pentru ground truth.
    Format acceptat (o criza pe linie):
        start end
        MM:SS MM:SS
        MM:SS-MM:SS
    Exemple:
        130 145
        02:10 02:25
        10:30-11:15

    Returneaza lista de (start_sec, end_sec) sau lista goala.
    """
    if not text or not text.strip():
        return []

    results = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # Normalizam separatori
        line = line.replace('-', ' ').replace(',', ' ').replace(';', ' ')
        parts = [p for p in line.split() if p]
        if len(parts) < 2:
            continue
        try:
            start = _parse_time(parts[0])
            end = _parse_time(parts[1])
            if start is not None and end is not None and end > start:
                results.append((start, end))
        except Exception:
            continue
    return results


def _parse_time(s):
    """Converteste string in secunde. Accepta: '130', '02:10', '01:23:45'."""
    if ':' in s:
        parts = s.split(':')
        parts = [int(p) for p in parts]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        elif len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
    else:
        return int(float(s))
    return None
