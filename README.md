# 🧠 Seizure Detection System

Aplicație web pentru detecția automată a crizelor epileptice din semnale EEG, folosind un ensemble LightGBM + EEGNet antrenat pe CHB-MIT Scalp EEG Database.

Lucrare de licență, Universitatea Babeș-Bolyai, 2026.

## Caracteristici

- ✅ Upload fișiere EDF (montaj bipolar CHB-MIT sau monopolar Siena, cu re-referențiere automată)
- ✅ Timeline interactiv cu probabilitățile modelului și episoadele detectate
- ✅ Threshold slider dinamic — ajustezi pragul, rezultatele se actualizează instant
- ✅ Ground truth overlay — automat din fișierele `summary.txt` CHB-MIT sau manual
- ✅ Metrici în timp real (sensibilitate, specificitate, F1, FPR/h, per-window și per-event)
- ✅ Export CSV cu predicții și episoade detectate
- ✅ Dark mode modern, dashboard responsive

## Structură proiect

```
seizure_app/
├── app.py                     # Streamlit entry point
├── inference/                 # Logica ML
│   ├── preprocessing.py       # Preprocessing EEG (identic cu NB1)
│   ├── features.py            # Feature extraction (identic cu NB1)
│   ├── models.py              # EEGNet class definition
│   └── pipeline.py            # Inference orchestration
├── ui/
│   └── timeline.py            # Plotly visualizations
├── utils/
│   ├── edf_parser.py          # Parser CHB-MIT annotations
│   └── metrics.py             # Clinical metrics
├── models/
│   └── ensemble_v2.pkl        # Trained model (TREBUIE ADAUGAT)
├── .streamlit/
│   └── config.toml            # Dark theme config
└── requirements.txt
```

## Setup local

### 1. Clonează / descarcă proiectul

```bash
git clone <repo> seizure_app
cd seizure_app
```

### 2. Instalează dependințele

Recomandăm Python 3.11 sau 3.12. Pe Mac cu M-series, torch va folosi automat MPS.

```bash
python -m venv venv
source venv/bin/activate           # pe Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Adaugă modelul antrenat

Copiază fișierul `ensemble_v2.pkl` în directorul `models/`:

```bash
cp /cale/catre/ensemble_v2.pkl models/
```

### 4. Rulează aplicația

```bash
streamlit run app.py
```

Aplicația se deschide automat la `http://localhost:8501`.

## Deploy pe Streamlit Cloud

Streamlit Cloud e cel mai simplu mod de a obține un URL public gratuit.

### Pași

1. **Creează un repository GitHub** cu tot conținutul proiectului.

2. **Modelul > 25MB?** Activează Git LFS:
   ```bash
   git lfs install
   git lfs track "models/*.pkl"
   git add .gitattributes
   git add models/ensemble_v2.pkl
   git commit -m "Add model via LFS"
   git push
   ```

3. **Conectează Streamlit Cloud** la https://share.streamlit.io:
   - Sign in cu GitHub
   - "New app" → selectează repo-ul
   - Main file: `app.py`
   - Python version: 3.11 sau 3.12
   - Deploy

4. **Setări avansate în dashboard:**
   - Memory: 1GB (default)
   - Include secrets dacă e cazul (nu pentru această aplicație)

### Limitări Streamlit Cloud gratuit

- 1GB RAM (atent la încărcarea EDF-urilor mari)
- CPU only (EEGNet va rula mai lent, dar funcționează)
- Upload max 300MB (configurat în `config.toml`)
- Hibernează după 7 zile de inactivitate (prima cerere după hibernare e lentă)

## Utilizare

### Flow tipic

1. **Încarcă un EDF** din zona de upload
2. **Ajustează pragul** din sidebar (opțional)
3. **Adaugă ground truth** (automat / manual) pentru calculul metricilor
4. **Inspectează timeline-ul** — zoom, pan, hover pentru detalii
5. **Verifică episoadele detectate** în tabelul de mai jos
6. **Exportă** predicțiile sau episoadele ca CSV

### Formate ground truth acceptate (manual)

```
# Comentariu
130 145                 # secunde (start, end)
02:10 02:25             # MM:SS
10:30-11:15             # MM:SS cu separator -
00:05:30 00:06:15       # HH:MM:SS
```

## Arhitectură tehnică

### Preprocessing
- Filtru Butterworth bandpass 0.5-45 Hz, ordin 4, zero-phase
- Filtru notch 50 Hz, Q=30
- Resampling la 256 Hz dacă e nevoie
- Normalizare z-score per canal per fereastră
- Clipping la ±5 deviații standard
- Re-referențiere automată monopolar → bipolar (T3/T4/T5/T6 → T7/T8/P7/P8)

### Feature extraction (831 features/fereastră)
- Band powers (delta/theta/alpha/beta/gamma) absolute și relative
- Rapoarte theta/alpha, delta/beta
- Spectral edge frequency (95%)
- Wavelet Daubechies db4, 5 niveluri (energie, std, entropie)
- Parametri Hjorth (activitate, mobilitate, complexitate)
- Statistici clasice (skewness, kurtosis, RMS, zero-crossing rate...)
- Entropii Shannon și spectrală
- Cross-correlation inter-canal

### Modele
- **LightGBM**: gradient boosting pe features tabulare, 194 features selectate
- **EEGNet v2**: CNN compact pe semnal brut, F1=32, D=6, F2=64
- **Ensemble**: `p_final = 0.5 · p_lgbm + 0.5 · p_eegnet`

### Post-procesare
- Smoothing cu medie glisantă (window=15)
- Filtru consecutivitate minimă (min_consecutive=5)
- Grupare în episoade + calcul latență vs ground truth

## Troubleshooting

**"Model-ul nu a fost găsit"**  
→ Verifică că `models/ensemble_v2.pkl` există.

**Upload EDF eșuează / timeout**  
→ Fișierul e prea mare. Redu durata EDF-ului sau crește `maxUploadSize` în `.streamlit/config.toml`.

**"Prea puține canale comune"**  
→ Fișierul nu are cel puțin 10 din cele 18 canale bipolare standard. Verifică formatul.

**Inferență lentă pe Streamlit Cloud**  
→ Normal, CPU only. Un EDF de 30 min durează 1-3 minute.

**Modelul încarcă dar dă erori la predicție**  
→ Arhitectura EEGNet din `inference/models.py` trebuie să fie **identică** cu cea din antrenare. Dacă ai modificat-o, re-salvează modelul.

## Credits

- **CHB-MIT Scalp EEG Database** — PhysioNet
- **EEGNet** — Lawhern et al. (2018)
- **LightGBM** — Microsoft Research (Ke et al., 2017)
- **Streamlit** — https://streamlit.io
- **Plotly** — https://plotly.com

## Licență

Uz academic. Pentru alte utilizări, contactează autorul.
