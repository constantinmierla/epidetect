"""
EpiDetect - Streamlit UI

Sistem complet pentru detectia crizelor epileptice din fisiere EDF.
Features:
- Upload EDF + detectare automata montaj (bipolar/monopolar)
- Timeline interactiv cu episoade si ground truth
- Threshold slider dinamic
- Zoom pe semnal EEG brut
- Explicabilitate SHAP per fereastra
- Metrici per-window si per-event
- Export CSV si PDF

Autor: Mierlă Constantin, Lucrare de licență, UBB Cluj-Napoca, 2026
"""
import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import time
import __main__

# Note: Change 'models' if your file is named something else, like 'networks.py'
from inference.models import EEGNet

# Trick pickle into finding the class
__main__.EEGNet = EEGNet
from pathlib import Path

from inference import load_model, run_inference, N_CHANNELS, WINDOW_SEC
from inference.pipeline import smooth_predictions, group_episodes

from ui import (create_timeline_figure, create_episodes_table_data, format_time,
                create_eeg_viewer, find_episode_for_time,
                compute_shap_for_window, create_shap_plot,
                create_shap_summary_by_category, prettify_feature_name)

from utils import (find_ground_truth_for_edf, parse_manual_ground_truth,
                   compute_window_metrics, compute_event_metrics,
                   generate_pdf_report)


# ============================================================================
# CONFIGURARE PAGINA
# ============================================================================
st.set_page_config(
    page_title='EpiDetect',
    page_icon='🧠',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    h1, h2, h3 { color: #fafafa; }

    [data-testid="stMetric"] {
        background-color: #1a1f2e;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #2a2f3e;
    }
    [data-testid="stMetricValue"] { color: #00d9ff !important; }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00d9ff 0%, #0099cc 100%);
        border: none;
        color: #0e1117;
        font-weight: 600;
    }

    .streamlit-expanderHeader {
        background-color: #1a1f2e;
        border-radius: 8px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0e1117;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1f2e;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00d9ff !important;
        color: #0e1117 !important;
    }

    .dataframe { font-size: 13px; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# INCARCARE MODEL
# ============================================================================
@st.cache_resource(show_spinner=False)
def get_model():
    model_path = Path('models/eeg_detection.pkl')
    if not model_path.exists():
        return None
    return load_model(model_path)


# ============================================================================
# HEADER
# ============================================================================
col_title, col_info = st.columns([3, 1])
with col_title:
    st.markdown("""
    # 🧠 EpiDetect
    **Detectie si monitorizare a crizelor epileptice din EEG**  
    """)

with col_info:
    model = get_model()
    if model is None:
        st.error('Model indisponibil')
    else:
        device_emoji = {'cuda': '🚀 CUDA', 'mps': '🍎 MPS', 'cpu': '💻 CPU'}
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a1f2e 0%, #0e1117 100%);
                    padding: 12px; border-radius: 8px; border: 1px solid #2a2f3e;
                    margin-top: 18px;'>
            <small style='color: #00d9ff; font-weight: 600;'>MODEL STATUS</small><br>
            <span style='color: #00ff99;'>● Active</span><br>
            <small>Device: {device_emoji.get(model['device'], model['device'])}</small>
        </div>
        """, unsafe_allow_html=True)

st.markdown('---')

if model is None:
    st.error("""
    ⚠️ **Model-ul nu a fost gasit**

    Aseaza fisierul `eeg_detection.pkl` in directorul `models/` si reporneste aplicatia.
    """)
    st.stop()


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown('### ⚙️ Configurare inferenta')

    threshold = st.slider(
        'Prag de decizie', 0.05, 0.95,
        value=float(model['default_threshold']), step=0.01,
        help='Pragul optimizat pe validare este ' +
             f'{model["default_threshold"]:.3f}. Valorile mai mici cresc sensibilitatea.'
    )

    with st.expander('🔧 Parametri avansati'):
        smoothing_window = st.slider(
            'Fereastra smoothing (esantioane)',
            1, 30, value=15, step=1
        )
        min_consecutive = st.slider(
            'Minim ferestre consecutive', 1, 10, value=5, step=1
        )

    st.markdown('---')
    st.markdown('### 📊 Ground Truth')

    gt_mode = st.radio(
        'Sursa ground truth',
        ['Fara ground truth', 'Automat (summary.txt)', 'Manual'],
    )

    manual_gt_text = None
    if gt_mode == 'Manual':
        manual_gt_text = st.text_area(
            'Intervale crize',
            placeholder='Exemplu:\n130 145\n02:10 02:25\n10:30-11:15',
            height=120,
        )

    st.markdown('---')

    if model.get('cv_metrics_lgb') is not None:
        with st.expander('📈 Metrici CV model'):
            df_lgb = pd.DataFrame(model['cv_metrics_lgb'])
            df_eeg = pd.DataFrame(model.get('cv_metrics_eeg', []))
            if not df_lgb.empty:
                st.markdown('**LightGBM**')
                st.markdown(f"- F1: {df_lgb['f1'].mean():.3f} ± {df_lgb['f1'].std():.3f}")
                st.markdown(f"- AUC: {df_lgb['auc'].mean():.3f} ± {df_lgb['auc'].std():.3f}")
                st.markdown(f"- Sens: {df_lgb['sensitivity'].mean():.3f}")
            if not df_eeg.empty:
                st.markdown('**EEGNet**')
                st.markdown(f"- F1: {df_eeg['f1'].mean():.3f} ± {df_eeg['f1'].std():.3f}")
                st.markdown(f"- AUC: {df_eeg['auc'].mean():.3f} ± {df_eeg['auc'].std():.3f}")
                st.markdown(f"- Sens: {df_eeg['sensitivity'].mean():.3f}")

    st.markdown("""
    <small style='color: #666;'>
    EpiDetect<br>
    Mierlă Constantin | UBB 2026
    </small>
    """, unsafe_allow_html=True)


# ============================================================================
# UPLOAD
# ============================================================================
st.markdown('### 📁 Incarcare fisier EEG')
uploaded_file = st.file_uploader(
    'Selecteaza un fisier EDF', type=['edf'],
    help='Format EDF (European Data Format). Accepta bipolar CHB-MIT '
         'si monopolar Siena (conversie automata).'
)

if uploaded_file is None:
    st.info("""
    **Incarca un fisier EDF pentru a incepe analiza.**

    **Formate acceptate:**
    - Montaj bipolar CHB-MIT (18 canale standard)
    - Montaj monopolar Siena (convertit automat la bipolar)
    
    """)
    st.stop()


# ============================================================================
# INFERENTA
# ============================================================================
with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp:
    tmp.write(uploaded_file.getvalue())
    tmp_path = tmp.name


@st.cache_data(show_spinner=False)
def run_cached_inference(file_hash, _tmp_path):
    progress_container = st.empty()
    progress_bar = progress_container.progress(0.0)
    progress_text = st.empty()

    def progress_callback(stage, progress):
        progress_bar.progress(min(progress, 1.0))
        progress_text.markdown(
            f'<small style="color: #00d9ff;">⏳ {stage}</small>',
            unsafe_allow_html=True
        )

    result = run_inference(
        _tmp_path, model,
        threshold=model['default_threshold'],
        progress_callback=progress_callback
    )
    progress_container.empty()
    progress_text.empty()
    return result


file_hash = hash(uploaded_file.getvalue())
t_start = time.time()
result = run_cached_inference(file_hash, tmp_path)
inference_time = time.time() - t_start

try:
    Path(tmp_path).unlink()
except Exception:
    pass

if 'error' in result:
    st.error(f'❌ **Eroare la procesare:** {result["error"]}')
    st.stop()


# Reaplicare threshold custom fara rerun
inf = result['inference']
probs_smoothed_new, alerts_new = smooth_predictions(
    inf['probs_ensemble'],
    window_size=smoothing_window,
    min_consecutive=min_consecutive,
    threshold=threshold
)
episodes_new = group_episodes(alerts_new, inf['starts_sec'],
                                probs_smoothed_new, window_sec=WINDOW_SEC)

result['inference']['threshold'] = threshold
result['inference']['probs_smoothed'] = probs_smoothed_new
result['inference']['alerts'] = alerts_new
result['episodes'] = episodes_new


# ============================================================================
# INFO FISIER
# ============================================================================
fi = result['file_info']

st.markdown('### 📄 Informatii fisier')
info_cols = st.columns(5)
info_cols[0].metric(
    'Fisier',
    uploaded_file.name[:20] + ('...' if len(uploaded_file.name) > 20 else '')
)
info_cols[1].metric('Durata', format_time(fi['duration_sec']))
info_cols[2].metric(
    'Frecventa',
    f"{fi['original_fs']} Hz" + (' → 256' if fi['was_resampled'] else '')
)
info_cols[3].metric('Canale', f"{fi['n_channels_found']}/18")

if fi['was_resampled']:
    st.info(f'ℹ️ Semnal resamplat de la {fi["original_fs"]} Hz la 256 Hz.')
if 'monopolar' in fi['montage_type']:
    st.info(
        f'ℹ️ **{fi["montage_type"]}** | '
        'canalele T3/T4/T5/T6 mapate la T7/T8/P7/P8.'
    )
if fi['missing_channels']:
    missing_str = ', '.join(fi['missing_channels'][:5])
    more = f' (+{len(fi["missing_channels"]) - 5} altele)' if len(fi['missing_channels']) > 5 else ''

# ============================================================================
# GROUND TRUTH
# ============================================================================
ground_truth = None

if gt_mode == 'Automat (summary.txt)':
    ground_truth = find_ground_truth_for_edf(uploaded_file.name)
    if ground_truth is None:
        st.info('ℹ️ Nu am gasit ground truth automat. Folosind doar predictiile modelului.')
    else:
        st.success(f'✅ Ground truth incarcat: {len(ground_truth)} criza/crize.')
elif gt_mode == 'Manual' and manual_gt_text:
    ground_truth = parse_manual_ground_truth(manual_gt_text)
    if ground_truth:
        st.success(f'✅ {len(ground_truth)} interval(e) de criza parsate.')
    else:
        st.warning('⚠️ Nu am putut parsa niciun interval. Verifica formatul.')


# ============================================================================
# METRICI (daca avem ground truth)
# ============================================================================
win_metrics = None
evt_metrics = None

if ground_truth:
    win_metrics = compute_window_metrics(
        inf['starts_sec'], alerts_new, ground_truth, window_sec=WINDOW_SEC
    )
    evt_metrics = compute_event_metrics(episodes_new, ground_truth)

    st.markdown('### 📊 Metrici pe acest fisier')
    metric_cols = st.columns(6)
    metric_cols[0].metric('Sensibilitate (win)', f"{win_metrics['sensitivity']:.2%}")
    metric_cols[1].metric('Specificitate', f"{win_metrics['specificity']:.2%}")
    metric_cols[2].metric('F1 Score', f"{win_metrics['f1']:.3f}")
    metric_cols[3].metric('FPR / ora', f"{win_metrics['fpr_per_hour']:.2f}")
    metric_cols[4].metric(
        'Crize prinse',
        f"{evt_metrics['tp']}/{evt_metrics['n_gt_seizures']}"
    )
    if evt_metrics['avg_latency_sec'] is not None:
        lat = evt_metrics['avg_latency_sec']
        lat_str = f'+{lat:.0f}s' if lat > 0 else f'{lat:.0f}s'
        metric_cols[5].metric(
            'Latenta medie', lat_str,
            help='Pozitiv = detectat inainte de debut clinic'
        )
    else:
        metric_cols[5].metric('Latenta medie', '-')

    with st.expander('🔍 Detalii confusion matrix (per fereastra)'):
        cm_cols = st.columns(4)
        cm_cols[0].markdown(f'**TP:** {win_metrics["tp"]:,}')
        cm_cols[1].markdown(f'**FP:** {win_metrics["fp"]:,}')
        cm_cols[2].markdown(f'**TN:** {win_metrics["tn"]:,}')
        cm_cols[3].markdown(f'**FN:** {win_metrics["fn"]:,}')


# ============================================================================
# TAB-URI PENTRU VIZUALIZARI
# ============================================================================
tab_timeline, tab_eeg, tab_shap, tab_episodes, tab_export = st.tabs([
    '📈 Timeline',
    '🌊 Semnal EEG',
    '🔬 Explicabilitate (SHAP)',
    '🔴 Episoade',
    '💾 Export',
])


# ---------- TAB 1: TIMELINE ----------
with tab_timeline:
    st.markdown('#### Timeline probabilitati')

    viz_controls = st.columns([1, 3])
    with viz_controls[0]:
        show_components = st.checkbox(
            'Afiseaza componente (LGBM / EEGNet)', value=False,
            key='show_comp_timeline'
        )

    fig_timeline = create_timeline_figure(
        result, show_components=show_components, ground_truth=ground_truth
    )
    st.plotly_chart(fig_timeline, use_container_width=True, config={
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d'],
    })

    # Summary
    sum_cols = st.columns(4)
    sum_cols[0].metric('Episoade detectate', len(episodes_new))
    sum_cols[1].metric(
        'Durata totala alerte',
        f'{sum(ep["duration_sec"] for ep in episodes_new):.0f}s'
                        if episodes_new else '0s'
    )
    if episodes_new:
        sum_cols[2].metric(
            'Probabilitate max',
            f'{max(ep["max_prob"] for ep in episodes_new):.3f}'
        )
    else:
        sum_cols[2].metric('Probabilitate max', '-')
    sum_cols[3].metric(
        'Avg prob (overall)',
        f'{float(inf["probs_ensemble"].mean()):.3f}'
    )


# ---------- TAB 2: EEG VIEWER ----------
with tab_eeg:
    st.markdown('#### Vizualizare semnal EEG brut')
    st.markdown("""
    <small style='color: #888;'>
    Selecteaza un moment in timp pentru a vedea semnalul brut pe toate cele 18 canale.
    Intervalele marcate cu rosu sunt episoadele detectate. Cele marcate cu verde sunt
    crizele reale (daca exista ground truth).
    </small>
    """, unsafe_allow_html=True)

    eeg_controls = st.columns([2, 1, 1])

    with eeg_controls[0]:
        # Quick jump la un episod
        jump_options = ['Selecteaza manual...']
        if episodes_new:
            for i, ep in enumerate(episodes_new[:20], 1):
                jump_options.append(
                    f'Episod {i}: {format_time(ep["start_sec"])} '
                    f'(prob {ep["max_prob"]:.2f})'
                )
        if ground_truth:
            for i, (gs, ge) in enumerate(ground_truth, 1):
                jump_options.append(
                    f'Criza reala {i}: {format_time(gs)} - {format_time(ge)}'
                )

        jump_selection = st.selectbox('Navigare rapida', jump_options)

    # Determinam center_sec din selectie sau slider
    if jump_selection.startswith('Episod'):
        ep_idx = int(jump_selection.split(':')[0].split(' ')[1]) - 1
        center_sec = (episodes_new[ep_idx]['start_sec'] +
                      episodes_new[ep_idx]['end_sec']) / 2
    elif jump_selection.startswith('Criza reala'):
        gt_idx = int(jump_selection.split(':')[0].split(' ')[2]) - 1
        center_sec = (ground_truth[gt_idx][0] + ground_truth[gt_idx][1]) / 2
    else:
        with eeg_controls[1]:
            center_sec = st.number_input(
                'Timp (secunde)',
                min_value=0.0,
                max_value=float(fi['duration_sec']),
                value=float(fi['duration_sec']) / 2,
                step=5.0,
            )

    with eeg_controls[2]:
        context_sec = st.select_slider(
            'Context',
            options=[10, 20, 30, 60, 120],
            value=30,
            format_func=lambda x: f'{x}s'
        )

    # Gasim episodul pentru highlight daca e unul in zona
    ep_for_highlight = find_episode_for_time(episodes_new, center_sec)
    highlight_start = ep_for_highlight['start_sec'] if ep_for_highlight else None
    highlight_end = ep_for_highlight['end_sec'] if ep_for_highlight else None

    fig_eeg = create_eeg_viewer(
        result['raw_data'],
        center_sec=center_sec,
        context_sec=context_sec,
        highlight_start_sec=highlight_start,
        highlight_end_sec=highlight_end,
        ground_truth_intervals=ground_truth,
    )
    st.plotly_chart(fig_eeg, use_container_width=True, config={
        'displayModeBar': True,
        'displaylogo': False,
    })

    # Info contextual
    info_text = f'**Timp central:** {format_time(center_sec)}'
    if ep_for_highlight:
        info_text += (
            f' | **In episod detectat:** {format_time(ep_for_highlight["start_sec"])} - '
            f'{format_time(ep_for_highlight["end_sec"])} '
            f'(prob max: {ep_for_highlight["max_prob"]:.3f})'
        )
    if ground_truth:
        for gs, ge in ground_truth:
            if gs <= center_sec <= ge:
                info_text += f' | **In criza reala:** {format_time(gs)} - {format_time(ge)}'
                break
    st.markdown(info_text)


# ---------- TAB 3: SHAP ----------
with tab_shap:
    st.markdown('#### Explicabilitate: ce features au contribuit?')
    st.markdown("""
    <small style='color: #888;'>
    Pentru o fereastra specifica, sistemul afiseaza contributiile SHAP ale features.
    Valorile <span style='color: #ff3366;'>rosii</span> (pozitive) impingered predictia
    spre "periculos", iar cele <span style='color: #00d9ff;'>albastre</span> (negative)
    spre "normal".
    </small>
    """, unsafe_allow_html=True)

    shap_controls = st.columns([2, 1])

    with shap_controls[0]:
        # Quick jump cu acelasi sistem ca tab-ul EEG
        shap_jump_options = ['Selecteaza manual...']
        if episodes_new:
            for i, ep in enumerate(episodes_new[:20], 1):
                shap_jump_options.append(
                    f'Episod {i}: {format_time(ep["start_sec"])} '
                    f'(prob {ep["max_prob"]:.2f})'
                )

        shap_jump = st.selectbox(
            'Navigare rapida', shap_jump_options, key='shap_jump'
        )

    with shap_controls[1]:
        if shap_jump.startswith('Episod'):
            ep_idx = int(shap_jump.split(':')[0].split(' ')[1]) - 1
            shap_time = (episodes_new[ep_idx]['start_sec'] +
                         episodes_new[ep_idx]['end_sec']) / 2
        else:
            shap_time = st.number_input(
                'Timp (secunde)',
                min_value=0.0,
                max_value=float(fi['duration_sec']),
                value=float(fi['duration_sec']) / 2,
                step=float(WINDOW_SEC),
                key='shap_time_input',
            )

    # Gasim fereastra cea mai apropiata de timpul selectat
    starts = inf['starts_sec']
    window_idx = int(np.argmin(np.abs(starts - shap_time)))
    actual_time = starts[window_idx]

    # Compute SHAP pentru aceasta fereastra
    # Compute SHAP pentru aceasta fereastra
    with st.spinner('Calculare SHAP...'):
        features_all = result['features'][window_idx]
        feature_names = result['feature_names']

        # 1. Curatam valorile infinite
        safe_features_all = np.nan_to_num(features_all, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. Scalam cele 831 features
        X_scaled = model['lgbm_scaler'].transform(safe_features_all.reshape(1, -1))[0]

        # 3. Selectam cele 194 features finale pentru SHAP
        if model['selected_features'] is not None:
            X_sel = X_scaled[model['selected_features']]
            selected_names = [feature_names[i] for i in model['selected_features']]
        else:
            X_sel = X_scaled
            selected_names = feature_names

        # 4. Calculam valorile SHAP
        shap_vals, base_val = compute_shap_for_window(X_sel, model['lgbm_model'])

    shap_cols = st.columns([2, 1])

    with shap_cols[0]:
        top_n = st.slider('Top N features', 5, 30, 15, key='shap_top_n')
        fig_shap, final_logit = create_shap_plot(
            shap_vals, selected_names, base_val, top_n=top_n
        )
        st.plotly_chart(fig_shap, use_container_width=True, config={
            'displayModeBar': False,
        })

    with shap_cols[1]:
        # Info fereastra
        prob_lgbm_w = inf['probs_lgbm'][window_idx]
        prob_ens_w = inf['probs_ensemble'][window_idx]

        st.markdown(f"""
        **Fereastra analizata:**
        - Timp: `{format_time(actual_time)}` - `{format_time(actual_time + WINDOW_SEC)}`
        - Prob LightGBM: **{prob_lgbm_w:.3f}**
        - Prob Ensemble: **{prob_ens_w:.3f}**
        """)

        if inf['probs_eegnet'] is not None:
            prob_eeg_w = inf['probs_eegnet'][window_idx]
            st.markdown(f'- Prob EEGNet: **{prob_eeg_w:.3f}**')

        # Categorii
        fig_cat = create_shap_summary_by_category(shap_vals, selected_names)
        st.plotly_chart(fig_cat, use_container_width=True, config={
            'displayModeBar': False,
        })


# ---------- TAB 4: EPISOADE ----------
with tab_episodes:
    st.markdown('#### Episoade detectate')

    if not episodes_new:
        st.info(
            'Nicio alerta detectata la pragul curent. '
            'Incearca sa scazi pragul in sidebar pentru mai multa sensibilitate.'
        )
    else:
        episodes_data = create_episodes_table_data(episodes_new, ground_truth)
        df_eps = pd.DataFrame(episodes_data)

        st.dataframe(df_eps, use_container_width=True, hide_index=True)

        st.markdown('#### Distributia episoadelor')
        dist_cols = st.columns(3)

        durations = [ep['duration_sec'] for ep in episodes_new]
        dist_cols[0].metric(
            'Durata medie', f'{np.mean(durations):.1f}s',
            delta=f'±{np.std(durations):.1f}s'
        )
        dist_cols[1].metric(
            'Durata mediana', f'{np.median(durations):.1f}s'
        )

        # Spacing intre episoade
        if len(episodes_new) > 1:
            gaps = []
            for i in range(1, len(episodes_new)):
                gap = episodes_new[i]['start_sec'] - episodes_new[i-1]['end_sec']
                gaps.append(gap)
            dist_cols[2].metric(
                'Gap mediu intre episoade',
                f'{np.mean(gaps):.0f}s'
            )
        else:
            dist_cols[2].metric('Gap mediu intre episoade', '-')


# ---------- TAB 5: EXPORT ----------
with tab_export:
    st.markdown('#### Descarca rezultatele analizei')

    export_cols = st.columns(3)

    # CSV predictii complete
    df_predictions = pd.DataFrame({
        'start_sec': inf['starts_sec'],
        'end_sec': inf['starts_sec'] + WINDOW_SEC,
        'prob_lgbm': inf['probs_lgbm'],
        'prob_eegnet': (inf['probs_eegnet'] if inf['probs_eegnet'] is not None
                        else np.full(len(inf['starts_sec']), np.nan)),
        'prob_ensemble': inf['probs_ensemble'],
        'prob_smoothed': probs_smoothed_new,
        'alert': alerts_new,
    })
    csv_pred = df_predictions.to_csv(index=False).encode('utf-8')
    export_cols[0].download_button(
        label='📊 Predictii complete (CSV)',
        data=csv_pred,
        file_name=f'{Path(uploaded_file.name).stem}_predictions.csv',
        mime='text/csv',
        use_container_width=True,
    )

    # CSV episoade
    if episodes_new:
        df_eps_export = pd.DataFrame([{
            'episode_id': i + 1,
            'start_sec': ep['start_sec'],
            'end_sec': ep['end_sec'],
            'duration_sec': ep['duration_sec'],
            'max_probability': ep['max_prob'],
            'n_windows': ep['n_windows'],
        } for i, ep in enumerate(episodes_new)])

        csv_eps = df_eps_export.to_csv(index=False).encode('utf-8')
        export_cols[1].download_button(
            label='🔴 Episoade (CSV)',
            data=csv_eps,
            file_name=f'{Path(uploaded_file.name).stem}_episodes.csv',
            mime='text/csv',
            use_container_width=True,
        )
    else:
        export_cols[1].button(
            '🔴 Episoade (CSV)', disabled=True,
            help='Nu exista episoade de exportat.',
            use_container_width=True,
        )

    # PDF raport
    with st.spinner(''):
        try:
            pdf_bytes = generate_pdf_report(
                result, uploaded_file.name,
                ground_truth=ground_truth,
                window_metrics=win_metrics,
                event_metrics=evt_metrics,
            )
            export_cols[2].download_button(
                label='📄 Raport complet (PDF)',
                data=pdf_bytes,
                file_name=f'{Path(uploaded_file.name).stem}_report.pdf',
                mime='application/pdf',
                use_container_width=True,
                type='primary',
            )
        except Exception as e:
            export_cols[2].error(f'PDF: {e}')

    st.markdown('---')
    st.markdown("""
    **Continut raport PDF:**
    - Informatii fisier si configuratie inferenta
    - Metrici per-window si per-event (daca exista ground truth)
    - Tabel cu episoadele detectate
    - Interpretare automata a rezultatelor
    - Note metodologice si limitari
    """)


# ============================================================================
# FOOTER
# ============================================================================
st.markdown('---')
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px; padding: 20px;'>
    EpiDetect | Lucrare de licenta | Mierlă Constantin<br>
    Universitatea Babes-Bolyai, Facultatea de Matematica si Informatica | 2026
</div>
""", unsafe_allow_html=True)
