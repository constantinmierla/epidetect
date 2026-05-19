"""
Vizualizare semnal EEG brut pentru o fereastra / interval specificat.
Afiseaza toate cele 18 canale stacked, cu highlight pe fereastra de interes.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from inference.preprocessing import COMMON_CHANNELS, FS


COLOR_BG = '#0e1117'
COLOR_GRID = '#2a2f3e'
COLOR_TEXT = '#fafafa'
COLOR_SIGNAL = '#00d9ff'
COLOR_HIGHLIGHT = '#ff3366'
COLOR_GT = '#00ff99'


def create_eeg_viewer(raw_data, center_sec, context_sec=30,
                       highlight_start_sec=None, highlight_end_sec=None,
                       ground_truth_intervals=None, fs=FS):
    """
    Afiseaza semnalul EEG brut pentru toate cele 18 canale.

    Args:
        raw_data: array (n_channels, n_samples) - semnal preprocesat
        center_sec: timpul central (in secunde)
        context_sec: cate secunde de context in jurul lui center_sec
        highlight_start_sec, highlight_end_sec: interval de highlight (episod detectat)
        ground_truth_intervals: lista de (start, end) crize reale
        fs: frecventa de esantionare

    Returns:
        plotly.graph_objects.Figure
    """
    n_channels = raw_data.shape[0]
    n_samples = raw_data.shape[1]
    duration_sec = n_samples / fs

    # Calculam fereastra de afisare
    half_window = context_sec / 2
    display_start_sec = max(0, center_sec - half_window)
    display_end_sec = min(duration_sec, center_sec + half_window)

    start_idx = int(display_start_sec * fs)
    end_idx = int(display_end_sec * fs)

    # Extragem doar portiunea de interes
    signal_slice = raw_data[:, start_idx:end_idx]
    time_axis = np.arange(start_idx, end_idx) / fs

    # Decimate daca sunt prea multe puncte (pentru performanta browser)
    if signal_slice.shape[1] > 10000:
        factor = signal_slice.shape[1] // 10000 + 1
        signal_slice = signal_slice[:, ::factor]
        time_axis = time_axis[::factor]

    # Construim figura cu un subplot per canal (stacked)
    fig = go.Figure()

    # Calculam offset-uri pentru stacking
    # Fiecare canal e deplasat vertical ca sa nu se suprapuna
    signal_range = 6.0  # +/-3 sigma dupa normalizare
    offsets = np.arange(n_channels) * signal_range

    # Adaugam fiecare canal ca linie separata
    for ch_idx in range(n_channels):
        ch_name = COMMON_CHANNELS[ch_idx] if ch_idx < len(COMMON_CHANNELS) else f'Ch{ch_idx}'
        y_shifted = signal_slice[ch_idx] + offsets[ch_idx]

        fig.add_trace(go.Scatter(
            x=time_axis,
            y=y_shifted,
            mode='lines',
            name=ch_name,
            line=dict(color=COLOR_SIGNAL, width=0.8),
            hovertemplate=f'<b>{ch_name}</b><br>Timp: %{{x:.2f}}s<br>Amplitudine: %{{customdata:.2f}}<extra></extra>',
            customdata=signal_slice[ch_idx],
            showlegend=False,
        ))

    # Highlight pentru intervalul episod detectat
    if highlight_start_sec is not None and highlight_end_sec is not None:
        fig.add_vrect(
            x0=highlight_start_sec, x1=highlight_end_sec,
            fillcolor=COLOR_HIGHLIGHT, opacity=0.15,
            line=dict(color=COLOR_HIGHLIGHT, width=1, dash='dot'),
            annotation_text='Episod detectat',
            annotation_position='top left',
            annotation=dict(font=dict(color=COLOR_HIGHLIGHT, size=10)),
        )

    # Ground truth overlay
    if ground_truth_intervals:
        for i, (gt_start, gt_end) in enumerate(ground_truth_intervals):
            # Afisam doar crizele care se suprapun cu fereastra vizibila
            if gt_end >= display_start_sec and gt_start <= display_end_sec:
                fig.add_vrect(
                    x0=max(gt_start, display_start_sec),
                    x1=min(gt_end, display_end_sec),
                    fillcolor=COLOR_GT, opacity=0.1,
                    line=dict(color=COLOR_GT, width=1),
                    annotation_text='Criza reala' if i == 0 else None,
                    annotation_position='top right',
                    annotation=dict(font=dict(color=COLOR_GT, size=10)),
                )

    # Linia centrala
    fig.add_vline(
        x=center_sec, line=dict(color='#ffaa00', width=1, dash='dash'),
        annotation_text=f't = {center_sec:.1f}s',
        annotation_position='top',
        annotation=dict(font=dict(color='#ffaa00', size=10)),
    )

    # Layout
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        height=600,
        margin=dict(l=80, r=30, t=40, b=50),
        hovermode='closest',
        font=dict(color=COLOR_TEXT, family='sans-serif', size=11),
        xaxis=dict(
            title='Timp (secunde)',
            gridcolor=COLOR_GRID,
            zerolinecolor=COLOR_GRID,
        ),
        yaxis=dict(
            title='Canale',
            tickmode='array',
            tickvals=offsets,
            ticktext=COMMON_CHANNELS[:n_channels],
            gridcolor=COLOR_GRID,
            zerolinecolor=COLOR_GRID,
        ),
    )

    return fig


def find_episode_for_time(episodes, time_sec):
    """Gaseste episodul care contine un timp specific, daca exista."""
    for ep in episodes:
        if ep['start_sec'] <= time_sec <= ep['end_sec']:
            return ep
    return None
