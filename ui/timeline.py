"""
Vizualizare timeline interactiv cu Plotly.
Dark mode, tech accents, responsive.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Paleta de culori (dark mode / tech)
COLOR_BG = '#0e1117'
COLOR_GRID = '#2a2f3e'
COLOR_TEXT = '#fafafa'
COLOR_ACCENT = '#00d9ff'        # cyan - probabilitati principale
COLOR_ALERT = '#ff3366'         # rosu neon - alerte
COLOR_THRESHOLD = '#ffaa00'     # galben - prag
COLOR_SECONDARY = '#9966ff'     # violet - model secundar
COLOR_GT = '#00ff99'            # verde neon - ground truth


def create_timeline_figure(results, show_components=True, ground_truth=None):
    """
    Creeaza figura Plotly cu timeline-ul probabilitatilor si episoadelor.

    Args:
        results: dict returnat de run_inference
        show_components: daca afiseaza curbele separate LGBM si EEGNet
        ground_truth: lista de tuples (start_sec, end_sec) cu crizele reale (optional)

    Returns:
        plotly.graph_objects.Figure
    """
    inf = results['inference']
    threshold = inf['threshold']
    starts_sec = inf['starts_sec']
    probs_smoothed = inf['probs_smoothed']
    alerts = inf['alerts']
    probs_lgbm = inf['probs_lgbm']
    probs_eegnet = inf['probs_eegnet']

    # Convertim in minute pentru lizibilitate
    starts_min = starts_sec / 60.0

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.75, 0.25],
        subplot_titles=('Probabilitate criza', 'Alerte detectate')
    )

    # Panoul 1 - probabilitati

    # Ground truth (crize reale) ca band de fundal - prima, sa fie in spate
    if ground_truth:
        for i, (gt_start, gt_end) in enumerate(ground_truth):
            fig.add_vrect(
                x0=gt_start / 60.0, x1=gt_end / 60.0,
                fillcolor=COLOR_GT, opacity=0.15,
                line_width=0,
                row=1, col=1,
                annotation_text='Criza reala' if i == 0 else None,
                annotation_position='top left',
                annotation=dict(font=dict(color=COLOR_GT, size=10))
            )

    # Componente individuale (LGBM, EEGNet)
    if show_components and probs_lgbm is not None:
        fig.add_trace(
            go.Scatter(
                x=starts_min, y=probs_lgbm,
                name='LightGBM',
                mode='lines',
                line=dict(color=COLOR_SECONDARY, width=1),
                opacity=0.4,
                hovertemplate='<b>LightGBM</b><br>Timp: %{x:.2f} min<br>Prob: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )

    if show_components and probs_eegnet is not None:
        fig.add_trace(
            go.Scatter(
                x=starts_min, y=probs_eegnet,
                name='EEGNet',
                mode='lines',
                line=dict(color='#ff9966', width=1),
                opacity=0.4,
                hovertemplate='<b>EEGNet</b><br>Timp: %{x:.2f} min<br>Prob: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )

    # Ensemble smoothed - curba principala
    fig.add_trace(
        go.Scatter(
            x=starts_min, y=probs_smoothed,
            name='Ensemble (smoothed)',
            mode='lines',
            line=dict(color=COLOR_ACCENT, width=2.5),
            fill='tozeroy',
            fillcolor='rgba(0, 217, 255, 0.1)',
            hovertemplate='<b>Ensemble</b><br>Timp: %{x:.2f} min<br>Prob: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Linia de threshold
    fig.add_hline(
        y=threshold, line=dict(color=COLOR_THRESHOLD, width=1.5, dash='dash'),
        annotation_text=f'Prag = {threshold:.2f}',
        annotation_position='right',
        annotation=dict(font=dict(color=COLOR_THRESHOLD, size=11)),
        row=1, col=1
    )

    # Marcarea episoadelor detectate ca benzi rosii
    for ep in results['episodes']:
        fig.add_vrect(
            x0=ep['start_sec'] / 60.0, x1=ep['end_sec'] / 60.0,
            fillcolor=COLOR_ALERT, opacity=0.25,
            line=dict(color=COLOR_ALERT, width=1),
            row=1, col=1
        )

    # Panoul 2 - alerte binare
    fig.add_trace(
        go.Scatter(
            x=starts_min, y=alerts,
            mode='lines',
            line=dict(color=COLOR_ALERT, width=0, shape='hv'),
            fill='tozeroy',
            fillcolor=COLOR_ALERT,
            name='Alerte',
            showlegend=False,
            hovertemplate='Timp: %{x:.2f} min<br>Alerta: %{y}<extra></extra>'
        ),
        row=2, col=1
    )

    # Layout global
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        height=500,
        margin=dict(l=50, r=50, t=60, b=50),
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom', y=1.08,
            xanchor='right', x=1,
            bgcolor='rgba(26, 31, 46, 0.8)',
            bordercolor=COLOR_GRID,
            borderwidth=1,
            font=dict(color=COLOR_TEXT)
        ),
        font=dict(color=COLOR_TEXT, family='sans-serif'),
    )

    fig.update_xaxes(
        title_text='Timp (minute)',
        gridcolor=COLOR_GRID,
        zerolinecolor=COLOR_GRID,
        row=2, col=1
    )
    fig.update_yaxes(
        title_text='Probabilitate',
        range=[0, 1.05],
        gridcolor=COLOR_GRID,
        zerolinecolor=COLOR_GRID,
        row=1, col=1
    )
    fig.update_yaxes(
        title_text='Alerta',
        range=[-0.1, 1.1],
        tickvals=[0, 1],
        gridcolor=COLOR_GRID,
        zerolinecolor=COLOR_GRID,
        row=2, col=1
    )

    return fig


def create_episodes_table_data(episodes, ground_truth=None):
    """
    Construieste datele pentru tabelul de episoade.
    Daca exista ground truth, calculeaza si overlap cu crizele reale.
    """
    rows = []
    for i, ep in enumerate(episodes, 1):
        row = {
            '#': i,
            'Start': format_time(ep['start_sec']),
            'Sfarsit': format_time(ep['end_sec']),
            'Durata': f"{ep['duration_sec']:.0f}s",
            'Prob. max': f"{ep['max_prob']:.3f}",
            'Nr. ferestre': ep['n_windows'],
        }
        # Daca avem ground truth, verificam daca episodul se suprapune cu o criza reala
        if ground_truth:
            overlap = False
            latency = None
            for gt_start, gt_end in ground_truth:
                # Overlap daca intervalele se intersecteaza
                if ep['start_sec'] < gt_end and ep['end_sec'] > gt_start:
                    overlap = True
                    # Latenta: cu cat a fost detectat inainte de debutul crizei
                    latency = gt_start - ep['start_sec']
                    break
            row['Confirmat'] = '✓ TP' if overlap else '✗ FP'
            if latency is not None:
                if latency > 0:
                    row['Latenta'] = f'+{latency:.0f}s inainte'
                else:
                    row['Latenta'] = f'{-latency:.0f}s dupa debut'
            else:
                row['Latenta'] = '-'
        rows.append(row)
    return rows


def format_time(seconds):
    """Formateaza secunde in MM:SS sau HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f'{h:02d}:{m:02d}:{s:02d}'
    return f'{m:02d}:{s:02d}'
