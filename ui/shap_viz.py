"""
Vizualizare SHAP pentru explicabilitate per-fereastra.
Arata ce features au contribuit la predictia modelului LightGBM pentru
o fereastra specifica.
"""
import numpy as np
import plotly.graph_objects as go
import shap


COLOR_BG = '#0e1117'
COLOR_GRID = '#2a2f3e'
COLOR_TEXT = '#fafafa'
COLOR_POS = '#ff3366'    # contributia spre "periculos"
COLOR_NEG = '#00d9ff'    # contributia spre "normal"


_shap_explainer_cache = None


def get_shap_explainer(lgbm_model):
    """Creeaza sau returneaza explainer-ul cached."""
    global _shap_explainer_cache
    if _shap_explainer_cache is None:
        _shap_explainer_cache = shap.TreeExplainer(lgbm_model)
    return _shap_explainer_cache


def compute_shap_for_window(features_scaled_selected, lgbm_model):
    """
    Calculeaza valorile SHAP pentru o singura fereastra.

    Args:
        features_scaled_selected: vector de features scaled + selected (shape: n_features)
        lgbm_model: modelul LightGBM

    Returns:
        (shap_values, base_value) - contributia fiecarui feature + valoarea de referinta
    """
    explainer = get_shap_explainer(lgbm_model)
    # SHAP asteapta batch, deci reshape la (1, n_features)
    X = features_scaled_selected.reshape(1, -1)
    shap_values = explainer.shap_values(X)

    # Pentru LightGBM binary, shap_values poate fi lista [class_0, class_1] sau array direct
    if isinstance(shap_values, list):
        shap_vals_pos = shap_values[1][0]  # pentru clasa pozitiva
        base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    else:
        shap_vals_pos = shap_values[0]
        base_val = explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[0]

    return shap_vals_pos, float(base_val)


def prettify_feature_name(name):
    """Transforma un nume tehnic de feature intr-unul lizibil."""
    # ch05_theta_alpha_ratio -> Ch5 Theta/Alpha ratio
    parts = name.split('_')
    result = []
    for p in parts:
        if p.startswith('ch') and p[2:].isdigit():
            result.append(f'Ch{int(p[2:])}')
        elif p in ['wt', 'ptp', 'rms']:
            result.append(p.upper())
        elif p in ['rel', 'cross', 'corr']:
            result.append(p)
        else:
            result.append(p.capitalize())
    # Cross-correlation features: cross_corr_05_12 -> Corr Ch5-Ch12
    if name.startswith('cross_corr_'):
        tokens = name.replace('cross_corr_', '').split('_')
        if len(tokens) == 2:
            return f'Corr Ch{int(tokens[0])}-Ch{int(tokens[1])}'
    return ' '.join(result)


def create_shap_plot(shap_values, feature_names, base_value, top_n=15):
    """
    Creeaza bar plot orizontal cu top N features ordonate dupa contributia absoluta.

    Args:
        shap_values: array cu valorile SHAP per feature
        feature_names: lista cu numele features (aceeasi lungime)
        base_value: valoarea de baseline a modelului
        top_n: cate features sa afiseze

    Returns:
        (plotly figure, prediction_value)
    """
    # Sortam dupa contributia absoluta descrescator
    abs_shap = np.abs(shap_values)
    top_idx = np.argsort(abs_shap)[::-1][:top_n]

    # Pentru afisare, ordonam crescator ca barele cele mai mari sa fie sus
    top_idx_display = top_idx[::-1]

    values = shap_values[top_idx_display]
    names = [prettify_feature_name(feature_names[i]) for i in top_idx_display]
    colors = [COLOR_POS if v > 0 else COLOR_NEG for v in values]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=values,
        y=names,
        orientation='h',
        marker=dict(color=colors, line=dict(color='#333', width=0.5)),
        hovertemplate='<b>%{y}</b><br>SHAP: %{x:+.4f}<extra></extra>',
        text=[f'{v:+.3f}' for v in values],
        textposition='outside',
        textfont=dict(color=COLOR_TEXT, size=10),
    ))

    # Linie verticala la 0
    fig.add_vline(x=0, line=dict(color='#666', width=1))

    # Predictia finala = base + suma contributiilor
    final_logit = base_value + np.sum(shap_values)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        height=max(400, 30 * top_n),
        margin=dict(l=180, r=60, t=60, b=50),
        title=dict(
            text=f'<b>Top {top_n} features</b> | Base: {base_value:+.3f} | Final: {final_logit:+.3f}',
            font=dict(color=COLOR_TEXT, size=14),
            x=0.02,
        ),
        xaxis=dict(
            title='Contributie SHAP (spre criza ->)',
            gridcolor=COLOR_GRID,
            zerolinecolor=COLOR_GRID,
        ),
        yaxis=dict(
            gridcolor=COLOR_GRID,
        ),
        showlegend=False,
        font=dict(color=COLOR_TEXT, family='sans-serif'),
    )

    return fig, final_logit


def create_shap_summary_by_category(shap_values, feature_names):
    """
    Agregheaza contributiile SHAP pe categorii de features.
    Utila pentru o privire de ansamblu: ce tip de features conteaza mai mult?
    """
    categories = {
        'Band Power (spectrale)': ['delta', 'theta', 'alpha', 'beta', 'gamma',
                                    'spectral_edge'],
        'Wavelet (time-frequency)': ['wt_'],
        'Hjorth (complexitate)': ['hjorth'],
        'Statistice (amplitudine)': ['std', 'variance', 'kurtosis', 'skewness',
                                      'rms', 'ptp', 'mean_', 'energy', 'zero_cross'],
        'Entropii (non-liniare)': ['entropy', 'hurst'],
        'Cross-channel': ['cross_corr'],
    }

    cat_contributions = {}
    for cat, keywords in categories.items():
        total = 0.0
        for i, fname in enumerate(feature_names):
            if any(kw in fname for kw in keywords):
                total += abs(shap_values[i])
        cat_contributions[cat] = total

    # Sortare descrescatoare
    sorted_cats = sorted(cat_contributions.items(), key=lambda x: x[1], reverse=True)

    cats = [c[0] for c in sorted_cats]
    vals = [c[1] for c in sorted_cats]

    colors = ['#ff3366', '#ff9966', '#ffaa00', '#00d9ff', '#9966ff', '#00ff99']

    fig = go.Figure(go.Pie(
        labels=cats,
        values=vals,
        hole=0.5,
        marker=dict(colors=colors[:len(cats)], line=dict(color=COLOR_BG, width=2)),
        textposition='inside',
        textinfo='percent',
        hovertemplate='<b>%{label}</b><br>Contributie: %{value:.3f} (%{percent})<extra></extra>',
    ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        title=dict(
            text='<b>Contributii pe categorii</b>',
            font=dict(color=COLOR_TEXT, size=13),
        ),
        legend=dict(
            orientation='v',
            yanchor='middle', y=0.5,
            xanchor='left', x=1.02,
            font=dict(color=COLOR_TEXT, size=10),
        ),
        font=dict(color=COLOR_TEXT, family='sans-serif'),
    )

    return fig
