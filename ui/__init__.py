"""Modul UI - componente vizuale cu Plotly."""
from .timeline import create_timeline_figure, create_episodes_table_data, format_time
from .eeg_viewer import create_eeg_viewer, find_episode_for_time
from .shap_viz import (compute_shap_for_window, create_shap_plot,
                        create_shap_summary_by_category, prettify_feature_name)

__all__ = [
    'create_timeline_figure', 'create_episodes_table_data', 'format_time',
    'create_eeg_viewer', 'find_episode_for_time',
    'compute_shap_for_window', 'create_shap_plot',
    'create_shap_summary_by_category', 'prettify_feature_name',
]
