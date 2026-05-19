"""Modul de inferenta pentru detectia crizelor epileptice."""
from .pipeline import load_model, run_inference
from .preprocessing import N_CHANNELS, FS, WINDOW_SEC, COMMON_CHANNELS
from .models import EEGNet

__all__ = ['load_model', 'run_inference', 'N_CHANNELS', 'FS',
           'WINDOW_SEC', 'COMMON_CHANNELS', 'EEGNet']
