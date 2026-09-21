"""Checkpoint and stopping helpers for TransCell training."""

import os

import numpy as np
import torch


class TranslationEarlyStopping:
    """Early stopping with best-loss bookkeeping for translation training."""

    def __init__(self, patience=30, verbose=False, checkpoint_file=""):
        self.patience = patience
        self.verbose = verbose
        self.checkpoint_file = checkpoint_file
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.inf
        self.best_metrics = {}

    def __call__(self, loss, model, metrics=None):
        if np.isnan(loss):
            self.early_stop = True
            return
        score = -loss
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
            self.best_metrics = dict(metrics or {})
            self.save_checkpoint(loss, model)
            return
        self.counter += 1
        if self.verbose:
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
        if self.counter >= self.patience:
            self.early_stop = True
            if hasattr(model, "load_model"):
                model.load_model(self.checkpoint_file)

    def save_checkpoint(self, loss, model):
        checkpoint_dir = os.path.dirname(self.checkpoint_file)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), self.checkpoint_file)
        self.loss_min = loss

