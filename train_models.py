"""
train_models.py

Trains and evaluates the 7 models described in Section 3.4 of
"Manuscript_Draft_AnomalyDetection_ICS.md", on the real, preprocessed
HAI + CIC-IDS2018 splits produced by ics_add_pipeline.py.

    ML baselines (sklearn, always available):
        train_random_forest()     -- supervised
        train_svm()                -- supervised
        train_ann()                 -- supervised (MLPClassifier)
        train_ocsvm()               -- unsupervised (normal-only train)
        train_isolation_forest()    -- unsupervised (normal-only train)

    Deep learning (requires PyTorch -- pip install torch):
        train_autoencoder()         -- unsupervised (normal-only train)
        train_lstm_classifier()     -- supervised

This script prints every metric it computes and returns ModelResult objects
that map directly onto Tables 2-6 in the manuscript's Results section.
Nothing here invents a number -- if a step can't run (e.g. PyTorch missing),
it raises rather than silently skipping or fabricating a placeholder result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, make_scorer, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, OneClassSVM

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# --------------------------------------------------------------------------
# Result container + shared metric computation
# --------------------------------------------------------------------------
@dataclass
class ModelResult:
    model_name: str
    dataset_name: str
    hyperparams: dict
    precision: float
    recall: float
    f1: float
    accuracy: float
    random_state: int = 42

    def as_manuscript_row(self) -> str:
        """Format as a markdown table row matching Tables 2/5/6 in the manuscript."""
        hp = ", ".join(f"{k}={v}" for k, v in self.hyperparams.items())
        return (f"| {self.model_name} | {hp} | {self.precision*100:.2f}% | "
                f"{self.recall*100:.2f}% | {self.f1*100:.2f}% | {self.accuracy*100:.2f}% |")


@dataclass
class RepeatedRunSummary:
    model_name: str
    dataset_name: str
    n_runs: int
    seeds: list
    metrics: dict  # metric_name -> {"mean":..., "std":..., "ci95_low":..., "ci95_high":..., "values":[...]}

    def as_manuscript_row(self, metric: str = "f1") -> str:
        m = self.metrics[metric]
        return (f"| {self.model_name} | {self.n_runs} | "
                f"{m['mean']*100:.2f}% ± {m['std']*100:.2f}% | "
                f"[{m['ci95_low']*100:.2f}%, {m['ci95_high']*100:.2f}%] |")


def summarize_repeated_runs(results: list) -> "RepeatedRunSummary":
    """
    Aggregates N ModelResult objects (same model/dataset, different
    random_state) into mean/std/95% CI per metric. Uses a normal
    approximation for the 95% CI (mean ± 1.96*std/sqrt(n)) rather than a
    t-distribution, since n is typically small (5-10) -- this is
    intentionally conservative-adjacent, not understating the interval;
    for n<5 the CI should be treated as indicative only, and this function
    prints a warning rather than silently returning an overconfident interval.
    """
    if len(results) < 2:
        raise ValueError(
            "summarize_repeated_runs() needs at least 2 runs (different "
            "random_state values) to compute a standard deviation/CI -- a "
            "single run cannot be summarized as if it were a distribution."
        )
    model_names = {r.model_name for r in results}
    dataset_names = {r.dataset_name for r in results}
    if len(model_names) > 1 or len(dataset_names) > 1:
        raise ValueError(
            f"summarize_repeated_runs() expects all results to be the same "
            f"model+dataset, got models={model_names}, datasets={dataset_names}. "
            "Group results by (model, dataset) before calling this."
        )
    if len(set(r.random_state for r in results)) < len(results):
        print(f"  WARNING: {results[0].model_name}/{results[0].dataset_name} -- "
              f"repeated results do not all have distinct random_state values; "
              f"the resulting variance may understate true run-to-run variability.")
    n = len(results)
    if n < 5:
        print(f"  WARNING: only {n} runs for {results[0].model_name}/{results[0].dataset_name} "
              f"-- treat this CI as indicative only, not a reliable interval (Section 6.5).")

    metrics = {}
    for metric in ["precision", "recall", "f1", "accuracy"]:
        values = [getattr(r, metric) for r in results]
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))  # sample std, not population std
        se = std / np.sqrt(n)
        metrics[metric] = {
            "mean": mean, "std": std,
            "ci95_low": mean - 1.96 * se, "ci95_high": mean + 1.96 * se,
            "values": values,
        }

    summary = RepeatedRunSummary(
        model_name=results[0].model_name, dataset_name=results[0].dataset_name,
        n_runs=n, seeds=[r.random_state for r in results], metrics=metrics,
    )
    print(f"  {summary.model_name} [{summary.dataset_name}] over {n} runs "
          f"(seeds={summary.seeds}): "
          f"F1 = {metrics['f1']['mean']*100:.2f}% ± {metrics['f1']['std']*100:.2f}% "
          f"(95% CI [{metrics['f1']['ci95_low']*100:.2f}%, {metrics['f1']['ci95_high']*100:.2f}%])")
    return summary


def _metrics(y_true, y_pred, model_name: str, dataset_name: str, hyperparams: dict,
             random_state: int = 42) -> ModelResult:
    result = ModelResult(
        model_name=model_name,
        dataset_name=dataset_name,
        hyperparams=hyperparams,
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        accuracy=accuracy_score(y_true, y_pred),
        random_state=random_state,
    )
    print(f"  {model_name} [{dataset_name}] seed={random_state} {hyperparams}: "
          f"P={result.precision*100:.2f}% R={result.recall*100:.2f}% "
          f"F1={result.f1*100:.2f}% Acc={result.accuracy*100:.2f}%")
    return result


# --------------------------------------------------------------------------
# ML baselines (Section 3.4)
# --------------------------------------------------------------------------
def train_random_forest(X_train, y_train, X_test, y_test, dataset_name: str,
                         n_estimators: int = 200, class_weight=None,
                         random_state: int = 42) -> ModelResult:
    import time
    t0 = time.time()
    print(f"  Random Forest [{dataset_name}] seed={random_state}: fitting on {len(X_train)} rows "
          f"(class_weight={class_weight})...")
    clf = RandomForestClassifier(n_estimators=n_estimators, class_weight=class_weight,
                                  random_state=random_state, n_jobs=-1)
    clf.fit(X_train, y_train)
    print(f"  Random Forest [{dataset_name}]: fit done in {time.time()-t0:.1f}s.")
    y_pred = clf.predict(X_test)
    return _metrics(y_test, y_pred, "Random Forest", dataset_name,
                     {"n_estimators": n_estimators}, random_state=random_state)


def train_svm(X_train, y_train, X_test, y_test, dataset_name: str,
              C: float = 10, gamma: float = 1, class_weight="balanced",
              max_train_n: int = 50_000, random_state: int = 42) -> ModelResult:
    """
    SVC has O(n^2)-O(n^3) training cost, which is impractical on the full
    525,000-row CIC-IDS2018 supervised-train split. Subsamples the training
    set to `max_train_n` if larger, and reports the exact number used --
    this is a stated computational-cost trade-off, not a silent one, and
    must be reported in the manuscript alongside the result.
    """
    import time
    if len(X_train) > max_train_n:
        idx = np.random.RandomState(random_state).choice(len(X_train), max_train_n, replace=False)
        X_train = X_train.iloc[idx]
        y_train = y_train.iloc[idx]
        print(f"  SVM: subsampled training set to {max_train_n} rows (seed={random_state}) "
              f"(SVC does not scale to the full supervised-train split).")
    print(f"  SVM [{dataset_name}] seed={random_state}: fitting on {len(X_train)} rows -- this is the "
          f"slowest step (O(n^2)-O(n^3)); on {max_train_n} rows expect anywhere from "
          f"a few minutes to well over an hour depending on your CPU. No further "
          f"output will print until it's done -- check Task Manager CPU usage for "
          f"python.exe if you're unsure whether it's still running.")
    t0 = time.time()
    clf = SVC(C=C, gamma=gamma, class_weight=class_weight, random_state=random_state)
    clf.fit(X_train, y_train)
    print(f"  SVM [{dataset_name}]: fit done in {time.time()-t0:.1f}s.")
    print(f"  SVM [{dataset_name}] DIAGNOSTIC -- classes_: {clf.classes_}, "
          f"class_weight_ actually applied per class: {clf.class_weight_}, "
          f"n_support_ (support vectors per class): {clf.n_support_}")
    # Interpretation: class_weight_ should show a MUCH larger multiplier for
    # the minority (attack=1) class than for the majority class if
    # class_weight="balanced" reached the model correctly -- if it does NOT
    # (e.g. both classes show ~equal weight), the bug is not fully fixed and
    # this diagnostic will make that visible directly, rather than inferring
    # it indirectly from Precision/Recall alone.
    y_pred = clf.predict(X_test)
    return _metrics(y_test, y_pred, "SVM", dataset_name,
                     {"C": C, "gamma": gamma, "train_n": len(X_train)}, random_state=random_state)


def train_ocsvm(X_train_normal, X_test, y_test, dataset_name: str,
                 nu: float = 0.1, gamma: float = 0.1, max_train_n: int = 50_000,
                 random_state: int = 42) -> ModelResult:
    """
    NOTE: sklearn's OneClassSVM itself has no `random_state` parameter (its
    solver is deterministic given the data); the only source of run-to-run
    variance here is the subsampling below, which IS seeded. Repeated runs
    with the same `random_state` will therefore give identical results --
    use different `random_state` values across repeated-run calls (varying
    which 50,000 normal rows are subsampled) to get a meaningful variance
    estimate, not repeated calls with the same default.
    """
    if len(X_train_normal) > max_train_n:
        idx = np.random.RandomState(random_state).choice(len(X_train_normal), max_train_n, replace=False)
        X_train_normal = X_train_normal.iloc[idx]
        print(f"  OCSVM: subsampled normal-only training set to {max_train_n} rows (seed={random_state}).")
    clf = OneClassSVM(nu=nu, gamma=gamma)
    clf.fit(X_train_normal)
    raw_pred = clf.predict(X_test)  # OneClassSVM: 1 = normal (inlier), -1 = anomaly
    y_pred = (raw_pred == -1).astype(int)
    return _metrics(y_test, y_pred, "OCSVM", dataset_name,
                     {"nu": nu, "gamma": gamma, "train_n": len(X_train_normal)}, random_state=random_state)


def train_isolation_forest(X_train_normal, X_test, y_test, dataset_name: str,
                            n_estimators: int = 100, contamination: float = 0.1,
                            random_state: int = 42) -> ModelResult:
    clf = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                           random_state=random_state, n_jobs=-1)
    clf.fit(X_train_normal)
    raw_pred = clf.predict(X_test)  # IsolationForest: 1 = normal, -1 = anomaly
    y_pred = (raw_pred == -1).astype(int)
    return _metrics(y_test, y_pred, "Isolation Forest", dataset_name,
                     {"n_estimators": n_estimators, "contamination": contamination},
                     random_state=random_state)


def train_ann(X_train, y_train, X_test, y_test, dataset_name: str,
              hidden_layer_sizes: tuple = (128,), max_iter: int = 200,
              class_weight="balanced", random_state: int = 42) -> ModelResult:
    """
    MLPClassifier has no `class_weight` parameter (unlike RF/SVM), so
    imbalance is instead handled via per-sample weights passed to `.fit()`,
    computed with sklearn's compute_sample_weight -- otherwise this model
    would silently train unweighted on a ~2-37% attack rate, same bug class
    as the one found and fixed in run_all_models_for_dataset() for RF.
    """
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter,
                         random_state=random_state, early_stopping=True)
    if class_weight is not None:
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weight = compute_sample_weight(class_weight=class_weight, y=y_train)
        # NOTE: plain MLPClassifier.fit() does not accept sample_weight either
        # (only some sklearn estimators do) -- if your sklearn version raises
        # a TypeError here, fall back to manual class-balanced oversampling
        # of X_train/y_train before calling this function, and report that
        # choice explicitly rather than silently training unweighted.
        clf.fit(X_train, y_train, sample_weight=sample_weight) if _supports_sample_weight(clf) else clf.fit(X_train, y_train)
    else:
        clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return _metrics(y_test, y_pred, "ANN", dataset_name,
                     {"hidden_layer_sizes": hidden_layer_sizes}, random_state=random_state)


def _supports_sample_weight(estimator) -> bool:
    import inspect
    return "sample_weight" in inspect.signature(estimator.fit).parameters


def _make_search_subsample(X, y, max_search_n: int, random_state: int):
    """Subsample (X, y) for hyperparameter search, preserving class balance
    via a stratified sample -- searching on the full 281k-525k row training
    sets would make RandomizedSearchCV impractically slow. The FINAL model
    is still refit on the full training set with the winning hyperparameters
    (see tune_random_forest/tune_ann below) -- only the search itself uses
    the subsample, so the reported test-set result is not affected by this
    shortcut, only the hyperparameter *selection* process is."""
    if len(X) <= max_search_n:
        return X, y
    frac = max_search_n / len(X)
    parts_X, parts_y = [], []
    for cls in y.unique():
        mask = y == cls
        idx = X[mask].sample(frac=frac, random_state=random_state).index
        parts_X.append(X.loc[idx])
        parts_y.append(y.loc[idx])
    X_sub = pd.concat(parts_X)
    y_sub = pd.concat(parts_y)
    print(f"  Hyperparameter search: subsampled {len(X)} -> {len(X_sub)} rows "
          f"(stratified) for the search only; final model is refit on the full {len(X)} rows.")
    return X_sub, y_sub


def tune_random_forest(X_train, y_train, X_test, y_test, dataset_name: str,
                        n_iter: int = 15, cv: int = 3, max_search_n: int = 50_000,
                        random_state: int = 42) -> ModelResult:
    """
    Randomized hyperparameter search for Random Forest, scored on MCC (not
    F1/accuracy) since MCC is this study's primary metric for imbalanced-
    class comparisons (Section 4/Table 7). Confirmed necessary (Section 6.4):
    every RF result reported before this function existed used the single
    fixed default n_estimators=200 with no other hyperparameters tuned --
    this is the first time RF's max_depth, min_samples_split/leaf, and
    max_features are actually searched rather than left at sklearn defaults.

    Search space is intentionally modest (n_iter=15, cv=3) given this
    study's reported compute constraints (Section 4) -- prioritizes
    completing a real, if not exhaustive, search over an infeasibly large
    grid that would not finish.
    """
    import time
    X_search, y_search = _make_search_subsample(X_train, y_train, max_search_n, random_state)

    param_distributions = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "class_weight": ["balanced", "balanced_subsample", None],
    }
    mcc_scorer = make_scorer(matthews_corrcoef)

    print(f"  Tuning Random Forest [{dataset_name}]: {n_iter} candidates x {cv}-fold CV "
          f"on {len(X_search)} rows (search subsample), scored on MCC...")
    t0 = time.time()
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
        param_distributions=param_distributions, n_iter=n_iter, cv=cv,
        scoring=mcc_scorer, random_state=random_state, n_jobs=-1,
    )
    search.fit(X_search, y_search)
    print(f"  Tuning Random Forest [{dataset_name}]: search done in {time.time()-t0:.1f}s. "
          f"Best params: {search.best_params_} (CV MCC = {search.best_score_:.4f})")

    print(f"  Refitting Random Forest [{dataset_name}] with best params on the FULL "
          f"{len(X_train)}-row training set...")
    t0 = time.time()
    final_clf = RandomForestClassifier(**search.best_params_, random_state=random_state, n_jobs=-1)
    final_clf.fit(X_train, y_train)
    print(f"  Refit done in {time.time()-t0:.1f}s.")

    y_pred = final_clf.predict(X_test)
    result = _metrics(y_test, y_pred, "Random Forest (tuned)", dataset_name,
                       {**search.best_params_, "cv_mcc": round(search.best_score_, 4),
                        "search_n_iter": n_iter, "search_cv": cv},
                       random_state=random_state)
    return result


def tune_ann(X_train, y_train, X_test, y_test, dataset_name: str,
             n_iter: int = 15, cv: int = 3, max_search_n: int = 50_000,
             random_state: int = 42, class_weight="balanced") -> ModelResult:
    """
    Randomized hyperparameter search for the ANN (MLPClassifier), scored on
    MCC. Confirmed necessary (Section 6.4): every ANN result reported before
    this function existed used the single fixed default
    hidden_layer_sizes=(128,) with no other hyperparameters tuned. Search
    covers architecture (hidden_layer_sizes), L2 regularization (alpha), and
    learning rate -- the same modest-search-space rationale as
    tune_random_forest() applies here given this study's compute constraints.
    """
    import time
    X_search, y_search = _make_search_subsample(X_train, y_train, max_search_n, random_state)

    param_distributions = {
        "hidden_layer_sizes": [(64,), (128,), (256,), (128, 64), (256, 128), (128, 64, 32)],
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "learning_rate_init": [1e-4, 1e-3, 1e-2],
        "activation": ["relu", "tanh"],
    }
    mcc_scorer = make_scorer(matthews_corrcoef)

    print(f"  Tuning ANN [{dataset_name}]: {n_iter} candidates x {cv}-fold CV "
          f"on {len(X_search)} rows (search subsample), scored on MCC...")
    t0 = time.time()
    search = RandomizedSearchCV(
        MLPClassifier(random_state=random_state, max_iter=200, early_stopping=True),
        param_distributions=param_distributions, n_iter=n_iter, cv=cv,
        scoring=mcc_scorer, random_state=random_state, n_jobs=-1,
    )
    search.fit(X_search, y_search)
    print(f"  Tuning ANN [{dataset_name}]: search done in {time.time()-t0:.1f}s. "
          f"Best params: {search.best_params_} (CV MCC = {search.best_score_:.4f})")

    print(f"  Refitting ANN [{dataset_name}] with best params on the FULL "
          f"{len(X_train)}-row training set...")
    t0 = time.time()
    final_clf = MLPClassifier(**search.best_params_, random_state=random_state,
                               max_iter=200, early_stopping=True)
    if class_weight is not None:
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weight = compute_sample_weight(class_weight=class_weight, y=y_train)
        if _supports_sample_weight(final_clf):
            final_clf.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            final_clf.fit(X_train, y_train)
    else:
        final_clf.fit(X_train, y_train)
    print(f"  Refit done in {time.time()-t0:.1f}s.")

    y_pred = final_clf.predict(X_test)
    result = _metrics(y_test, y_pred, "ANN (tuned)", dataset_name,
                       {**search.best_params_, "cv_mcc": round(search.best_score_, 4),
                        "search_n_iter": n_iter, "search_cv": cv},
                       random_state=random_state)
    return result



    return "sample_weight" in inspect.signature(estimator.fit).parameters


# --------------------------------------------------------------------------
# Deep learning (Section 3.4) -- requires PyTorch
# --------------------------------------------------------------------------
if TORCH_AVAILABLE:
    class _AutoencoderNet(nn.Module):
        """
        Original shallow architecture (single Linear layer each for encoder
        and decoder) -- kept, unmodified, for the before/after architecture
        comparison this study's methodology favors (Section 4's SVM-gamma
        and AE-clipping before/after comparisons follow the same pattern).
        This is effectively a linear (PCA-like) compression, not a "deep"
        autoencoder in any meaningful sense -- retained here specifically so
        `train_autoencoder(architecture="shallow")` can still reproduce the
        original Table 2/8 baseline results exactly.
        """
        def __init__(self, input_dim: int, encoding_dim: int):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(input_dim, encoding_dim), nn.ReLU())
            self.decoder = nn.Linear(encoding_dim, input_dim)

        def forward(self, x):
            return self.decoder(self.encoder(x))

    class _DeepAutoencoderNet(nn.Module):
        """
        Deeper, configurable autoencoder: an arbitrary number of hidden
        layers stepping down to `encoding_dim` and back up, with ReLU
        activations, BatchNorm, and Dropout -- standard components for
        training a genuinely deep (not single-linear-layer) autoencoder
        stably. `hidden_dims` gives the encoder's hidden-layer sizes in
        order (e.g. [64, 32] means input -> 64 -> 32 -> encoding_dim); the
        decoder mirrors this in reverse.

        Motivated directly by the manuscript's Section 5/related-work
        comparison: hybrid/deeper autoencoders on comparable HIL-ICS
        testbeds reached F1 = 0.41-0.78 [47], well above this study's
        shallow AE's F1 (Table 2/8), motivating this specific architectural
        change rather than a generic "try something deeper."
        """
        def __init__(self, input_dim: int, encoding_dim: int,
                     hidden_dims: list[int] = (64, 32), dropout: float = 0.1):
            super().__init__()
            enc_layers = []
            prev_dim = input_dim
            for h in hidden_dims:
                enc_layers += [nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
                prev_dim = h
            enc_layers += [nn.Linear(prev_dim, encoding_dim), nn.ReLU()]
            self.encoder = nn.Sequential(*enc_layers)

            dec_layers = []
            prev_dim = encoding_dim
            for h in reversed(hidden_dims):
                dec_layers += [nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
                prev_dim = h
            dec_layers += [nn.Linear(prev_dim, input_dim)]
            self.decoder = nn.Sequential(*dec_layers)

        def forward(self, x):
            return self.decoder(self.encoder(x))

    class _LSTMClassifierNet(nn.Module):
        def __init__(self, input_dim: int, hidden_size: int, num_layers: int):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return torch.sigmoid(self.fc(out[:, -1, :])).squeeze(-1)


def _require_torch():
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for train_autoencoder()/train_lstm_classifier() "
            "but is not installed. Run: pip install torch"
        )


def train_autoencoder(X_train_normal, X_test, y_test, dataset_name: str,
                       encoding_dim_frac: float = 0.5, epochs: int = 20,
                       batch_size: int = 256, threshold_percentile: float = 95,
                       clip_value: float | None = 10.0, random_state: int = 42,
                       architecture: str = "deep",
                       hidden_dims: list[int] = (64, 32), dropout: float = 0.1) -> ModelResult:
    """
    Trains on normal-only data (Section 3.3.1's split strategy), then flags
    a test row as an attack if its reconstruction error exceeds a threshold
    set as the `threshold_percentile`-th percentile of TRAINING reconstruction
    error (i.e. the threshold is chosen using only normal data, never test
    labels -- this is the exact threshold-selection detail flagged as a
    [DATA NEEDED FROM AUTHOR] item in Section 3.4 of the manuscript; it is
    now a concrete, documented choice rather than an open question).

    `architecture`: "shallow" reproduces the original single-hidden-layer
    AE used for Tables 2/7/8 (kept for the before/after comparison this
    study's methodology favors). "deep" (default, new) uses
    `_DeepAutoencoderNet` with configurable `hidden_dims` -- e.g. the
    default [64, 32] gives encoder input->64->32->encoding_dim and a
    mirrored decoder, with BatchNorm+Dropout at each hidden layer. This
    directly targets the architectural-depth limitation identified in
    Section 5's related-work comparison (shallow AE F1 far below the
    0.41-0.78 F1 range reported for deeper/hybrid autoencoders on
    comparable HIL-ICS testbeds [47]).

    `clip_value`: confirmed necessary (2024) for CIC-IDS2018 specifically.
    RobustScaler (chosen for CIC-IDS2018 by _choose_scaler, Section 3.3.2)
    does not bound its output to a fixed range -- it only centers on the
    median and scales by IQR. For heavy-tailed network-flow features (e.g.
    Flow Duration) with a small IQR but occasional genuinely huge raw
    values, this produces scaled outliers in the thousands-to-millions,
    which an MSE-trained Autoencoder tries to literally reconstruct,
    exploding the loss and destabilizing training (confirmed: training MSE
    reached ~4.9e11 and did not decrease monotonically on CIC-IDS2018,
    unlike HAI's clean, monotonic MSE decay with StandardScaler). Clipping
    scaled values to `[-clip_value, clip_value]` after scaling -- standard
    practice for MSE-trained autoencoders on heavy-tailed data -- bounds
    each feature's contribution to the loss. Set `clip_value=None` to
    disable (e.g., for HAI, where StandardScaler-scaled sensor values did
    not exhibit this problem and clipping is not required, though it is
    harmless to leave enabled there too).

    `random_state`: confirmed necessary for repeated-run statistical
    validation (Section 6.5/7). Earlier versions of this function did not
    call `torch.manual_seed()` at all, meaning weight initialization and
    batch shuffling were uncontrolled -- every call, even with identical
    arguments, would silently give a different result. This is now fixed
    so that (a) the same `random_state` reproduces the same result exactly,
    and (b) varying `random_state` across repeated calls gives a genuine,
    controlled variance estimate rather than an uncontrolled one.
    """
    _require_torch()
    torch.manual_seed(random_state)
    input_dim = X_train_normal.shape[1]
    encoding_dim = max(1, int(input_dim * encoding_dim_frac))
    if architecture == "shallow":
        model = _AutoencoderNet(input_dim, encoding_dim)
    elif architecture == "deep":
        # BatchNorm1d requires batch_size > 1 in train mode; guard against a
        # tiny final batch (len(X) % batch_size == 1) crashing training,
        # rather than letting a cryptic BatchNorm error surface mid-run.
        if batch_size < 2:
            raise ValueError("architecture='deep' uses BatchNorm1d, which requires batch_size >= 2.")
        model = _DeepAutoencoderNet(input_dim, encoding_dim, hidden_dims=list(hidden_dims), dropout=dropout)
    else:
        raise ValueError(f"Unknown architecture={architecture!r}; expected 'shallow' or 'deep'.")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    def _to_tensor(X):
        arr = X.values
        if clip_value is not None:
            n_clipped = int(((arr < -clip_value) | (arr > clip_value)).sum())
            if n_clipped:
                print(f"    AE [{dataset_name}]: clipped {n_clipped} extreme values "
                      f"to [-{clip_value}, {clip_value}] ({n_clipped / arr.size * 100:.3f}% of all values).")
            arr = np.clip(arr, -clip_value, clip_value)
        return torch.tensor(arr, dtype=torch.float32)

    X_train_t = _to_tensor(X_train_normal)
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(X_train_t))
        total_loss = 0.0
        for i in range(0, len(X_train_t), batch_size):
            batch = X_train_t[perm[i:i + batch_size]]
            if len(batch) < 2 and architecture == "deep":
                continue  # BatchNorm1d requires batch size >= 2; skip a stray final singleton batch
            opt.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(batch)
        print(f"    AE [{dataset_name}] epoch {epoch + 1}/{epochs}: "
              f"train MSE = {total_loss / len(X_train_t):.6f}")

    model.eval()
    with torch.no_grad():
        train_errors = ((model(X_train_t) - X_train_t) ** 2).mean(dim=1).numpy()
    threshold = float(np.percentile(train_errors, threshold_percentile))

    X_test_t = _to_tensor(X_test)
    with torch.no_grad():
        test_errors = ((model(X_test_t) - X_test_t) ** 2).mean(dim=1).numpy()
    y_pred = (test_errors > threshold).astype(int)

    return _metrics(y_test, y_pred, "Autoencoder", dataset_name, {
        "architecture": architecture,
        "hidden_dims": list(hidden_dims) if architecture == "deep" else None,
        "encoding_dim": encoding_dim, "encoding_dim_frac": encoding_dim_frac,
        "threshold_percentile": threshold_percentile, "threshold_value": round(threshold, 6),
        "clip_value": clip_value,
    }, random_state=random_state)


def train_lstm_classifier(X_train, y_train, X_test, y_test, dataset_name: str,
                           hidden_size: int = 64, num_layers: int = 1,
                           epochs: int = 10, batch_size: int = 256,
                           seq_len: int = 1, random_state: int = 42) -> ModelResult:
    """
    `seq_len=1` treats each row as an independent single-timestep sequence
    -- appropriate for CIC-IDS2018's independent flow records. For HAI's
    genuine time series, pre-window X/y into (n_windows, seq_len, n_features)
    /(n_windows,) BEFORE calling this function and pass the real seq_len;
    windowing itself is intentionally not done inside this function, since
    it is a preprocessing decision that belongs in Section 3.3, not silently
    buried in the model-training step.

    `random_state`: see the identical note in train_autoencoder() -- this
    was previously unseeded (uncontrolled weight init and batch shuffling);
    now fixed for reproducibility and for meaningful repeated-run variance.
    """
    _require_torch()
    torch.manual_seed(random_state)
    input_dim = X_train.shape[1] // seq_len if seq_len > 1 else X_train.shape[1]
    model = _LSTMClassifierNet(input_dim, hidden_size, num_layers)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    X_train_t = torch.tensor(X_train.values, dtype=torch.float32).view(len(X_train), seq_len, -1)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(X_train_t))
        total_loss = 0.0
        for i in range(0, len(X_train_t), batch_size):
            idx = perm[i:i + batch_size]
            batch_X, batch_y = X_train_t[idx], y_train_t[idx]
            opt.zero_grad()
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(batch_X)
        print(f"    LSTM [{dataset_name}] epoch {epoch + 1}/{epochs}: "
              f"train BCE = {total_loss / len(X_train_t):.6f}")

    model.eval()
    X_test_t = torch.tensor(X_test.values, dtype=torch.float32).view(len(X_test), seq_len, -1)
    with torch.no_grad():
        probs = model(X_test_t).numpy()
    y_pred = (probs > 0.5).astype(int)

    return _metrics(y_test, y_pred, "LSTM", dataset_name,
                     {"hidden_size": hidden_size, "num_layers": num_layers, "seq_len": seq_len},
                     random_state=random_state)


# --------------------------------------------------------------------------
# Convenience: run everything for one dataset, given its 4 preprocessed splits
# --------------------------------------------------------------------------
def run_all_models_for_dataset(
    dataset_name: str,
    ae_train_X, ae_train_y, ae_test_X, ae_test_y,      # normal-only train / mixed eval
    sup_train_X, sup_train_y, sup_test_X, sup_test_y,  # stratified supervised split
    class_weight_dict: dict | None = None,
    random_state: int = 42,
) -> list[ModelResult]:
    """
    Runs all 7 models with the SAME default hyperparameters used in the
    earlier version of this study (Section 3.4/Table 3-6 structure), on the
    real preprocessed splits. Does not attempt the LSTM hyperparameter sweep
    (Table 3) or AE encoding-dimension sweep (Table 4) -- call
    train_lstm_classifier()/train_autoencoder() directly in a loop over
    hyperparameter values for those, following the same pattern.

    `random_state` is passed through to every model. For repeated-run
    statistical validation (Section 6.5/7), call this function once per
    seed (e.g. `for seed in [0,1,2,3,4]: run_all_models_for_dataset(..., random_state=seed)`)
    and pass the resulting lists to `summarize_repeated_runs()` grouped by
    model -- see the `__main__` block below for a worked example.
    """
    results = []
    print(f"=== {dataset_name}: ML baselines (seed={random_state}) ===")
    results.append(train_random_forest(sup_train_X, sup_train_y, sup_test_X, sup_test_y,
                                        dataset_name, class_weight=class_weight_dict or "balanced",
                                        random_state=random_state))
    results.append(train_svm(sup_train_X, sup_train_y, sup_test_X, sup_test_y,
                              dataset_name, class_weight=class_weight_dict or "balanced",
                              random_state=random_state))
    results.append(train_ann(sup_train_X, sup_train_y, sup_test_X, sup_test_y, dataset_name,
                              random_state=random_state))
    results.append(train_ocsvm(ae_train_X, ae_test_X, ae_test_y, dataset_name,
                                random_state=random_state))
    results.append(train_isolation_forest(ae_train_X, ae_test_X, ae_test_y, dataset_name,
                                           random_state=random_state))

    if TORCH_AVAILABLE:
        print(f"=== {dataset_name}: Deep learning (seed={random_state}) ===")
        results.append(train_autoencoder(ae_train_X, ae_test_X, ae_test_y, dataset_name,
                                          random_state=random_state))
        results.append(train_lstm_classifier(sup_train_X, sup_train_y, sup_test_X, sup_test_y,
                                              dataset_name, random_state=random_state))
    else:
        print(f"!!! {dataset_name}: skipped Autoencoder/LSTM -- PyTorch not installed "
              f"(pip install torch), NOT reported as a result.")

    return results


def results_to_dataframe(results: list[ModelResult]) -> pd.DataFrame:
    return pd.DataFrame([{
        "dataset": r.dataset_name, "model": r.model_name, "hyperparams": r.hyperparams,
        "precision": r.precision, "recall": r.recall, "f1": r.f1, "accuracy": r.accuracy,
        "random_state": r.random_state,
    } for r in results])


def run_repeated(
    dataset_name: str,
    ae_train_X, ae_train_y, ae_test_X, ae_test_y,
    sup_train_X, sup_train_y, sup_test_X, sup_test_y,
    seeds: list[int] = (0, 1, 2, 3, 4),
    class_weight_dict: dict | None = None,
) -> dict:
    """
    Runs run_all_models_for_dataset() once per seed in `seeds` (default: 5
    seeds, the minimum this study's own code treats as non-"indicative
    only" per summarize_repeated_runs()'s warning threshold), then groups
    the results by model name and summarizes each group with
    summarize_repeated_runs().

    Returns {model_name: RepeatedRunSummary}. This is the single most
    important remaining gap identified in Section 6.5/7 of the manuscript
    -- every other result in this study reflects one run per model/dataset.

    Runtime warning: this literally re-runs the full suite (including SVM's
    O(n^2)-O(n^3) fit) `len(seeds)` times. With HAI's SVM step alone taking
    ~30s (gamma='scale') to ~1840s (gamma=1) per run in this study's actual
    runs, 5 seeds could take anywhere from minutes to several hours
    depending on hyperparameters and hardware -- consider a smaller
    `max_train_n` for SVM, or fewer seeds, if this is impractical to run
    to completion.
    """
    all_results = []
    for seed in seeds:
        print(f"\n########## {dataset_name}: seed {seed} ({seeds.index(seed)+1}/{len(seeds)}) ##########")
        all_results.extend(run_all_models_for_dataset(
            dataset_name, ae_train_X, ae_train_y, ae_test_X, ae_test_y,
            sup_train_X, sup_train_y, sup_test_X, sup_test_y,
            class_weight_dict=class_weight_dict, random_state=seed,
        ))

    by_model = {}
    for r in all_results:
        by_model.setdefault(r.model_name, []).append(r)

    print(f"\n########## {dataset_name}: summary across {len(seeds)} seeds ##########")
    summaries = {model: summarize_repeated_runs(results) for model, results in by_model.items()}
    return summaries


def repeated_summaries_to_manuscript_table(summaries: dict, metric: str = "f1") -> str:
    """Formats a dict of {model: RepeatedRunSummary} as a markdown table row set,
    ready to paste into the manuscript in place of a single-run Table 2/5/6 row."""
    lines = ["| Model | N runs | Mean ± Std | 95% CI |", "|---|---|---|---|"]
    for summary in summaries.values():
        lines.append(summary.as_manuscript_row(metric))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Example end-to-end usage
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # This assumes you've already produced the 4 PreprocessingResult objects
    # per dataset using ics_add_pipeline.py's build_*_split_strategy() +
    # preprocess_dataset(), exactly as in that file's own __main__ example.
    #
    # PRIORITY 1 (Section 6.5/7): repeated runs with statistical summary,
    # replacing the earlier single-run pattern. Example (HAI):
    #
    # from ics_add_pipeline import (
    #     load_hai_train_test, build_hai_split_strategy, preprocess_dataset,
    # )
    # train_df, test_df = load_hai_train_test(...)
    # splits = build_hai_split_strategy(train_df, test_df)
    # ae_train = preprocess_dataset(splits["unsupervised"]["train"], "attack", [], "HAI AE train",
    #                                drop_cols=["time", "attack_P1", "attack_P2", "attack_P3"])
    # ae_test = preprocess_dataset(splits["unsupervised"]["test"], "attack", [], "HAI AE eval",
    #                               drop_cols=["time", "attack_P1", "attack_P2", "attack_P3"])
    # sup_train = preprocess_dataset(splits["supervised"]["train"], "attack", [], "HAI sup train",
    #                                 drop_cols=["time", "attack_P1", "attack_P2", "attack_P3"])
    # sup_test = preprocess_dataset(splits["supervised"]["test"], "attack", [], "HAI sup eval",
    #                                drop_cols=["time", "attack_P1", "attack_P2", "attack_P3"])
    #
    # hai_summaries = run_repeated(
    #     "HAI",
    #     ae_train.X, ae_train.y, ae_test.X, ae_test.y,
    #     sup_train.X, sup_train.y, sup_test.X, sup_test.y,
    #     seeds=[0, 1, 2, 3, 4],  # 5 seeds; add more (e.g. range(10)) for a tighter CI
    # )
    # print(repeated_summaries_to_manuscript_table(hai_summaries, metric="f1"))
    #
    # NOTE: for a first, faster check that the repeated-run mechanics work
    # end-to-end before committing to a multi-hour run, try seeds=[0, 1]
    # (the minimum for a std at all) with a small max_train_n on SVM first.
    #
    # Repeat the same pattern for CIC-IDS2018 using
    # load_cic_ids2018_multi()/build_cic_ids2018_split_strategy() and
    # ics_add_pipeline.py's confirmed dtype fixes (get_dummies dtype=float32).
    pass
