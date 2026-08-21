"""
ics_add_pipeline.py

Reference implementation of the Section 3.3 methodology from
"Manuscript_Draft_AnomalyDetection_ICS.md":

  PRIMARY PATH (this revision):
    - HAI:         load_hai_csv()               -- pre-labeled, ready to use
    - CIC-IDS2018: load with plain pandas.read_csv() -- pre-labeled, ready to use
    - Independent per-dataset preprocessing (3.3.2) -> preprocess_dataset()

  DEFERRED / FUTURE WORK (ICS-ADD, Section 3.2 & 7 -- kept, not deleted):
    - Flow-based feature extraction from raw PCAP  -> load_flow_features()
    - Dual-source, event-log-anchored labeling     -> label_flows_dual_source()

This script does NOT run an experiment on its own -- it has no access to the
actual dataset files. It is a documented, runnable pipeline you run yourself
once you have downloaded:
  - HAI:          https://github.com/icsdataset/hai  (Shin et al., 2020; no
                   registration required -- pick a version, e.g. HAI 21.03)
  - CIC-IDS2018:  https://www.unb.ca/cic/datasets/ids-2018.html
                  Download via: aws s3 sync --no-sign-request --region ca-central-1 \
                    "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" <dest-dir>
                  (region confirmed working: ca-central-1; a Kaggle mirror of
                  the same CSVs also exists if you'd rather avoid the AWS CLI)
                  Already ships CICFlowMeter-V3 flow-feature CSVs with labels
                  pre-assigned by the dataset's creators -- just load the
                  provided per-day CSV directly with pandas and pass it to
                  preprocess_dataset(). Any redistribution of this dataset
                  requires citing it per its license -- see manuscript
                  reference [44].
  - ICS-ADD (deferred): https://dx.doi.org/10.21227/4zht-tr07
                  (Gaggero & Armellin, 2024; requires an IEEE DataPort
                  account -- this is why HAI was chosen as the primary
                  dataset for this revision; extract_flows_with_cicflowmeter()
                  and label_flows_dual_source() below are kept ready for when
                  this becomes available, per Section 7 of the manuscript.)

Every TODO marks a place that needs your real file paths / decisions.
Nothing below invents numbers; unresolved choices raise a clear error instead
of silently defaulting, so a run either produces real results or fails loudly.
"""

from __future__ import annotations

import subprocess
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight


# --------------------------------------------------------------------------
# 0. Attack timeline (published in Gaggero et al., 2024, Table 1 / Figure 5)
# --------------------------------------------------------------------------
# These are the reported *start* timestamps of each attack step on the day
# the ICS-ADD dataset was captured. They are used only as a fallback label
# source (see label_flows_dual_source, tier 3) -- primary labels come from
# the SCADA/SIEM event logs themselves.
#
# TODO: confirm exact date + timezone of the capture from the dataset's
# metadata before using these in a real run; the paper reports times only
# (HH:MM), not a full timestamp.
ATTACK_TIMELINE = {
    "c2c_channel":         ("12:17", "12:18"),
    "port_scanning":       ("12:18", "12:19"),
    "password_bruteforce": ("12:19", "12:20"),
    "modbus_scanning":     ("12:20", "12:21"),
    "arp_spoofing":        ("12:21", "12:21"),
    "fdi_modbus":          ("12:21", "12:24"),
    "dos":                 ("12:24", "12:25"),
}
NORMAL_LABEL = "normal"


# --------------------------------------------------------------------------
# 1. Flow-based feature extraction (Section 3.3.1, paragraph 1)
# --------------------------------------------------------------------------
def extract_flows_with_cicflowmeter(pcap_path: Path, out_csv_path: Path,
                                     cicflowmeter_cmd: str = "cicflowmeter") -> Path:
    """
    Run CICFlowMeter (external CLI tool, install separately) on a raw .pcap
    to produce per-flow statistical features, matching the feature family
    already used for CIC-IDS2018 (Section 3.3.1's feature-alignment goal).

    Zeek is an equally valid alternative exporter; if you prefer it, write a
    sibling function `extract_flows_with_zeek()` that shells out to
    `zeek -r <pcap>` and post-processes conn.log into the same schema
    expected by label_flows_dual_source() (must include a 'timestamp' column).

    Raises FileNotFoundError if the exporter isn't installed, rather than
    silently skipping extraction.
    """
    if shutil.which(cicflowmeter_cmd) is None:
        raise FileNotFoundError(
            f"'{cicflowmeter_cmd}' not found on PATH. Install CICFlowMeter "
            "(https://github.com/ahlashkari/CICFlowMeter) or implement "
            "extract_flows_with_zeek() as an alternative exporter."
        )
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [cicflowmeter_cmd, str(pcap_path), str(out_csv_path.parent)],
        check=True,
    )
    return out_csv_path


def load_flow_features(flow_csv_path: Path, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Load exporter output and normalize the timestamp column to pandas datetime."""
    df = pd.read_csv(flow_csv_path)
    if timestamp_col not in df.columns:
        raise ValueError(
            f"Expected a '{timestamp_col}' column in {flow_csv_path}. "
            "CICFlowMeter's default column is usually 'Timestamp' -- "
            "rename it or pass timestamp_col=... explicitly."
        )
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    return df


def load_hai_csv(csv_path: Path, label_col: str = "attack") -> pd.DataFrame:
    """
    Load a HAI dataset CSV. Confirmed real schema (reported by the user from
    an actual downloaded HAI 21.03-family file, 2024): 80 sensor/process
    columns (P1_*, P2_*, P3_*, P4_*), a 'time' column, and FOUR label
    columns: 'attack' (binary, overall), plus 'attack_P1'/'attack_P2'/
    'attack_P3' (per-subsystem binary flags -- useful for a multi-label or
    per-process breakdown analysis beyond the binary Section 3.4 baseline).
    """
    df = pd.read_csv(csv_path)
    required = {label_col, "attack_P1", "attack_P2", "attack_P3", "time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Expected columns {required} in {csv_path}, missing: {missing}. "
            "Schema may differ between HAI versions -- re-verify before proceeding."
        )
    return df


def load_hai_train_test(
    train_files: list[Path], test_files: list[Path], label_col: str = "attack"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate HAI's multiple train/test CSVs into two DataFrames."""
    train_df = pd.concat([load_hai_csv(f, label_col) for f in train_files], ignore_index=True)
    test_df = pd.concat([load_hai_csv(f, label_col) for f in test_files], ignore_index=True)
    return train_df, test_df


def build_hai_split_strategy(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str = "attack",
    supervised_test_size: float = 0.3,
    random_state: int = 42,
) -> dict:
    """
    Implements the split strategy required because HAI's train files are
    100% normal traffic (confirmed: train attack-label value counts show a
    single class), so no supervised model can be trained on `train_df` alone
    (Section 3.4 of the manuscript).

    Returns a dict with two independent splits:
      - 'unsupervised': {'train': train_df (all normal), 'test': test_df}
        for the Autoencoder, matching HAI's intended usage.
      - 'supervised': {'train': ..., 'test': ...} a stratified split carved
        out of test_df (which contains both classes) for LSTM/RF/SVM/ANN.
        `train_df` is NOT reused here because it contributes no positive
        examples -- silently mixing it in would just inflate the normal
        class further without adding information.

    Raises if `train_df` unexpectedly contains attack examples, since that
    would mean this whole split strategy's premise is wrong and needs
    re-deriving rather than silently applied anyway.
    """
    if train_df[label_col].nunique() != 1 or train_df[label_col].iloc[0] != 0:
        raise ValueError(
            "Expected HAI train files to be 100% normal (single class, "
            f"value 0) for '{label_col}'. Got value counts: "
            f"{train_df[label_col].value_counts().to_dict()}. The supervised "
            "split strategy below assumes this; re-check before proceeding."
        )

    from sklearn.model_selection import train_test_split

    sup_train, sup_test = train_test_split(
        test_df,
        test_size=supervised_test_size,
        stratify=test_df[label_col],
        random_state=random_state,
    )

    print("--- HAI split-strategy summary (report these numbers in the manuscript) ---")
    print(f"Unsupervised (AE) train (normal-only): {len(train_df)} rows")
    print(f"Unsupervised (AE) eval (test_df, mixed): {len(test_df)} rows, "
          f"{test_df[label_col].mean()*100:.2f}% attack")
    print(f"Supervised train (stratified split of test_df): {len(sup_train)} rows, "
          f"{sup_train[label_col].mean()*100:.2f}% attack")
    print(f"Supervised eval (stratified split of test_df): {len(sup_test)} rows, "
          f"{sup_test[label_col].mean()*100:.2f}% attack")

    return {
        "unsupervised": {"train": train_df, "test": test_df},
        "supervised": {"train": sup_train, "test": sup_test},
    }


# --------------------------------------------------------------------------
# 2. Dual-source, event-log-anchored labeling (Section 3.3.1, paragraph 2)
# --------------------------------------------------------------------------
# NOTE: everything in this section (load_scada_events, load_siem_events,
# label_flows_dual_source) is for the DEFERRED, future-work ICS-ADD dataset
# (Section 3.2 / Section 7). It is not needed for the primary HAI +
# CIC-IDS2018 pipeline used in this revision -- kept here so it is ready
# once ICS-ADD access is arranged.
@dataclass
class LabelingConfig:
    fdi_tolerance: pd.Timedelta = pd.Timedelta(seconds=2)      # tier-2 tolerance
    boundary_exclusion: pd.Timedelta = pd.Timedelta(seconds=2)  # tier-3 boundary drop
    flow_ts_col: str = "timestamp"


def load_scada_events(scada_events_csv: Path, ts_col: str = "Date/Time") -> pd.DataFrame:
    """
    Load ICS-ADD's ScadaBR_events.csv (PLC pointer-value change log).
    TODO: confirm the real column names once you have the file -- 'Date/Time'
    is a placeholder; the paper describes the file's *content* (every
    configured-pointer change) but this script does not assume its exact schema.
    """
    df = pd.read_csv(scada_events_csv)
    if ts_col not in df.columns:
        raise ValueError(
            f"Expected timestamp column '{ts_col}' in {scada_events_csv}; "
            "inspect the file's real header and update ts_col."
        )
    df[ts_col] = pd.to_datetime(df[ts_col])
    return df.rename(columns={ts_col: "event_time"})


def load_siem_events(siem_events_csv: Path, ts_col: str = "Date/Time") -> pd.DataFrame:
    """Load ICS-ADD's OSSIM_Events.csv (SIEM/NIDS/firewall log)."""
    df = pd.read_csv(siem_events_csv)
    if ts_col not in df.columns:
        raise ValueError(
            f"Expected timestamp column '{ts_col}' in {siem_events_csv}; "
            "inspect the file's real header and update ts_col."
        )
    df[ts_col] = pd.to_datetime(df[ts_col])
    return df.rename(columns={ts_col: "event_time"})


def _timeline_label_for_timestamp(ts: pd.Timestamp, timeline: dict) -> str | None:
    """Tier-3 fallback only: which published attack stage (if any) contains `ts`."""
    t = ts.strftime("%H:%M")
    for stage, (start, end) in timeline.items():
        if start <= t <= end:
            return stage
    return None


def label_flows_dual_source(
    flows: pd.DataFrame,
    scada_events: pd.DataFrame,
    siem_events: pd.DataFrame,
    timeline: dict = ATTACK_TIMELINE,
    config: LabelingConfig = LabelingConfig(),
) -> pd.DataFrame:
    """
    Implements the three-tier procedure from Section 3.3.1:

      Tier 1 (primary):   fdi_modbus flows confirmed via ScadaBR_events.csv
                           (a logged PLC state change within `fdi_tolerance`
                           of the flow's timestamp, with no matching operator
                           command -- i.e. an *unexplained* state change).
      Tier 2 (secondary):  all other stages, confirmed via a matching
                           OSSIM_Events.csv entry within `fdi_tolerance`.
      Tier 3 (fallback):   remaining unlabeled flows inside a published
                           timeline window, but NOT within `boundary_exclusion`
                           of that window's edge, get a fallback label with
                           label_confidence='timeline_only'. Flows inside the
                           exclusion band are DROPPED (Section 3.3.1, item 3)
                           rather than force-labeled.
      Everything else -> 'normal'.

    Returns `flows` with two new columns: 'label' and 'label_confidence'
    ('event_log' | 'timeline_only').
    """
    ts_col = config.flow_ts_col
    flows = flows.copy()
    flows["label"] = NORMAL_LABEL
    flows["label_confidence"] = "event_log"  # default; overwritten for tier-3 rows
    drop_mask = pd.Series(False, index=flows.index)

    # --- Tier 1: FDI confirmed against ScadaBR_events.csv ------------------
    # An "unexplained" state change = a logged pointer change with no
    # corresponding operator-initiated command in the same log. The exact
    # column identifying "operator-initiated" depends on ScadaBR_events.csv's
    # real schema (TODO once the file is available) -- placeholder below
    # assumes a boolean-like column 'is_manual_command'.
    if "is_manual_command" not in scada_events.columns:
        raise ValueError(
            "ScadaBR_events.csv is expected to distinguish manual/operator "
            "commands from automatic/attack-caused changes so Tier 1 can "
            "confirm FDI flows. Update this check once the real column name "
            "is known -- do not silently treat all changes as unexplained."
        )
    unexplained_changes = scada_events.loc[~scada_events["is_manual_command"], "event_time"]

    for i, row in flows.iterrows():
        matches = (unexplained_changes - row[ts_col]).abs() <= config.fdi_tolerance
        if matches.any():
            flows.at[i, "label"] = "fdi_modbus"
            flows.at[i, "label_confidence"] = "event_log"

    # --- Tier 2: remaining stages confirmed against OSSIM_Events.csv -------
    still_normal = flows["label"] == NORMAL_LABEL
    for stage in [s for s in timeline if s != "fdi_modbus"]:
        # TODO: replace this timestamp-only join with a real content match
        # (e.g. matching src/dst IP or the alarm's stated attack type in
        # OSSIM_Events.csv) once the file's real schema is known -- a
        # timestamp-only join is a placeholder, not the final Tier-2 rule.
        stage_events = siem_events["event_time"]
        for i, row in flows.loc[still_normal].iterrows():
            matches = (stage_events - row[ts_col]).abs() <= config.fdi_tolerance
            if matches.any():
                flows.at[i, "label"] = stage
                flows.at[i, "label_confidence"] = "event_log"

    # --- Tier 3: timeline fallback + boundary exclusion ---------------------
    still_normal = flows["label"] == NORMAL_LABEL
    for i, row in flows.loc[still_normal].iterrows():
        stage = _timeline_label_for_timestamp(row[ts_col], timeline)
        if stage is None:
            continue
        start_str, end_str = timeline[stage]
        day = row[ts_col].strftime("%Y-%m-%d")
        start_ts = pd.Timestamp(f"{day} {start_str}")
        end_ts = pd.Timestamp(f"{day} {end_str}")
        near_boundary = (
            abs(row[ts_col] - start_ts) <= config.boundary_exclusion
            or abs(row[ts_col] - end_ts) <= config.boundary_exclusion
        )
        if near_boundary:
            drop_mask.at[i] = True
        else:
            flows.at[i, "label"] = stage
            flows.at[i, "label_confidence"] = "timeline_only"

    n_dropped = int(drop_mask.sum())
    flows = flows.loc[~drop_mask].reset_index(drop=True)

    # Reporting block required by Section 3.3.1's final [DATA NEEDED] note:
    print("--- Labeling summary (report these numbers in the manuscript) ---")
    print(flows.groupby(["label", "label_confidence"]).size().to_string())
    print(f"Flows dropped for boundary ambiguity: {n_dropped}")

    return flows


# --------------------------------------------------------------------------
# 3. Independent per-dataset preprocessing (Section 3.3.2)
# --------------------------------------------------------------------------
@dataclass
class PreprocessingResult:
    X: pd.DataFrame
    y: pd.Series
    scaler_name: str
    class_weights: dict = field(default_factory=dict)


def _choose_scaler(numeric_df: pd.DataFrame) -> tuple[str, object]:
    """
    Data-driven scaler selection per dataset, replacing the single fixed
    scaler reused across datasets in the earlier version of this study
    (Section 3.3.2 / 6.3). Heuristic: high outlier prevalence -> RobustScaler;
    heavy-tailed but not extremely outlier-dominated -> StandardScaler;
    otherwise MinMaxScaler. This heuristic itself should be reported and can
    be replaced with a validation-set comparison of downstream model
    performance if you want a stronger justification than a heuristic.
    """
    outlier_frac = {}
    for col in numeric_df.columns:
        q1, q3 = numeric_df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            outlier_frac[col] = 0.0
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_frac[col] = ((numeric_df[col] < lower) | (numeric_df[col] > upper)).mean()

    mean_outlier_frac = float(np.mean(list(outlier_frac.values()))) if outlier_frac else 0.0
    if mean_outlier_frac > 0.05:
        return "RobustScaler", RobustScaler()
    elif mean_outlier_frac > 0.01:
        return "StandardScaler", StandardScaler()
    else:
        return "MinMaxScaler", MinMaxScaler()


def preprocess_dataset(
    df: pd.DataFrame,
    label_col: str,
    categorical_cols: list[str],
    dataset_name: str,
    drop_cols: list[str] | None = None,
) -> PreprocessingResult:
    """
    One-hot encodes categorical_cols, scales the remaining numeric columns
    with an independently chosen scaler (see _choose_scaler), and computes
    class weights for the (likely imbalanced) label distribution -- all per
    dataset rather than shared, per Section 3.3.2.

    `drop_cols`: columns to exclude entirely before feature processing --
    e.g. HAI's 'time' (a string timestamp, not a numeric feature) and its
    extra per-subsystem label columns 'attack_P1'/'attack_P2'/'attack_P3'
    (targets, not inputs, when using the overall 'attack' column as label_col).
    Confirmed necessary (2024): omitting this caused a TypeError inside
    _choose_scaler's quantile computation, since 'time' is a string column
    and would otherwise silently be treated as numeric.
    """
    df = df.drop(columns=drop_cols) if drop_cols else df
    y = df[label_col]
    X_raw = df.drop(columns=[label_col])

    X_cat = pd.get_dummies(X_raw[categorical_cols], columns=categorical_cols, dtype=np.float32) if categorical_cols else pd.DataFrame(index=X_raw.index)
    numeric_cols = [c for c in X_raw.columns if c not in categorical_cols]
    X_num = X_raw[numeric_cols]

    non_numeric = [c for c in numeric_cols if not pd.api.types.is_numeric_dtype(X_num[c])]
    if non_numeric:
        raise ValueError(
            f"Column(s) {non_numeric} in '{dataset_name}' are non-numeric but not "
            f"listed in categorical_cols or drop_cols. Add them to one or the other "
            f"explicitly -- do not let this pass silently, as it will corrupt scaling."
        )

    scaler_name, scaler = _choose_scaler(X_num)
    X_num_scaled = pd.DataFrame(
        scaler.fit_transform(X_num), columns=numeric_cols, index=X_num.index
    )

    X = pd.concat([X_num_scaled, X_cat], axis=1)

    dup_cols = X.columns[X.columns.duplicated()].unique().tolist()
    if dup_cols:
        raise ValueError(
            f"Duplicate column names in '{dataset_name}' after encoding: {dup_cols}. "
            "This typically means a categorical column had mixed int/string "
            "representations of the same category across concatenated input "
            "files (confirmed cause for CIC-IDS2018's 'Protocol' column, now "
            "fixed at the source in load_cic_ids2018_csv) -- check dtypes of "
            "the raw categorical column(s) before one-hot encoding rather than "
            "letting sklearn fail deep inside model.fit()."
        )

    # Confirmed bug (2024): mixing float32 (scaled numeric columns) and bool
    # (pandas' default get_dummies dtype) columns makes DataFrame.values
    # silently return dtype=object, which sklearn tolerates but
    # torch.tensor() rejects outright -- failing deep inside
    # train_autoencoder()/train_lstm_classifier(), far from this actual
    # cause. Now fixed at the source (get_dummies(..., dtype=np.float32)
    # above), but verified here too so any future regression is caught here.
    if X.values.dtype == object:
        raise TypeError(
            f"'{dataset_name}': X.values has dtype=object, which will break "
            "torch.tensor() in train_autoencoder()/train_lstm_classifier() "
            "even though sklearn models will silently accept it. Check for "
            "a dtype mismatch between one-hot-encoded and scaled numeric "
            "columns (e.g. bool vs float32) rather than letting this surface "
            "later as a confusing torch error."
        )

    classes = sorted(y.unique())
    weights = compute_class_weight(class_weight="balanced", classes=np.array(classes), y=y)
    class_weights = dict(zip(classes, weights))

    print(f"--- Preprocessing summary for {dataset_name} (report in manuscript) ---")
    print(f"Chosen scaler: {scaler_name}")
    print(f"Class weights: {class_weights}")

    return PreprocessingResult(X=X, y=y, scaler_name=scaler_name, class_weights=class_weights)


def load_cic_ids2018_csv(
    csv_path: Path,
    label_col: str = "Label",
    sample_n: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load a CIC-IDS2018 per-day CSV. Confirmed real schema (from an actual
    downloaded file, 2024): 80 CICFlowMeter-V3 columns including 'Protocol'
    (a numeric IANA protocol code, but nominal -- kept as a categorical
    column for one-hot encoding), a string 'Timestamp' column, and a string
    'Label' column (e.g. 'Benign' or an attack-type name).

    Confirmed real-world issue #1 (from an actual run, 2024): reading the
    raw CSV triggers a pandas DtypeWarning across nearly all "numeric"
    columns, and naive quantile computation later fails with
    `TypeError: '<' not supported between instances of 'int' and 'str'`.
    This is CIC-IDS2018's well-known embedded-duplicate-header-row artifact
    (a literal header row re-embedded mid-file when source files were
    concatenated). Handled explicitly below.

    Confirmed real-world issue #2 (from an actual run, 2024): combining
    multiple full day-files (~1M rows each, 80 columns, with string
    Timestamp/Label columns) exceeded available RAM during
    train_test_split on a consumer machine (~5.2M rows total triggered a
    numpy MemoryError). This function now (a) downcasts numeric dtypes to
    float32/int32 after cleaning -- roughly halving numeric memory use --
    and (b) optionally performs a STRATIFIED per-file sample via `sample_n`
    (preserving the file's original attack rate, not just a naive
    `df.sample()`, since a non-stratified sample from a >90%-attack day
    like 02-21 could otherwise distort that day's attack rate). Both
    changes are reported explicitly (row counts, sampled attack rate)
    since they are data-volume-reduction decisions that must appear in the
    manuscript's Section 3.2, not a silent shortcut.

    This function:
      1) drops 'Timestamp' (not a numeric feature),
      2) drops any row where `label_col`'s value equals the literal header
         name (the embedded-header-row artifact) -- reported explicitly,
      3) adds a binary 'attack' column (Label != 'Benign'),
      4) coerces every column except {label_col, 'attack', 'Protocol'} to
         numeric with pd.to_numeric(errors='coerce'),
      5) replaces +/-inf with NaN in 'Flow Byts/s'/'Flow Pkts/s' and drops
         those rows, then drops any *other* row left with NaN,
      6) downcasts numeric dtypes to reduce memory,
      7) if `sample_n` is set and the file has more rows than `sample_n`,
         takes a stratified-by-'attack' sample of that size.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    csv_path = Path(csv_path)  # normalize in case a plain string was passed
    required = {label_col, "Protocol", "Timestamp", "Flow Byts/s", "Flow Pkts/s"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Expected columns {required} in {csv_path}, missing: {missing}. "
            "CIC-IDS2018 CSVs sometimes have leading/trailing whitespace or "
            "a BOM in column names -- try df.columns = df.columns.str.strip() "
            "if this fires unexpectedly."
        )

    df = df.drop(columns=["Timestamp"])

    n_before = len(df)
    df = df[df[label_col] != label_col]
    n_header_rows = n_before - len(df)
    if n_header_rows:
        print(f"CIC-IDS2018: dropped {n_header_rows} embedded duplicate-header rows.")

    # Confirmed bug (2024): a file affected by the embedded-header-row issue
    # gets its 'Protocol' column inferred as object/string dtype by pandas
    # (e.g. "6") for ALL rows in that file, while unaffected files infer it
    # as int (6). Concatenating multiple day-files then mixes int(6) and
    # str("6") in the same column, which get_dummies renders as the SAME
    # column name ("Protocol_6") for two DIFFERENT category values -- causing
    # a duplicate-column-name crash deep inside sklearn/narwhals at model-fit
    # time, far from this actual cause. Casting to a single consistent dtype
    # here, before any concatenation across files happens, prevents this.
    df["Protocol"] = df["Protocol"].astype(str)

    df["attack"] = (df[label_col] != "Benign").astype(int)

    numeric_cols = [c for c in df.columns if c not in (label_col, "attack", "Protocol")]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    n_before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Flow Byts/s", "Flow Pkts/s"])
    n_dropped_inf = n_before - len(df)
    if n_dropped_inf:
        print(f"CIC-IDS2018: dropped {n_dropped_inf} rows with Infinity in "
              f"'Flow Byts/s' or 'Flow Pkts/s' (zero-duration flows).")

    n_before = len(df)
    df = df.dropna(subset=numeric_cols)
    n_dropped_malformed = n_before - len(df)
    if n_dropped_malformed:
        print(f"CIC-IDS2018: dropped {n_dropped_malformed} additional rows with "
              f"non-numeric/malformed values in numeric columns.")

    # Downcast to reduce memory footprint (confirmed necessary -- see docstring).
    float_cols = df[numeric_cols].select_dtypes(include="float64").columns
    int_cols = df[numeric_cols].select_dtypes(include="int64").columns
    df[float_cols] = df[float_cols].astype("float32")
    df[int_cols] = df[int_cols].astype("int32")

    if sample_n is not None and len(df) > sample_n:
        frac = sample_n / len(df)
        # NOTE: deliberately NOT using groupby("attack").apply(...) here --
        # confirmed (2024) that pandas >=2.2's grouping-column-exclusion
        # behavior in DataFrameGroupBy.apply() silently drops the 'attack'
        # column from the result when the applied function returns a frame
        # that still contains it, causing a downstream KeyError. Sampling
        # each class subset explicitly avoids relying on that behavior.
        parts = [
            group.sample(frac=frac, random_state=random_state)
            for _, group in df.groupby("attack")
        ]
        df = pd.concat(parts, ignore_index=True)
        print(f"CIC-IDS2018: stratified-sampled {csv_path.name} down to {len(df)} rows "
              f"(attack rate preserved: {df['attack'].mean()*100:.2f}%).")

    return df


def load_cic_ids2018_multi(
    csv_paths: list[Path],
    label_col: str = "Label",
    sample_n_per_file: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load and concatenate multiple CIC-IDS2018 per-day CSVs (each cleaned via
    load_cic_ids2018_csv). Combining several attack-days is recommended over
    a single day, since each CIC-IDS2018 day typically covers only one or
    two attack types -- a single day would give CIC-IDS2018 far less attack
    diversity than HAI's dozens of scenarios, undermining the cross-dataset
    comparison in Section 5. Reports the per-file and combined row counts so
    which days were actually used is documented, not left implicit.

    `sample_n_per_file`: if set, each file is stratified-sampled down to at
    most this many rows (preserving that file's attack rate) before
    concatenation -- confirmed necessary in practice, since combining full
    day-files (~1M rows each) caused a MemoryError during train_test_split
    on a consumer machine. Report the chosen value and resulting combined
    size explicitly in the manuscript; this is a stated data-volume
    reduction, not a silent one.
    """
    frames = []
    for p in csv_paths:
        df = load_cic_ids2018_csv(p, label_col=label_col, sample_n=sample_n_per_file, random_state=random_state)
        print(f"  {p.name}: {len(df)} rows, {df['attack'].mean()*100:.2f}% attack")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    print(f"CIC-IDS2018 combined: {len(combined)} rows from {len(csv_paths)} day-files, "
          f"{combined['attack'].mean()*100:.2f}% attack overall.")
    return combined


def build_cic_ids2018_split_strategy(
    df: pd.DataFrame,
    label_col: str = "attack",
    ae_normal_train_frac: float = 0.7,
    supervised_test_size: float = 0.3,
    random_state: int = 42,
) -> dict:
    """
    Mirrors build_hai_split_strategy()'s two-split design (Section 3.3.1),
    adapted because CIC-IDS2018 -- unlike HAI -- does NOT ship a
    pre-separated, normal-only train file; a single day's CSV mixes normal
    and attack traffic together. To keep the Autoencoder's training regime
    comparable across both datasets (train on normal-only, per Section 3.4),
    this function carves a normal-only training split out of the combined
    data itself, rather than silently training the AE on mixed data (which
    would make the two datasets' AE results not comparable):

      - Unsupervised (AE) train: `ae_normal_train_frac` of the NORMAL rows only.
      - Unsupervised (AE) eval: the remaining normal rows + ALL attack rows.
      - Supervised (LSTM classifier, RF, SVM, ANN) train/eval: a stratified
        split of the FULL combined dataset (both classes), since these
        models require attack examples in training, unlike HAI where no
        such examples existed in the train files at all.
    """
    from sklearn.model_selection import train_test_split
    import gc

    normal_df = df[df[label_col] == 0]
    attack_df = df[df[label_col] == 1]

    ae_train, normal_holdout = train_test_split(
        normal_df, train_size=ae_normal_train_frac, random_state=random_state
    )
    ae_eval = pd.concat([normal_holdout, attack_df]).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)
    del normal_df, attack_df, normal_holdout
    gc.collect()

    sup_train, sup_test = train_test_split(
        df, test_size=supervised_test_size, stratify=df[label_col], random_state=random_state
    )
    gc.collect()

    print("--- CIC-IDS2018 split-strategy summary (report these numbers in the manuscript) ---")
    print(f"Unsupervised (AE) train (normal-only, {ae_normal_train_frac:.0%} of normal rows): {len(ae_train)} rows")
    print(f"Unsupervised (AE) eval (remaining normal + all attack): {len(ae_eval)} rows, "
          f"{ae_eval[label_col].mean()*100:.2f}% attack")
    print(f"Supervised train (stratified split of full combined set): {len(sup_train)} rows, "
          f"{sup_train[label_col].mean()*100:.2f}% attack")
    print(f"Supervised eval (stratified split of full combined set): {len(sup_test)} rows, "
          f"{sup_test[label_col].mean()*100:.2f}% attack")

    return {
        "unsupervised": {"train": ae_train, "test": ae_eval},
        "supervised": {"train": sup_train, "test": sup_test},
    }



# --------------------------------------------------------------------------
# 4. Example end-to-end usage (edit paths, then run)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # ---- Primary path (this revision): HAI + CIC-IDS2018 ----
    # HAI: confirmed real file layout is 3 train CSVs (100% normal) + 5 test
    # CSVs (mixed normal/attack, ~2.23% attack rate).
    HAI_DIR = Path("data/hai")
    hai_train_df, hai_test_df = load_hai_train_test(
        train_files=[HAI_DIR / f"train{i}.csv" for i in (1, 2, 3)],
        test_files=[HAI_DIR / f"test{i}.csv" for i in (1, 2, 3, 4, 5)],
    )
    hai_splits = build_hai_split_strategy(hai_train_df, hai_test_df)

    # Autoencoder: train on normal-only data, evaluate on the full test set.
    ae_train_result = preprocess_dataset(
        hai_splits["unsupervised"]["train"],
        label_col="attack", categorical_cols=[], dataset_name="HAI (AE train, normal-only)",
    )
    ae_test_result = preprocess_dataset(
        hai_splits["unsupervised"]["test"],
        label_col="attack", categorical_cols=[], dataset_name="HAI (AE eval)",
    )

    # LSTM / RF / SVM / ANN: stratified split of the labeled test set.
    sup_train_result = preprocess_dataset(
        hai_splits["supervised"]["train"],
        label_col="attack", categorical_cols=[], dataset_name="HAI (supervised train)",
    )
    sup_test_result = preprocess_dataset(
        hai_splits["supervised"]["test"],
        label_col="attack", categorical_cols=[], dataset_name="HAI (supervised eval)",
    )

    # CIC-IDS2018: combine multiple day-files for attack diversity (Section
    # 3.2). sample_n_per_file caps memory use -- confirmed necessary: the
    # full 5-file combination (~5.2M rows) caused a MemoryError during
    # train_test_split on a consumer machine. 150k/file -> ~750k rows total,
    # comfortably fits in memory while preserving each day's attack rate.
    CIC_DIR = Path("data/cic_ids2018")
    cic_df = load_cic_ids2018_multi(
        csv_paths=[
            CIC_DIR / "02-14-2018.csv",  # Brute-Force
            CIC_DIR / "02-16-2018.csv",  # DoS
            CIC_DIR / "02-21-2018.csv",  # DDoS
            CIC_DIR / "02-22-2018.csv",  # Web Attack
            CIC_DIR / "03-02-2018.csv",  # Botnet
        ],
        sample_n_per_file=150_000,
    )
    cic_splits = build_cic_ids2018_split_strategy(cic_df)

    cic_ae_train_result = preprocess_dataset(
        cic_splits["unsupervised"]["train"].drop(columns=["Label"]),
        label_col="attack", categorical_cols=["Protocol"], dataset_name="CIC-IDS2018 (AE train, normal-only)",
    )
    cic_ae_test_result = preprocess_dataset(
        cic_splits["unsupervised"]["test"].drop(columns=["Label"]),
        label_col="attack", categorical_cols=["Protocol"], dataset_name="CIC-IDS2018 (AE eval)",
    )
    cic_sup_train_result = preprocess_dataset(
        cic_splits["supervised"]["train"].drop(columns=["Label"]),
        label_col="attack", categorical_cols=["Protocol"], dataset_name="CIC-IDS2018 (supervised train)",
    )
    cic_sup_test_result = preprocess_dataset(
        cic_splits["supervised"]["test"].drop(columns=["Label"]),
        label_col="attack", categorical_cols=["Protocol"], dataset_name="CIC-IDS2018 (supervised eval)",
    )

    # ---- Deferred path (future work, Section 7): ICS-ADD ----
    # Uncomment once ICS-ADD access is arranged.
    #
    # ICS_ADD_PCAP = Path("data/ics_add/traffic_capture_span.pcap")
    # ICS_ADD_FLOW_CSV = Path("data/ics_add/traffic_capture_span_flows.csv")
    # ICS_ADD_SCADA_EVENTS = Path("data/ics_add/ScadaBR_events.csv")
    # ICS_ADD_SIEM_EVENTS = Path("data/ics_add/OSSIM_Events.csv")
    #
    # if not ICS_ADD_FLOW_CSV.exists():
    #     extract_flows_with_cicflowmeter(ICS_ADD_PCAP, ICS_ADD_FLOW_CSV)
    #
    # flows = load_flow_features(ICS_ADD_FLOW_CSV)
    # scada_events = load_scada_events(ICS_ADD_SCADA_EVENTS)
    # siem_events = load_siem_events(ICS_ADD_SIEM_EVENTS)
    # labeled = label_flows_dual_source(flows, scada_events, siem_events)
    # ics_add_result = preprocess_dataset(
    #     labeled, label_col="label", categorical_cols=[], dataset_name="ICS-ADD"
    # )
