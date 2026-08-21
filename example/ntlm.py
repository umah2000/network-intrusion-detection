from ics_add_pipeline import (
    load_hai_train_test, build_hai_split_strategy,
    load_cic_ids2018_multi, build_cic_ids2018_split_strategy,
    preprocess_dataset,
)
from train_models import run_all_models_for_dataset, results_to_dataframe
from pathlib import Path

from train_models import train_svm


# --- HAI ---
HAI_DIR = Path("hai01")
train_df, test_df = load_hai_train_test(
    train_files=[HAI_DIR / f"train{i}.csv" for i in (1,2,3)],
    test_files=[HAI_DIR / f"test{i}.csv" for i in (1,2,3,4,5)],
)
splits = build_hai_split_strategy(train_df, test_df)

ae_train = preprocess_dataset(
    splits["unsupervised"]["train"], "attack", [], "HAI AE train",
    drop_cols=["time", "attack_P1", "attack_P2", "attack_P3"],
)
ae_test = preprocess_dataset(
    splits["unsupervised"]["test"], "attack", [], "HAI AE eval",
    drop_cols=["time", "attack_P1", "attack_P2", "attack_P3"],
)
sup_train = preprocess_dataset(
    splits["supervised"]["train"], "attack", [], "HAI sup train",
    drop_cols=["time", "attack_P1", "attack_P2", "attack_P3"],
)
sup_test = preprocess_dataset(
    splits["supervised"]["test"], "attack", [], "HAI sup eval",
    drop_cols=["time", "attack_P1", "attack_P2", "attack_P3"],
)
result = train_svm(sup_train.X, sup_train.y, sup_test.X, sup_test.y, "HAI", gamma="scale")
"""
hai_results = run_all_models_for_dataset(
    "HAI",
    ae_train.X, ae_train.y, ae_test.X, ae_test.y,
    sup_train.X, sup_train.y, sup_test.X, sup_test.y,
)
print(results_to_dataframe(hai_results))
"""
