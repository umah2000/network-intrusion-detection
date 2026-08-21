from ics_add_pipeline import (
    load_hai_train_test, build_hai_split_strategy,
    load_cic_ids2018_multi, build_cic_ids2018_split_strategy,
    preprocess_dataset,
)
from train_models import run_all_models_for_dataset, results_to_dataframe
from pathlib import Path

# --- HAI ---
def quick_subsample(df, label_col="attack", n=5000, random_state=42):

    if len(df) <= n:
        return df
    frac = n / len(df)
    parts = [g.sample(frac=frac, random_state=random_state) for _, g in df.groupby(label_col)]
    return pd.concat(parts, ignore_index=True)
HAI_DIR = Path("CIC")
train_df, test_df = load_hai_train_test(
    train_files=[HAI_DIR / f"train{i}.csv" for i in (1,2,3)],
    test_files=[HAI_DIR / f"test{i}.csv" for i in (1,2,3,4,5)],
)
# مثال روی HAI:
splits = build_hai_split_strategy(train_df, test_df)

small_ae_train = quick_subsample(splits["unsupervised"]["train"], n=5000)
small_ae_test  = quick_subsample(splits["unsupervised"]["test"], n=5000)
small_sup_train = quick_subsample(splits["supervised"]["train"], n=5000)
small_sup_test  = quick_subsample(splits["supervised"]["test"], n=5000)

ae_train = preprocess_dataset(small_ae_train, "attack", [], "HAI AE train (5k test)")
ae_test  = preprocess_dataset(small_ae_test, "attack", [], "HAI AE eval (5k test)")
sup_train = preprocess_dataset(small_sup_train, "attack", [], "HAI sup train (5k test)")
sup_test  = preprocess_dataset(small_sup_test, "attack", [], "HAI sup eval (5k test)")

results = run_all_models_for_dataset(
    "HAI (5k test)",
    ae_train.X, ae_train.y, ae_test.X, ae_test.y,
    sup_train.X, sup_train.y, sup_test.X, sup_test.y,
)
print(results_to_dataframe(results))