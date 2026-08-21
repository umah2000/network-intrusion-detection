# Network Intrusion Detection using Deep Learning

A comprehensive system for detecting threats in computer networks using deep learning algorithms and artificial intelligence techniques.

## 📋 Project Description

This project compares different machine learning and deep learning algorithms for detecting network attacks:

- **Autoencoder (Unsupervised analysis)**
- **SVM** (Support Vector Machine)
- **Random Forest**
- **LSTM Classifier**
- **Artificial Neural Networks**

### Supported Datasets

1. **HAI Dataset** - Real-world attacks on industrial control systems
2. **CIC-IDS2018** - Training dataset for threat detection

## 🚀 Installation and Setup

### System Requirements

- Python 3.8+
- pip

### Download and install

```bash
# Clone the repositorygit
 clone https://github.com/yourusername/network-intrusion-detection.git

cd network-intrusion-detection

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # روی Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📦 Main dependencies

```
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
keras>=2.8.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 📁 Project structure

```
network-intrusion-detection/
├── train_models.py           #Model training functions
├── ics_add_pipeline.py       #Data loading and preprocessing
├── examples/
│   ├── hai_example.py        # Example for HAI Dataset
│   ├── cic_ids2018_example.py # Example for CIC-IDS2018
├── requirements.txt          #Project dependencies
├── LICENSE                   # License
└── README.md                 # This file
```
## 💻 How to use

### Simple example: Using CIC-IDS2018

```python
from ics_add_pipeline import (
    load_cic_ids2018_multi,
    build_cic_ids2018_split_strategy,
    preprocess_dataset,
)
from train_models import run_all_models_for_dataset, results_to_dataframe
from pathlib import Path

# Load data

CIC_DIR = Path("path/to/cic_ids2018")
selected_files = [
    CIC_DIR / "02-14-2018.csv",  # Brute-Force
    CIC_DIR / "02-16-2018.csv",  # DoS
    CIC_DIR / "02-21-2018.csv",  # DDoS
    CIC_DIR / "02-22-2018.csv",  # Web Attack
    CIC_DIR / "03-02-2018.csv",  # Botnet
]

cic_df = load_cic_ids2018_multi(selected_files, sample_n_per_file=150_000)
splits = build_cic_ids2018_split_strategy(cic_df)

# Data preprocessing

ae_train = preprocess_dataset(
    splits["unsupervised"]["train"].drop(columns=["Label"]),
    "attack", ["Protocol"], "CIC AE train"
)
ae_test = preprocess_dataset(
    splits["unsupervised"]["test"].drop(columns=["Label"]),
    "attack", ["Protocol"], "CIC AE eval"
)

sup_train = preprocess_dataset(
    splits["supervised"]["train"].drop(columns=["Label"]),
    "attack", ["Protocol"], "CIC sup train"
)
sup_test = preprocess_dataset(
    splits["supervised"]["test"].drop(columns=["Label"]),
    "attack", ["Protocol"], "CIC sup eval"
)

# Training all models

results = run_all_models_for_dataset(
    "CIC-IDS2018",
    ae_train.X, ae_train.y, ae_test.X, ae_test.y,
    sup_train.X, sup_train.y, sup_test.X, sup_test.y,
)

# Show results

print(results_to_dataframe(results))
```

### Autoencoder Tutorial


```python
from train_models import train_autoencoder

# Shallow version

ae_shallow = train_autoencoder(
    ae_train.X, ae_test.X, ae_test.y, 
    "CIC-IDS2018", 
    architecture="shallow"
)

# Deep version (Deep)

ae_deep = train_autoencoder(
    ae_train.X, ae_test.X, ae_test.y, 
    "CIC-IDS2018", 
    architecture="deep"
)

print(f"Shallow F1: {ae_shallow.f1}")
print(f"Deep F1: {ae_deep.f1}")
```

### Setting hyperparameters


```python
from train_models import tune_random_forest, tune_ann

# Random Forest setup

rf_tuned = tune_random_forest(
    sup_train.X, sup_train.y, sup_test.X, sup_test.y, 
    "CIC-IDS2018",
    n_iter=15, cv=3, max_search_n=50_000,
)

# Neural network tuning

ann_tuned = tune_ann(
    sup_train.X, sup_train.y, sup_test.X, sup_test.y, 
    "CIC-IDS2018",
    n_iter=15, cv=3, max_search_n=50_000,
)

print(f"RF F1: {rf_tuned.f1}, Hyperparams: {rf_tuned.hyperparams}")
print(f"ANN F1: {ann_tuned.f1}, Hyperparams: {ann_tuned.hyperparams}")
```

## 📊 Main functions
### `train_models.py` module

- `train_svm()` - SVM training
- `train_autoencoder()` - Autoencoder training
- `train_lstm_classifier()` - LSTM training
- `tune_random_forest()` - Random Forest tuning
- `tune_ann()` - Neural network tuning
- `run_all_models_for_dataset()` - Run all models
- `run_repeated()` - Repeated run with different seeds

### `ics_add_pipeline.py` module

- `load_hai_train_test()` - Load HAI Dataset
- `load_cic_ids2018_multi()` - Load CIC-IDS2018
- `build_hai_split_strategy()` - HAI segmentation
- `build_cic_ids2018_split_strategy()` - CIC-IDS2018 segmentation
- `preprocess_dataset()` - Preprocessing and Normalization

## 📈 Evaluation Criteria

- **F1-Score**
- **Precision**
- **Recall**
- **Accuracy**
- **ROC-AUC**
## 📝 Available examples

Working examples in the `examples/` folder:

1. `hai_example.py` - Using HAI Dataset
2. `cic_ids2018_example.py` - Using CIC-IDS2018


## 🔍 Important points

- **Big data**: If the dataset is very large, use `sample_n_per_file`
- **Limited RAM**: Reduce `max_search_n` value in hyperparameter tuning
- **Repeatability**: Always set seed for repeatable results

## 🤝 Contribute

To contribute:

1. Fork
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is released under the MIT License. See the `LICENSE` file for details.

## 👨‍💻 Authors

- **Your Name** - Fatahi, Mohammad; Barati, Hmazeh.

## 📞 Contact & Support

If you have any questions:
- Open Issues on GitHub
- Contact me

## 🔗 Related Resources


- [CIC-IDS2018 Dataset](https://www.unb.ca/cic/datasets/ids-2018.html)
- [HAI Dataset](https://github.com/LoJoDR/HAI-1.0)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow/Keras](https://www.tensorflow.org/)

## 📚 Research references

- Deep learning approaches for network intrusion detection
- Unsupervised learning with Autoencoders
- Hyperparameter optimization techniques

---

**Last update**: 2026
