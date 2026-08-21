# Network Intrusion Detection using Deep Learning

سامانه‌ای جامع برای شناسایی تهدیدات در شبکه‌های رایانه‌ای با استفاده از الگوریتم‌های یادگیری عمیق و تکنیک‌های هوش مصنوعی.

## 📋 توصیف پروژه

این پروژه الگوریتم‌های مختلف یادگیری ماشین و یادگیری عمیق را برای تشخیص حملات شبکه‌ای مقایسه می‌کند:

- **Autoencoder (تحلیل ناظر نشده)**
- **SVM** (Support Vector Machine)
- **Random Forest**
- **LSTM Classifier**
- **Artificial Neural Networks**

### مجموعه‌های داده پشتیبانی شده

1. **HAI Dataset** - حملات واقعی در سیستم‌های کنترلی صنعتی
2. **CIC-IDS2018** - مجموعه داده آموزشی برای تشخیص تهدیدات

## 🚀 نصب و راه‌اندازی

### الزامات سیستمی

- Python 3.8+
- pip

### دریافت و نصب

```bash
# کلون کردن مخزن
git clone https://github.com/yourusername/network-intrusion-detection.git
cd network-intrusion-detection

# ایجاد محیط مجازی (اختیاری اما توصیه شده)
python -m venv venv
source venv/bin/activate  # روی Windows: venv\Scripts\activate

# نصب وابستگی‌ها
pip install -r requirements.txt
```

## 📦 وابستگی‌های اصلی

```
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
keras>=2.8.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 📁 ساختار پروژه

```
network-intrusion-detection/
├── train_models.py           # توابع آموزش مدل‌ها
├── ics_add_pipeline.py       # بارگذاری و پیش‌پردازش داده‌ها
├── examples/
│   ├── hai_example.py        # مثال برای HAI Dataset
│   ├── cic_ids2018_example.py # مثال برای CIC-IDS2018
│   └── hyperparameter_tuning.py # تنظیم ابرپارامترها
├── requirements.txt          # وابستگی‌های پروژه
├── LICENSE                   # مجوز
└── README.md                 # این فایل
```

## 💻 نحوه استفاده

### مثال ساده: استفاده از CIC-IDS2018

```python
from ics_add_pipeline import (
    load_cic_ids2018_multi,
    build_cic_ids2018_split_strategy,
    preprocess_dataset,
)
from train_models import run_all_models_for_dataset, results_to_dataframe
from pathlib import Path

# بارگذاری داده‌ها
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

# پیش‌پردازش داده‌ها
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

# آموزش تمام مدل‌ها
results = run_all_models_for_dataset(
    "CIC-IDS2018",
    ae_train.X, ae_train.y, ae_test.X, ae_test.y,
    sup_train.X, sup_train.y, sup_test.X, sup_test.y,
)

# نمایش نتایج
print(results_to_dataframe(results))
```

### آموزش Autoencoder

```python
from train_models import train_autoencoder

# نسخه سطحی (Shallow)
ae_shallow = train_autoencoder(
    ae_train.X, ae_test.X, ae_test.y, 
    "CIC-IDS2018", 
    architecture="shallow"
)

# نسخه عمیق (Deep)
ae_deep = train_autoencoder(
    ae_train.X, ae_test.X, ae_test.y, 
    "CIC-IDS2018", 
    architecture="deep"
)

print(f"Shallow F1: {ae_shallow.f1}")
print(f"Deep F1: {ae_deep.f1}")
```

### تنظیق ابرپارامترها

```python
from train_models import tune_random_forest, tune_ann

# تنظیق Random Forest
rf_tuned = tune_random_forest(
    sup_train.X, sup_train.y, sup_test.X, sup_test.y, 
    "CIC-IDS2018",
    n_iter=15, cv=3, max_search_n=50_000,
)

# تنظیق شبکه عصبی
ann_tuned = tune_ann(
    sup_train.X, sup_train.y, sup_test.X, sup_test.y, 
    "CIC-IDS2018",
    n_iter=15, cv=3, max_search_n=50_000,
)

print(f"RF F1: {rf_tuned.f1}, Hyperparams: {rf_tuned.hyperparams}")
print(f"ANN F1: {ann_tuned.f1}, Hyperparams: {ann_tuned.hyperparams}")
```

## 📊 توابع اصلی

### ماژول `train_models.py`

- `train_svm()` - آموزش SVM
- `train_autoencoder()` - آموزش Autoencoder
- `train_lstm_classifier()` - آموزش LSTM
- `tune_random_forest()` - تنظیق Random Forest
- `tune_ann()` - تنظیق شبکه عصبی
- `run_all_models_for_dataset()` - اجرای تمام مدل‌ها
- `run_repeated()` - اجرای تکراری با seed‌های مختلف

### ماژول `ics_add_pipeline.py`

- `load_hai_train_test()` - بارگذاری HAI Dataset
- `load_cic_ids2018_multi()` - بارگذاری CIC-IDS2018
- `build_hai_split_strategy()` - تقسیم‌بندی HAI
- `build_cic_ids2018_split_strategy()` - تقسیم‌بندی CIC-IDS2018
- `preprocess_dataset()` - پیش‌پردازش و نرمال‌سازی

## 📈 معیارهای ارزیابی

- **F1-Score**
- **Precision**
- **Recall**
- **Accuracy**
- **ROC-AUC**

## 📝 مثال‌های موجود

مثال‌های کاربردی در پوشه `examples/`:

1. `hai_example.py` - استفاده از HAI Dataset
2. `cic_ids2018_example.py` - استفاده از CIC-IDS2018
3. `hyperparameter_tuning.py` - تنظیم ابرپارامترها

## 🔍 نکات مهم

- **داده‌های بزرگ**: اگر dataset بسیار بزرگ است، از `sample_n_per_file` استفاده کنید
- **RAM محدود**: مقدار `max_search_n` را در تنظیم ابرپارامترها کاهش دهید
- **تکرار‌پذیری**: همیشه seed را تنظیم کنید برای نتایج قابل تکرار

## 🤝 مشارکت

برای مشارکت:

1. Fork کنید
2. یک branch جدید ایجاد کنید (`git checkout -b feature/AmazingFeature`)
3. تغییرات را commit کنید (`git commit -m 'Add some AmazingFeature'`)
4. Push کنید (`git push origin feature/AmazingFeature`)
5. Pull Request باز کنید

## 📄 مجوز

این پروژه تحت مجوز MIT منتشر شده است. برای جزئیات به فایل `LICENSE` مراجعه کنید.

## 👨‍💻 نویسندگان

- **نام شما** - کار اصلی

## 📞 تماس و پشتیبانی

اگر سوالی دارید:
- Issues را در GitHub باز کنید
- با من تماس بگیرید

## 🔗 منابع مرتبط

- [CIC-IDS2018 Dataset](https://www.unb.ca/cic/datasets/ids-2018.html)
- [HAI Dataset](https://github.com/LoJoDR/HAI-1.0)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow/Keras](https://www.tensorflow.org/)

## 📚 مراجع تحقیقاتی

- Deep learning approaches for network intrusion detection
- Unsupervised learning with Autoencoders
- Hyperparameter optimization techniques

---

**آخرین بروزرسانی**: 2026
