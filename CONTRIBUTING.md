# راهنمای مشارکت

از اینکه می‌خواهید در این پروژه مشارکت کنید سپاس می‌گوییم! 🙏

## مراحل مشارکت

### 1. Fork کردن مخزن

```bash
# روی GitHub صفحه پروژه را باز کنید
# بر روی دکمه Fork کلیک کنید
```

### 2. کلون کردن Fork شما

```bash
git clone https://github.com/yourusername/network-intrusion-detection.git
cd network-intrusion-detection
```

### 3. ایجاد یک Branch جدید

```bash
git checkout -b feature/your-feature-name
# یا
git checkout -b bugfix/bug-name
```

### 4. انجام تغییرات

- کد را بهبود دهید
- نظرات و documentation را اضافه کنید
- تست‌ها را اجرا کنید

### 5. Commit کردن تغییرات

```bash
git add .
git commit -m "توضیح تغییرات شما"
```

### نام‌گذاری مناسب برای Commit:

```
fix: رفع خطای معین
feature: افزودن ویژگی جدید
docs: بهبود documentation
refactor: بازسازی کد
test: افزودن تست‌های جدید
perf: بهبود کارایی
```

### 6. Push کردن به GitHub

```bash
git push origin feature/your-feature-name
```

### 7. ایجاد Pull Request

- به صفحه GitHub خود برگردید
- بر روی دکمه "Compare & pull request" کلیک کنید
- توضیحات واضح درباره تغییرات خود بنویسید

## استانداردهای کد

### Python Style

- از [PEP 8](https://www.python.org/dev/peps/pep-0008/) پیروی کنید
- حداکثر 100 کاراکتر در هر خط
- نام‌های متغیرها را انگلیسی و معنادار انتخاب کنید

### مثال:

```python
def train_svm(X_train, y_train, X_test, y_test, dataset_name, gamma="scale"):
    """
    آموزش SVM

    Parameters
    ----------
    X_train : array-like
        داده‌های آموزشی
    y_train : array-like
        برچسب‌های آموزشی
    X_test : array-like
        داده‌های آزمایشی
    y_test : array-like
        برچسب‌های آزمایشی
    dataset_name : str
        نام مجموعه داده
    gamma : str
        پارامتر gamma

    Returns
    -------
    ModelResult
        نتایج آموزش
    """
    pass
```

## Documentation

- توضیحات واضح برای توابع جدید اضافه کنید
- از Docstring استفاده کنید
- README را به‌روز کنید در صورت نیاز

## تست‌ها

اگر ممکن است تست‌های جدید اضافه کنید:

```python
def test_preprocess_dataset():
    """تست تابع preprocess_dataset"""
    # ...
```

## مسائل و پیشنهادات

اگر یک مشکل یا پیشنهاد دارید:

1. [Issues](../../issues) را چک کنید
2. اگر موجود نیست، یک Issue جدید ایجاد کنید
3. توضیح واضحی بنویسید

### صورت مشکل گزارش‌گیری خوب:

- **عنوان**: مختصر و واضح
- **توضیح**: جزئیات مشکل
- **مراحل تکرار**: نحوه تکرار خطا
- **رفتار انتظار‌رفته**: چه باید اتفاق بیفتد
- **اسکریین‌شات**: در صورت لزوم

## سوالات

سوالات خود را در [Discussions](../../discussions) مطرح کنید.

---

**از مشارکت شما ممنون هستیم! 🎉**
