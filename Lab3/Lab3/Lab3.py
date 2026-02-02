# -*- coding: utf-8 -*-
"""
Лабораторная работа №3
Бинарная и многоклассовая классификация. Оценка качества задачи классификации
"""

import sqlite3  # работа с SQLite
from pathlib import Path  # удобные пути

import numpy as np  # численные расчеты
import pandas as pd  # табличные данные
import matplotlib  # базовый matplotlib
matplotlib.use("TkAgg")  # отдельные окна графиков
import matplotlib.pyplot as plt  # графики

from sklearn.model_selection import train_test_split  # разбиение train/test
from sklearn.preprocessing import StandardScaler  # нормализация
from sklearn.pipeline import Pipeline  # пайплайны
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.linear_model import LogisticRegression  # логистическая регрессия
from sklearn.svm import SVC, LinearSVC  # SVM
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


# -----------------------------
# Вспомогательные функции
# -----------------------------

def basic_info(df: pd.DataFrame, name: str) -> None:
    """Вывод базовой информации о датасете."""
    rows, cols = df.shape
    memory_usage = df.memory_usage(deep=True).sum()
    print(f"\n{name}: количество строк = {rows}, количество столбцов = {cols}")
    print(f"{name}: память = {memory_usage} байт")


def numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Статистика для числовых признаков."""
    numeric_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
    stats = df[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    stats = stats[["min", "50%", "mean", "max", "25%", "75%"]]
    stats.columns = ["min", "median", "mean", "max", "p25", "p75"]
    return stats


def categorical_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Статистика для категориальных признаков."""
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns
    summary = pd.DataFrame(index=cat_cols, columns=["mode", "mode_count"])
    for col in cat_cols:
        if df[col].notna().any():
            mode_val = df[col].mode()[0]
            mode_count = df[col].value_counts().iloc[0]
            summary.loc[col] = [mode_val, mode_count]
    return summary


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Обработка пропусков: числовые -> медиана, категориальные -> мода."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype in ["int64", "float64", "int32", "float32"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            if df[col].notna().any():
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna("unknown")
    return df


def handle_outliers_iqr(df: pd.DataFrame, exclude: list[str] | None = None) -> pd.DataFrame:
    """Обработка выбросов по IQR: ограничение значений в [Q1-1.5*IQR, Q3+1.5*IQR]."""
    df = df.copy()
    exclude = exclude or []
    numeric_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
    for col in numeric_cols:
        if col in exclude:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=low, upper=high)
    return df


def prepare_features(df: pd.DataFrame, target_col: str):
    """Разделение на признаки и целевую переменную, кодирование категорий."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # One-Hot Encoding для категориальных признаков
    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return X, y


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """Оценка качества модели."""
    y_pred = model.predict(X_test)

    # Метрики качества
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    err = 1 - acc

    # ROC-AUC
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    roc_auc = roc_auc_score(y_test, y_score)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{name}:")
    print(f"Accuracy = {acc:.4f}")
    print(f"Precision = {prec:.4f}")
    print(f"Recall = {rec:.4f}")
    print(f"F1 = {f1:.4f}")
    print(f"Error rate = {err:.4f}")
    print(f"ROC-AUC = {roc_auc:.4f}")
    print("Confusion matrix:\n", cm)

    return {"name": name, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "err": err, "roc_auc": roc_auc, "cm": cm}


def plot_roc_curves(results, y_test, models, X_test, title: str) -> None:
    """Построение ROC-кривых для нескольких моделей."""
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def stratified_sample(df: pd.DataFrame, target_col: str, n: int) -> pd.DataFrame:
    """Стратифицированная выборка для ускорения обучения."""
    if n >= len(df):
        return df
    # Используем train_test_split для корректной стратификации
    sample_df, _ = train_test_split(
        df, train_size=n, stratify=df[target_col], random_state=42
    )
    return sample_df.reset_index(drop=True)


# -----------------------------
# Часть 1: Общая часть (Kaggle dataset по варианту)
# Вариант 7 -> набор playground-series-s3e10 (train.csv)
# -----------------------------
print("=== ОБЩАЯ ЧАСТЬ: Kaggle dataset ===")

lab3_dir = Path(__file__).resolve().parents[1]
train_path = lab3_dir.parent / "Lab3" / "train.csv"

if not train_path.exists():
    raise FileNotFoundError(f"Файл train.csv не найден: {train_path}")

kaggle_df = pd.read_csv(train_path)

# Базовая информация
basic_info(kaggle_df, "Kaggle")

# EDA
print("\nKaggle: числовая статистика")
print(numeric_stats(kaggle_df))
print("\nKaggle: категориальная статистика")
print(categorical_stats(kaggle_df))

# Пропуски
print("\nKaggle: пропуски")
print(kaggle_df.isna().sum())

# Целевая переменная
target_col = "Class"
if target_col not in kaggle_df.columns:
    raise ValueError("Целевой столбец 'Class' не найден")

# Обработка пропусков и выбросов
kaggle_df = handle_missing(kaggle_df)
kaggle_df = handle_outliers_iqr(kaggle_df, exclude=[target_col])

# Для ускорения обучения берем стратифицированную подвыборку
kaggle_train = stratified_sample(kaggle_df, target_col, n=5000)

# Подготовка признаков
X, y = prepare_features(kaggle_train, target_col)

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Модели
models = {
    "KNN": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier())]),
    "LogReg": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]),
    "SVM": Pipeline([("scaler", StandardScaler()), ("model", LinearSVC(max_iter=2000))]),
}

# Обучение и оценка
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    res = evaluate_model(name, model, X_test, y_test)
    results.append(res)

# ROC-кривые
plot_roc_curves(results, y_test, models, X_test, "ROC-кривые (Kaggle dataset)")

# Лучший алгоритм
best = max(results, key=lambda r: r["roc_auc"])
print(f"\nЛучший алгоритм (по ROC-AUC): {best['name']}")


# -----------------------------
# Часть 2: Самостоятельная часть (данные из Лабы 1 - F1 SQLite)
# -----------------------------
print("\n=== САМОСТОЯТЕЛЬНАЯ ЧАСТЬ: F1 SQLite ===")

# Путь к БД F1
f1_db = lab3_dir.parent / "Lab1" / "Formula 1 Race Data" / "Formula1.sqlite"
if not f1_db.exists():
    raise FileNotFoundError(f"База данных не найдена: {f1_db}")

conn = sqlite3.connect(f1_db)

# Выбор нужных столбцов
query = """
SELECT
    r.raceId,
    r.year,
    r.round,
    r.name AS race_name,
    c.country AS circuit_country,
    d.nationality AS driver_nationality,
    con.name AS constructor_name,
    res.grid,
    res.positionOrder,
    res.points,
    res.laps,
    res.statusId
FROM results res
JOIN races r ON r.raceId = res.raceId
JOIN drivers d ON d.driverId = res.driverId
JOIN constructors con ON con.constructorId = res.constructorId
JOIN circuits c ON c.circuitId = r.circuitId
"""

f1_df = pd.read_sql_query(query, conn)
conn.close()

# Создаем бинарную целевую переменную: набрал очки или нет
f1_df["scored_points"] = (f1_df["points"] > 0).astype(int)

# Базовая информация
basic_info(f1_df, "F1")

# EDA
print("\nF1: числовая статистика")
print(numeric_stats(f1_df))
print("\nF1: категориальная статистика")
print(categorical_stats(f1_df))

# Пропуски
print("\nF1: пропуски")
print(f1_df.isna().sum())

# Обработка пропусков и выбросов
f1_df = handle_missing(f1_df)
f1_df = handle_outliers_iqr(f1_df, exclude=["scored_points"])

# Подготовка признаков
X2, y2 = prepare_features(f1_df.drop(columns=["points"]), "scored_points")

# Разделение на train/test
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42, stratify=y2
)

# Модели
models2 = {
    "KNN": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier())]),
    "LogReg": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]),
    "SVM": Pipeline([("scaler", StandardScaler()), ("model", LinearSVC(max_iter=2000))]),
}

# Обучение и оценка
results2 = []
for name, model in models2.items():
    model.fit(X2_train, y2_train)
    res = evaluate_model(name, model, X2_test, y2_test)
    results2.append(res)

# ROC-кривые
plot_roc_curves(results2, y2_test, models2, X2_test, "ROC-кривые (F1 dataset)")

# Лучший алгоритм
best2 = max(results2, key=lambda r: r["roc_auc"])
print(f"\nЛучший алгоритм (по ROC-AUC): {best2['name']}")


"""
ДЛЯ ОТЧЕТА (подсказки для другой ИИ, удалить перед сдачей):

1) Общая часть (Kaggle dataset, вариант 7):
- Использован train.csv из playground-series-s3e10.
- Целевая переменная: Class (бинарная классификация).
- Проведен EDA: строки/столбцы, память, статистика числовых, мода/частота категориальных.
- Пропуски обработаны, выбросы ограничены IQR.
- Категориальные признаки кодированы (OneHot).
- Для ускорения обучение выполнено на стратифицированной подвыборке (n=5000).
- Train/test split 80/20.
- Построены 3 модели: KNN, Logistic Regression, SVM (LinearSVC) с нормализацией.
- Метрики: Accuracy, Precision, Recall, F1, Error rate, Confusion Matrix, ROC-AUC.
- Построены ROC-кривые, выбран лучший алгоритм по ROC-AUC.

2) Самостоятельная часть (F1 SQLite из Лабы 1):
- Сформирован набор данных из results+races+drivers+constructors+circuits.
- Целевая переменная: scored_points (1 если points > 0, иначе 0).
- Проведен EDA, обработка пропусков и выбросов.
- Категориальные признаки: race_name, circuit_country, driver_nationality, constructor_name.
- Построены 3 модели (KNN, LogReg, SVM) с нормализацией.
- Метрики и ROC-кривые посчитаны, выбран лучший алгоритм.

3) Ответы на контрольные вопросы:
- Классификация: отнесение объектов к заранее заданным классам.
- Бинарная vs многоклассовая: 2 класса против 3+ классов.
- Логистическая регрессия: вероятность класса, линейная модель с сигмоидой.
- Отбор признаков: по p-value, регуляризации, важности, корреляции.
- Оценка качества: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix.
- KNN: основан на близости, нет обучения параметров; логрег — параметрическая модель.
"""
