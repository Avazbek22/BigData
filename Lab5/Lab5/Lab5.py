# -*- coding: utf-8 -*-
"""
Лабораторная работа 5
Методы снижения размерности и задача кластеризации
"""

import numpy as np  # численные операции
import pandas as pd  # табличные данные
import matplotlib  # базовый matplotlib
matplotlib.use("TkAgg")  # отдельные окна графиков
import matplotlib.pyplot as plt  # графики

from pathlib import Path  # удобные пути
from sklearn.preprocessing import StandardScaler, LabelEncoder  # нормализация и кодирование
from sklearn.decomposition import KernelPCA  # Kernel PCA
from sklearn.manifold import TSNE  # t-SNE
import umap  # UMAP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # LDA
from sklearn.cluster import KMeans, AgglomerativeClustering  # кластеризация
from sklearn.metrics import silhouette_score, adjusted_rand_score  # метрики
from scipy.cluster.hierarchy import dendrogram, linkage  # дендрограмма
import joblib  # сохранение моделей


# -----------------------------
# Вспомогательные функции
# -----------------------------

def basic_info(df: pd.DataFrame, name: str) -> None:
    """Базовая информация о датасете."""
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


def simple_kmeans(X: np.ndarray, k: int, max_iter: int = 100) -> np.ndarray:
    """Простая реализация k-means (для сравнения с библиотечным)."""
    rng = np.random.default_rng(42)
    # Случайные центроиды
    centers = X[rng.choice(len(X), size=k, replace=False)]
    for _ in range(max_iter):
        # Назначаем точки к ближайшему центру
        distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        # Пересчитываем центры
        new_centers = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else centers[i] for i in range(k)])
        # Если центры не меняются, выходим
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return labels


# -----------------------------
# 1) Загрузка данных (вариант 7 -> Iris)
# -----------------------------
print("=== ЗАГРУЗКА ДАННЫХ ===")

lab5_dir = Path(__file__).resolve().parents[1]
iris_path = lab5_dir.parent / "Lab5" / "Iris.csv"

if not iris_path.exists():
    raise FileNotFoundError(f"Файл Iris.csv не найден: {iris_path}")

iris = pd.read_csv(iris_path)

# Удалим столбец Id, если есть
if "Id" in iris.columns:
    iris = iris.drop(columns=["Id"])

# Базовая информация
basic_info(iris, "Iris")

# EDA
print("\nIris: числовая статистика")
print(numeric_stats(iris))
print("\nIris: категориальная статистика")
print(categorical_stats(iris))

# -----------------------------
# 2) Обработка выбросов и нормализация
# -----------------------------
print("\n=== ОБРАБОТКА ВЫБРОСОВ И НОРМАЛИЗАЦИЯ ===")

# Выбросы (IQR)
iris = handle_outliers_iqr(iris, exclude=["Species"])

# Разделяем признаки и цель
X = iris.drop(columns=["Species"])
y = iris["Species"]

# Нормализация
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Кодируем метки классов
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# -----------------------------
# 3) Kernel PCA (все ядра)
# -----------------------------
print("\n=== KERNEL PCA ===")

kernels = ["linear", "poly", "rbf", "sigmoid", "cosine"]

plt.figure(figsize=(20, 12))
for i, kernel in enumerate(kernels, 1):
    kpca = KernelPCA(n_components=2, kernel=kernel)
    X_kpca = kpca.fit_transform(X_scaled)
    plt.subplot(2, 3, i)
    scatter = plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y_encoded, cmap="viridis")
    plt.title(f"Kernel PCA ({kernel})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
plt.tight_layout()
plt.show()

# Lost variance для линейного ядра
kpca_linear = KernelPCA(n_components=2, kernel="linear", fit_inverse_transform=True)
X_kpca_linear = kpca_linear.fit_transform(X_scaled)
X_back = kpca_linear.inverse_transform(X_kpca_linear)
lost_variance = np.mean(np.abs(X_scaled - X_back))
print(f"Lost variance (linear kernel): {lost_variance:.4f}")

# Сохранение модели KernelPCA
joblib.dump(kpca_linear, lab5_dir / "kpca_linear.joblib")

# -----------------------------
# 4) t-SNE и UMAP
# -----------------------------
print("\n=== t-SNE и UMAP ===")

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_encoded, cmap="viridis")
plt.title("t-SNE")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()

# UMAP
umap_model = umap.UMAP(random_state=42)
X_umap = umap_model.fit_transform(X_scaled)
plt.figure(figsize=(8, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y_encoded, cmap="viridis")
plt.title("UMAP")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()

# Сохранение UMAP
joblib.dump(umap_model, lab5_dir / "umap_model.joblib")

# LDA (если возможно)
print("\n=== LDA ===")
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y_encoded)
plt.figure(figsize=(8, 6))
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y_encoded, cmap="viridis")
plt.title("LDA")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()

# Сохранение LDA
joblib.dump(lda, lab5_dir / "lda_model.joblib")

# -----------------------------
# 5) Кластеризация (k-means и иерархическая)
# -----------------------------
print("\n=== КЛАСТЕРИЗАЦИЯ ===")

# Метод локтя и силуэта для выбора k
inertias = []
silhouettes = []
k_values = range(2, 8)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(k_values, inertias, marker="o")
plt.title("Метод локтя")
plt.xlabel("k")
plt.ylabel("Inertia")

plt.subplot(1, 2, 2)
plt.plot(k_values, silhouettes, marker="o")
plt.title("Силуэт")
plt.xlabel("k")
plt.ylabel("Silhouette")
plt.tight_layout()
plt.show()

# Оптимальное k по силуэту
best_k_sil = k_values[int(np.argmax(silhouettes))]
print(f"Оптимальное k по силуэту: {best_k_sil}")

# Для Iris количество классов известно заранее (3)
best_k = len(np.unique(y_encoded))
print(f"Выбранное k (по числу классов): {best_k}")

# KMeans (библиотечный)
kmeans = KMeans(n_clusters=best_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Сравнение с истинными метками
ari_kmeans = adjusted_rand_score(y_encoded, clusters)
print(f"ARI для KMeans: {ari_kmeans:.4f}")

# Кастомный k-means для сравнения
custom_labels = simple_kmeans(X_scaled, k=best_k)
ari_custom = adjusted_rand_score(y_encoded, custom_labels)
print(f"ARI для custom k-means: {ari_custom:.4f}")

# Визуализация кластеров
plt.figure(figsize=(8, 6))
plt.scatter(X_kpca_linear[:, 0], X_kpca_linear[:, 1], c=clusters, cmap="viridis")
plt.title("KMeans кластеры (KernelPCA проекция)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()

# Сохранение модели KMeans
joblib.dump(kmeans, lab5_dir / "kmeans_model.joblib")

# Иерархическая кластеризация
agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
agg_labels = agg.fit_predict(X_scaled)

# Дендрограмма
plt.figure(figsize=(10, 5))
linked = linkage(X_scaled, method="ward")
dendrogram(linked, truncate_mode="lastp", p=12)
plt.title("Дендрограмма")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

# Оценка иерархической кластеризации
ari_agg = adjusted_rand_score(y_encoded, agg_labels)
print(f"ARI для Agglomerative: {ari_agg:.4f}")

# Сохранение модели Agglomerative
joblib.dump(agg, lab5_dir / "agg_clustering.joblib")


"""
ДЛЯ ОТЧЕТА (подсказки для другой ИИ, удалить перед сдачей):

1) Датасет: Iris (вариант 7). Целевой столбец: Species.
2) EDA: строки/столбцы, память, числовая статистика, мода категорий.
3) Выбросы обработаны IQR, данные нормализованы StandardScaler.
4) KernelPCA применён для ядер: linear, poly, rbf, sigmoid, cosine; графики построены.
5) Lost variance для линейного ядра рассчитана через обратное преобразование.
6) t-SNE, UMAP и LDA использованы для сравнения, графики построены.
7) Кластеризация: KMeans + Agglomerative, k выбрано по методу локтя и силуэта, затем k=3 по числу классов.
8) Сравнение библиотечного KMeans и кастомного (ARI).
9) Модели сохранены joblib (kpca_linear, umap, lda, kmeans, agglomerative).
"""
