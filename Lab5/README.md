# Лабораторная работа 5

**Тема:** Методы снижения размерности и задача кластеризации.

**Сложность:** Well-done (снижение размерности + кластеризация + сравнение).

## Датасет
- Iris (вариант 7), файл `Iris.csv`

## Что реализовано
- EDA, обработка выбросов, нормализация
- KernelPCA со всеми ядрами
- Lost variance для линейного ядра
- t-SNE и UMAP
- KMeans и Agglomerative clustering
- Метод локтя и силуэта
- Сравнение custom k-means и библиотечного (ARI)
- Сохранение моделей joblib

## Запуск
```bash
py -3 Lab5\Lab5\Lab5.py
```

## Зависимости
```bash
pip install numpy pandas matplotlib scikit-learn umap-learn scipy
```
