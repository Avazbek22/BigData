# Лабораторная работа 3

**Тема:** Бинарная и многоклассовая классификация. Оценка качества задачи классификации.

**Сложность:** Well-done (общая часть + самостоятельная часть, 3 алгоритма, нормализация).

## Общая часть (Kaggle dataset)
- Датасет: `train.csv` (playground-series-s3e10)
- Целевая переменная: `Class`
- EDA: размеры, память, статистика, пропуски, выбросы
- 3 модели: KNN, Logistic Regression, SVM
- Метрики: Accuracy, Precision, Recall, F1, Error rate, Confusion Matrix, ROC-AUC
- ROC-кривые и выбор лучшего алгоритма

## Самостоятельная часть (F1 SQLite из Лабы 1)
- Сформирован датасет из таблиц results+races+drivers+constructors+circuits
- Целевая переменная: `scored_points` (1 если points > 0)
- EDA, пропуски, выбросы, кодирование категорий
- 3 модели: KNN, Logistic Regression, SVM
- Метрики и ROC-кривые

## Запуск
Из корня проекта:
```bash
py -3 Lab3\Lab3\Lab3.py
```

## Зависимости
```bash
pip install numpy pandas matplotlib scikit-learn
```
