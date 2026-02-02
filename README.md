# BigData Labs

## Лабораторная работа 1 (Вариант №7)
**Тема:** Реляционные данные. Исследовательский анализ данных. Построение визуализаций данных OLAP.

**Сложность:** Rare

**База данных:** Formula 1 Race Data (SQLite)

### Содержание проекта (Lab1)
- `Lab1/Formula 1 Race Data/Formula1.sqlite` — база данных SQLite
- `Lab1/Lab1/Lab1.py` — скрипт анализа данных
- `Lab1/Lab1/plots/` — автоматически создаваемая папка с графиками

### Требования
- Python 3.x
- Зависимости: `pandas`, `numpy`, `matplotlib`, `seaborn`

### Установка зависимостей
```bash
pip install pandas numpy matplotlib seaborn
```

### Запуск
Из корня проекта:
```bash
py -3 Lab1\Lab1\Lab1.py
```

### Что делает скрипт
- Подключается к SQLite базе данных
- Объединяет таблицы `results`, `races`, `drivers`, `constructors`, `circuits` в один DataFrame
- Строит 2 гистограммы количественных признаков (`points`, `grid`)
- Строит 1 многомерный график (grid vs position с учетом конструктора и очков)
- Сохраняет графики в `Lab1/Lab1/plots/`
