
import sqlite3  # модуль для работы с SQLite
from pathlib import Path  # удобная работа с путями

import numpy as np  # численные операции
import pandas as pd  # табличные данные
import matplotlib  # базовый matplotlib
matplotlib.use("TkAgg")  # принудительно показывать отдельные окна графиков
import matplotlib.pyplot as plt  # построение графиков
import seaborn as sns  # красивые статистические графики


# -----------------------------
# Настройки и вспомогательные функции
# -----------------------------
def connect_to_database(db_path: Path) -> sqlite3.Connection:
    """Подключение к SQLite."""
    return sqlite3.connect(db_path)  # возвращаем объект соединения


def get_db_path() -> Path:
    """Относительный путь к базе данных, чтобы скрипт работал на любом ПК."""
    # Скрипт лежит в Lab1/Lab1, база — в Lab1/Formula 1 Race Data
    base_dir = Path(__file__).resolve().parents[1]  # поднимаемся на папку Lab1
    return base_dir / "Formula 1 Race Data" / "Formula1.sqlite"  # полный путь к БД


def ensure_output_dir() -> Path:
    """Папка для сохранения графиков."""
    out_dir = Path(__file__).resolve().parent / "plots"  # путь к папке plots
    out_dir.mkdir(parents=True, exist_ok=True)  # создаём папку, если её нет
    return out_dir  # возвращаем путь




# -----------------------------
# Загрузка и объединение данных
# -----------------------------
def load_joined_dataframe(conn: sqlite3.Connection) -> pd.DataFrame:
    """Объединяем несколько таблиц в одну для анализа."""
    query = """
        SELECT
            r.raceId,
            r.year,
            r.round,
            r.name AS race_name,
            r.date,
            c.name AS circuit_name,
            c.country AS circuit_country,
            d.driverId,
            d.forename,
            d.surname,
            d.nationality AS driver_nationality,
            con.constructorId,
            con.name AS constructor_name,
            con.nationality AS constructor_nationality,
            res.grid,
            res.position,
            res.points,
            res.laps
        FROM results res
        JOIN races r ON r.raceId = res.raceId
        JOIN drivers d ON d.driverId = res.driverId
        JOIN constructors con ON con.constructorId = res.constructorId
        JOIN circuits c ON c.circuitId = r.circuitId
    """
    df = pd.read_sql_query(query, conn)  # читаем SQL в DataFrame
    return df  # возвращаем объединённую таблицу


# -----------------------------
# Анализ и визуализация
# -----------------------------
def plot_histograms(df: pd.DataFrame, out_dir: Path) -> None:
    """Две гистограммы количественных признаков."""
    # 1) Гистограмма очков
    plt.figure(figsize=(10, 5))  # создаём окно графика
    sns.histplot(df["points"].dropna(), bins=20, kde=True, color="steelblue")  # строим гистограмму
    plt.title("Распределение набранных очков (points)")  # заголовок
    plt.xlabel("Очки")  # подпись оси X
    plt.ylabel("Количество записей")  # подпись оси Y
    plt.tight_layout()  # подгоняем отступы
    plt.savefig(out_dir / "hist_points.png", dpi=150)  # сохраняем файл
    plt.show()  # показываем окно с графиком

    # 2) Гистограмма стартовой позиции
    plt.figure(figsize=(10, 5))  # создаём окно графика
    sns.histplot(df["grid"].dropna(), bins=20, kde=True, color="seagreen")  # строим гистограмму
    plt.title("Распределение стартовой позиции (grid)")  # заголовок
    plt.xlabel("Стартовая позиция")  # подпись оси X
    plt.ylabel("Количество записей")  # подпись оси Y
    plt.tight_layout()  # подгоняем отступы
    plt.savefig(out_dir / "hist_grid.png", dpi=150)  # сохраняем файл
    plt.show()  # показываем окно с графиком


def plot_multivariate(df: pd.DataFrame, out_dir: Path) -> None:
    """Многомерный график из 3–4 признаков."""
    # Берём топ-6 конструкторов по числу записей, чтобы график не был перегружен
    top_constructors = df["constructor_name"].value_counts().head(6).index  # выбираем топ
    df_top = df[df["constructor_name"].isin(top_constructors)].copy()  # фильтруем данные

    # Многомерный scatter: grid (x), position (y), points (размер), constructor (категория)
    plt.figure(figsize=(12, 6))  # создаём окно графика
    sns.scatterplot(
        data=df_top,
        x="grid",
        y="position",
        hue="constructor_name",
        size="points",
        sizes=(20, 200),
        alpha=0.7
    )  # строим точечный график
    plt.title("Связь стартовой позиции и финишной позиции\nс учётом конструктора и очков")  # заголовок
    plt.xlabel("Стартовая позиция (grid)")  # подпись X
    plt.ylabel("Финишная позиция (position)")  # подпись Y
    plt.legend(title="Конструктор", bbox_to_anchor=(1.02, 1), loc="upper left")  # легенда
    plt.tight_layout()  # подгоняем отступы
    plt.savefig(out_dir / "scatter_grid_position_points.png", dpi=150)  # сохраняем файл
    plt.show()  # показываем окно с графиком


def main() -> None:
    # Получаем путь к БД
    db_path = get_db_path()  # путь к базе
    if not db_path.exists():  # проверка наличия файла
        raise FileNotFoundError(f"База данных не найдена: {db_path}")  # ошибка, если файла нет

    # Подключаемся к БД
    conn = connect_to_database(db_path)  # открываем соединение
    out_dir = ensure_output_dir()  # папка для графиков

    try:
        # Печатаем список таблиц
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;",
            conn
        )  # SQL-запрос на список таблиц
        print("Таблицы в базе:\n", tables)  # выводим таблицы

        # Загружаем объединённый DataFrame
        df = load_joined_dataframe(conn)  # общая таблица для анализа

        # Приводим типы там, где нужно
        df["points"] = pd.to_numeric(df["points"], errors="coerce")  # очки в число
        df["grid"] = pd.to_numeric(df["grid"], errors="coerce")  # стартовая позиция в число
        df["position"] = pd.to_numeric(df["position"], errors="coerce")  # финишная позиция в число
        df["laps"] = pd.to_numeric(df["laps"], errors="coerce")  # круги в число

        # Краткое описание данных
        print("\nРазмер объединённой таблицы:", df.shape)  # размер DataFrame
        print("\nПервые строки:\n", df.head())  # первые строки
        print(
            "\nСтатистическое описание числовых признаков:\n",
            df[["points", "grid", "position", "laps"]].describe()
        )  # статистика числовых признаков

        # Одномерный анализ
        plot_histograms(df, out_dir)  # строим две гистограммы

        # Многомерный анализ
        plot_multivariate(df, out_dir)  # строим многомерный график

    finally:
        # Закрываем соединение с БД
        conn.close()  # закрываем соединение


if __name__ == "__main__":
    main()

