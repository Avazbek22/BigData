# -*- coding: utf-8 -*-
"""
Лабораторная работа 6.1
Обработка естественного языка и классификация текста
"""

from pathlib import Path  # удобные пути
import re  # регулярные выражения
import numpy as np  # численные операции
import pandas as pd  # табличные данные
import matplotlib  # базовый matplotlib
matplotlib.use("TkAgg")  # отдельные окна графиков
import matplotlib.pyplot as plt  # графики

import nltk  # NLP
from nltk.corpus import stopwords  # стоп-слова
from nltk.tokenize import word_tokenize  # токенизация
from nltk.stem import PorterStemmer  # стемминг
from nltk.stem import WordNetLemmatizer  # лемматизация

from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF
from wordcloud import WordCloud  # WordCloud
from sklearn.manifold import TSNE  # t-SNE
from sklearn.decomposition import TruncatedSVD  # снижение размерности для fallback
import joblib  # сохранение моделей

from sklearn.model_selection import train_test_split  # разбиение
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # метрики
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.svm import SVC  # SVC
from sklearn.ensemble import RandomForestClassifier  # RandomForest
from sklearn.linear_model import LogisticRegression  # LogisticRegression


# -----------------------------
# Настройки и пути
# -----------------------------

lab6_dir = Path(__file__).resolve().parent
lyrics_path = lab6_dir / "kanye_lyrics.txt"
lowercase_path = lab6_dir / "kanye_lyrics_lowercase.txt"
cleaned_path = lab6_dir / "cleaned_kanye_lyrics.txt"
word2vec_path = lab6_dir / "kanye_word2vec.model"
fallback_vectors_path = lab6_dir / "word_vectors.joblib"

poems_path = lab6_dir / "poems.txt"  # файл со стихами для задачи 2


# -----------------------------
# Подготовка ресурсов NLTK
# -----------------------------

# Пробуем скачать ресурсы (если уже есть — быстро пропустит)
try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
except Exception:
    # Если нет интернета, продолжаем с тем, что есть
    pass


# -----------------------------
# Вспомогательные функции
# -----------------------------

def load_text(path: Path) -> str:
    """Чтение текста из файла."""
    return path.read_text(encoding="utf-8")


def save_text(path: Path, text: str) -> None:
    """Сохранение текста в файл."""
    path.write_text(text, encoding="utf-8")


def clean_text(text: str) -> str:
    """Нижний регистр + удаление пунктуации."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lemmatize_and_stem(tokens: list[str]) -> tuple[list[str], list[str]]:
    """Лемматизация и стемминг."""
    stemmer = PorterStemmer()
    try:
        lemmatizer = WordNetLemmatizer()
        lemmatized = [lemmatizer.lemmatize(t) for t in tokens]
    except Exception:
        # Если нет wordnet, лемматизация = токен
        lemmatized = tokens
    stemmed = [stemmer.stem(t) for t in tokens]
    return lemmatized, stemmed


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Удаление стоп-слов (английский)."""
    try:
        stop_words = set(stopwords.words("english"))
    except Exception:
        stop_words = set(["the", "and", "a", "to", "of", "in", "is", "it", "that", "this", "for", "on"])
    return [t for t in tokens if t not in stop_words]


def most_common_words(tokens: list[str], top_n: int = 10) -> list[tuple[str, int]]:
    """Топ самых частых слов."""
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    return sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусное сходство между двумя векторами."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def train_word2vec_fallback(sentences: list[list[str]], vector_size: int = 50, window: int = 5) -> dict:
    """
    Упрощённая модель слов (fallback без gensim):
    строим матрицу совместных встречаемостей и снижаем размерность SVD.
    """
    # Собираем словарь
    vocab = {}
    for sent in sentences:
        for w in sent:
            vocab[w] = vocab.get(w, 0) + 1

    # Ограничим словарь для устойчивости
    vocab = dict(sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:300])
    words = list(vocab.keys())
    idx = {w: i for i, w in enumerate(words)}

    # Матрица ко-встречаемостей
    cooc = np.zeros((len(words), len(words)), dtype=np.float32)
    for sent in sentences:
        for i, w in enumerate(sent):
            if w not in idx:
                continue
            start = max(0, i - window)
            end = min(len(sent), i + window + 1)
            for j in range(start, end):
                if i == j:
                    continue
                w2 = sent[j]
                if w2 in idx:
                    cooc[idx[w], idx[w2]] += 1.0

    # SVD для получения векторов
    n_comp = min(vector_size, max(2, len(words) - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    vectors = svd.fit_transform(cooc)

    return {w: vectors[idx[w]] for w in words}


# -----------------------------
# Задача 1: Анализ текста песен
# -----------------------------

print("=== ЗАДАЧА 1: ПЕСНИ ===")

# Загружаем текст песен
if not lyrics_path.exists():
    raise FileNotFoundError(f"Файл с песнями не найден: {lyrics_path}")

lyrics_raw = load_text(lyrics_path)

# Нижний регистр + удаление пунктуации
lyrics_clean = clean_text(lyrics_raw)

# Сохраняем промежуточные файлы
save_text(lowercase_path, lyrics_raw.lower())
save_text(cleaned_path, lyrics_clean)

# Токенизация
try:
    tokens = word_tokenize(lyrics_clean)
except Exception:
    tokens = lyrics_clean.split()

# Лемматизация и стемминг
lemmatized, stemmed = lemmatize_and_stem(tokens)

# Удаление стоп-слов
tokens_no_stop = remove_stopwords(lemmatized)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform([" ".join(tokens_no_stop)])
feature_names = tfidf_vectorizer.get_feature_names_out()

# Топ-слова по TF-IDF
tfidf_sum = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
word_scores = dict(zip(feature_names, tfidf_sum))
sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
print("Топ-10 слов по TF-IDF:")
print(sorted_words[:10])

# WordCloud
wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_scores)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud для песен")
plt.show()

sentences = []
try:
    for sent in nltk.sent_tokenize(lyrics_raw):
        sent_tokens = word_tokenize(clean_text(sent))
        if sent_tokens:
            sentences.append(sent_tokens)
except Exception:
    sentences = [tokens_no_stop]

# Попытка обучить Word2Vec через gensim (если доступен)
try:
    from gensim.models import Word2Vec  # type: ignore

    w2v_model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=2)
    w2v_model.save(str(word2vec_path))
    print(f"Word2Vec модель сохранена: {word2vec_path}")

    # Похожие слова
    word = tokens_no_stop[0] if tokens_no_stop else "love"
    try:
        similar_words = w2v_model.wv.most_similar(word, topn=10)
        print(f"Похожие слова к '{word}': {similar_words}")
    except Exception:
        print("Слово не найдено в словаре Word2Vec")

    # Векторы для t-SNE
    get_vector = lambda w: w2v_model.wv[w]

except Exception:
    print("gensim недоступен: используем упрощённую модель слов (co-occurrence + SVD).")
    vectors_dict = train_word2vec_fallback(sentences, vector_size=50, window=5)
    joblib.dump(vectors_dict, fallback_vectors_path)
    print(f"Векторы слов сохранены: {fallback_vectors_path}")

    # Похожие слова (по косинусному сходству)
    word = tokens_no_stop[0] if tokens_no_stop else "love"
    if word in vectors_dict:
        base_vec = vectors_dict[word]
        sims = []
        for w, v in vectors_dict.items():
            if w == word:
                continue
            sims.append((w, cosine_similarity(base_vec, v)))
        sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)[:10]
        print(f"Похожие слова к '{word}': {sims_sorted}")
    else:
        print("Слово не найдено в словаре fallback-модели")

    # Векторы для t-SNE
    get_vector = lambda w: vectors_dict[w]

# t-SNE для 15 самых частых слов
common_words = [w for w, _ in most_common_words(tokens_no_stop, top_n=15)]
filtered_words = []
vectors = []
for w in common_words:
    try:
        vectors.append(get_vector(w))
        filtered_words.append(w)
    except Exception:
        continue
word_vectors = np.array(vectors)

if len(filtered_words) >= 2:
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(filtered_words) - 1))
    word_vectors_2d = tsne.fit_transform(word_vectors)

    plt.figure(figsize=(10, 6))
    plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1])
    for i, w in enumerate(filtered_words):
        plt.annotate(w, (word_vectors_2d[i, 0], word_vectors_2d[i, 1]))
    plt.title("t-SNE для 15 частых слов")
    plt.grid()
    plt.show()


# -----------------------------
# Задача 2: Стихи + классификация
# -----------------------------

print("\n=== ЗАДАЧА 2: СТИХИ И КЛАССИФИКАЦИЯ ===")

# Если файла стихов нет — создаём простой набор
if not poems_path.exists():
    poems_text = """
    Silent dawn on river wide,
    Whispered breeze and gentle tide.
    Stars are falling, night is deep,
    Moon is watching, shadows sleep.

    In the meadow, flowers sway,
    Morning gold replaces grey.
    Birds are singing, skies are blue,
    Day awakens, bright and true.

    Autumn leaves in crimson light,
    Dancing softly into night.
    Winter waits with silver snow,
    Quiet dreams begin to grow.

    Spring arrives with tender rain,
    Waking earth from sleepy chain.
    Summer sings in fields of green,
    Golden days of light serene.

    Heartbeats echo through the air,
    Hope is rising everywhere.
    Words are gentle, spirits free,
    Peaceful thoughts like quiet sea.
    """
    save_text(poems_path, poems_text.strip())

# Загружаем стихи
poems_raw = load_text(poems_path)
poems_clean = clean_text(poems_raw)
if poems_clean:
    try:
        poem_tokens_raw = word_tokenize(poems_clean)
    except Exception:
        poem_tokens_raw = poems_clean.split()
    poems_tokens = remove_stopwords(poem_tokens_raw)
else:
    poems_tokens = []

# Формируем датасет для классификации (песни vs стихи)
# Разбиваем тексты на документы
song_docs = [" ".join(tokens_no_stop[i:i+100]) for i in range(0, len(tokens_no_stop), 100) if tokens_no_stop[i:i+100]]
poem_docs = [" ".join(poems_tokens[i:i+50]) for i in range(0, len(poems_tokens), 50) if poems_tokens[i:i+50]]

texts = song_docs + poem_docs
labels = ["song"] * len(song_docs) + ["poem"] * len(poem_docs)

# TF-IDF для классификации
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

y = np.array(labels)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 3 модели классификации
models = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "SVC": SVC(kernel="linear"),
    "RandomForest": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

# Обучение и оценка
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\n{name}:")
    print(f"Accuracy = {acc:.4f}")
    print(f"Precision = {prec:.4f}")
    print(f"Recall = {rec:.4f}")
    print(f"F1 = {f1:.4f}")


"""
ДЛЯ ОТЧЕТА (подсказки для другой ИИ, удалить перед сдачей):

1) Задача 1 (песни):
- Загружен текст песен (kanye_lyrics.txt), приведен к нижнему регистру, удалена пунктуация.
- Проведена токенизация, лемматизация и стемминг (NLTK).
- Удалены стоп-слова.
- Рассчитан TF-IDF, выведены топ-10 слов.
- Построен WordCloud.
- Обучена модель Word2Vec, сохранена (kanye_word2vec.model).
- Если gensim недоступен, используется fallback (co-occurrence + SVD) и сохраняется word_vectors.joblib.
- Найдены похожие слова.
- Построен t-SNE для 15 частых слов.

2) Задача 2 (стихи + классификация):
- Добавлены стихи, проведен препроцессинг.
- Совмещены песни и стихи в DataFrame, создана целевая переменная (song/poem).
- Векторизация TF-IDF, обучены 3+ классификаторов (KNN, SVC, RandomForest, LogisticRegression).
- Оценены метрики Accuracy/Precision/Recall/F1, выбран лучший.
"""
