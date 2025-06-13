# -*- coding: cp1251 -*-import re
from pymorphy3 import MorphAnalyzer
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from sklearn.manifold import TSNE
from collections import Counter

# Убедитесь, что у вас установлены необходимые библиотеки
# pip install pymorphy3 nltk

# Загрузка текста из файла
def load_lyrics(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Лемматизация и стемминг
def process_lyrics(lyrics):
    morph = MorphAnalyzer()
    
    # Разделение текста на слова
    words = re.findall(r'\b\w+\b', lyrics.lower())  # Приводим к нижнему регистру и извлекаем слова
    lemmatized_words = []
    stemmed_words = []
    
    for word in words:
        # Лемматизация
        parsed_word = morph.parse(word)[0]
        lemmatized_words.append(parsed_word.normal_form)
        
        # Стемминг (в данном случае просто используем лемматизацию, так как pymorphy3 не поддерживает стемминг)
        stemmed_words.append(parsed_word.normal_form)  # В pymorphy3 стемминг не реализован, используем лемматизацию
    
    return lemmatized_words, stemmed_words


# Путь к файлу с текстом песен
file_path = r"C:\Users\avazb\Desktop\BigData\Lab6\Lab6\kanye_lyrics.txt"

# Загрузка текста
lyrics = load_lyrics(file_path)

lowercase_lyrics = lyrics.lower()

cleaned_lyrics = re.sub(r'[^\w\s]', '', lyrics.lower())

# (Опционально) Запись результата в новый файл
output_path = r"C:\Users\avazb\Desktop\BigData\Lab6\Lab6\kanye_lyrics_lowercase.txt"
with open(output_path, 'w', encoding='utf-8') as file:
    file.write(cleaned_lyrics)

lyrics = lowercase_lyrics

# Обработка текста
lemmatized_words, stemmed_words = process_lyrics(lyrics)

# Вывод результатов
print("Лемматизированные слова:")
print(lemmatized_words[:50])  # Выводим первые 50 лемматизированных слов

print("\nСтеммированные слова:")
print(stemmed_words[:50])  # Выводим первые 50 стеммированных слов





# Загрузка необходимых ресурсов NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Загрузка стоп-слов на английском языке
stop_words = set(stopwords.words('english'))

# Функция для удаления стоп-слов
def remove_stopwords(text):
    # Токенизация текста
    words = word_tokenize(text)
    # Фильтрация слов
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Чтение текста из файла
file_path = r"C:\Users\avazb\Desktop\BigData\Lab6\Lab6\kanye_lyrics_lowercase.txt"
with open(file_path, 'r', encoding='utf-8') as file:
    lyrics = file.read()

# Удаление стоп-слов
cleaned_lyrics = remove_stopwords(lyrics)

# Сохранение очищенного текста в новый файл
cleaned_file_path = r"C:\Users\avazb\Desktop\BigData\Lab6\Lab6\cleaned_kanye_lyrics.txt"
with open(cleaned_file_path, 'w', encoding='utf-8') as cleaned_file:
    cleaned_file.write(cleaned_lyrics)

print(f"Очищенные тексты сохранены в {cleaned_file_path}")



#8#8#8#8#8#8#8#8#8#8#8#8#8#8#8#8#8#8#8#8#8#8
# Создание объекта TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Применение TF-IDF к очищенному тексту
tfidf_matrix = tfidf_vectorizer.fit_transform([cleaned_lyrics])

# Получение словарей и значений TF-IDF
feature_names = tfidf_vectorizer.get_feature_names_out()
dense = tfidf_matrix.todense()
denselist = dense.tolist()
df_tfidf = pd.DataFrame(denselist, columns=feature_names)

# Вывод TF-IDF значений
print(df_tfidf)


# Суммирование TF-IDF значений по всем словам
tfidf_sum = df_tfidf.sum(axis=0)

# Сортировка слов по убыванию TF-IDF значений
sorted_words = tfidf_sum.sort_values(ascending=False)

# Вывод 10 наиболее частых слов
print("10 наиболее частых слов:")
print(sorted_words.head(10))


# Генерация WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(sorted_words)

# Визуализация WordCloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


sentences = nltk.sent_tokenize(lyrics)  # Разделение на предложения
tokenized_sentences = [word_tokenize(re.sub(r'\W+', ' ', sentence.lower())) for sentence in sentences]

# Обучение модели Word2Vec
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Сохранение модели
model.save(r"C:\Users\avazb\Desktop\BigData\Lab6\Lab6\kanye_word2vec.model")

print("Модель Word2Vec обучена и сохранена.")

# Получение схожих слов
word = "test"
num_similar_words = 10

try:
    similar_words = model.wv.most_similar(word, topn=num_similar_words)
except KeyError:
    print(f"Слово '{word}' отсутствует в словаре модели.")
    similar_words = []

print(f"Схожие слова к '{word}':")
for similar_word, similarity in similar_words:
    print(f"{similar_word}: {similarity:.4f}")




# Получение 15 наиболее частых слов из модели
word_freq = Counter(model.wv.index_to_key)
most_common_words = word_freq.most_common(15)
words = [word for word, _ in most_common_words]

# Получение векторов для наиболее частых слов
word_vectors = np.array([model.wv[word] for word in words])

# Применение t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(words) - 1))
word_vectors_2d = tsne.fit_transform(word_vectors)

# Визуализация
plt.figure(figsize=(10, 6))
plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1])

# Добавление меток
for i, w in enumerate(words):
    plt.annotate(w, (word_vectors_2d[i, 0], word_vectors_2d[i, 1]))

plt.title("t-SNE визуализация 15 наиболее частых слов")
plt.xlabel("t-SNE компонент 1")
plt.ylabel("t-SNE компонент 2")
plt.grid()
plt.show()