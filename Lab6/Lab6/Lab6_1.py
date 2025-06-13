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

# ���������, ��� � ��� ����������� ����������� ����������
# pip install pymorphy3 nltk

# �������� ������ �� �����
def load_lyrics(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# ������������ � ��������
def process_lyrics(lyrics):
    morph = MorphAnalyzer()
    
    # ���������� ������ �� �����
    words = re.findall(r'\b\w+\b', lyrics.lower())  # �������� � ������� �������� � ��������� �����
    lemmatized_words = []
    stemmed_words = []
    
    for word in words:
        # ������������
        parsed_word = morph.parse(word)[0]
        lemmatized_words.append(parsed_word.normal_form)
        
        # �������� (� ������ ������ ������ ���������� ������������, ��� ��� pymorphy3 �� ������������ ��������)
        stemmed_words.append(parsed_word.normal_form)  # � pymorphy3 �������� �� ����������, ���������� ������������
    
    return lemmatized_words, stemmed_words


# ���� � ����� � ������� �����
file_path = r"C:\Users\avazb\Desktop\BigData\Lab6\Lab6\kanye_lyrics.txt"

# �������� ������
lyrics = load_lyrics(file_path)

lowercase_lyrics = lyrics.lower()

cleaned_lyrics = re.sub(r'[^\w\s]', '', lyrics.lower())

# (�����������) ������ ���������� � ����� ����
output_path = r"C:\Users\avazb\Desktop\BigData\Lab6\Lab6\kanye_lyrics_lowercase.txt"
with open(output_path, 'w', encoding='utf-8') as file:
    file.write(cleaned_lyrics)

lyrics = lowercase_lyrics

# ��������� ������
lemmatized_words, stemmed_words = process_lyrics(lyrics)

# ����� �����������
print("����������������� �����:")
print(lemmatized_words[:50])  # ������� ������ 50 ����������������� ����

print("\n�������������� �����:")
print(stemmed_words[:50])  # ������� ������ 50 �������������� ����





# �������� ����������� �������� NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# �������� ����-���� �� ���������� �����
stop_words = set(stopwords.words('english'))

# ������� ��� �������� ����-����
def remove_stopwords(text):
    # ����������� ������
    words = word_tokenize(text)
    # ���������� ����
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# ������ ������ �� �����
file_path = r"C:\Users\avazb\Desktop\BigData\Lab6\Lab6\kanye_lyrics_lowercase.txt"
with open(file_path, 'r', encoding='utf-8') as file:
    lyrics = file.read()

# �������� ����-����
cleaned_lyrics = remove_stopwords(lyrics)

# ���������� ���������� ������ � ����� ����
cleaned_file_path = r"C:\Users\avazb\Desktop\BigData\Lab6\Lab6\cleaned_kanye_lyrics.txt"
with open(cleaned_file_path, 'w', encoding='utf-8') as cleaned_file:
    cleaned_file.write(cleaned_lyrics)

print(f"��������� ������ ��������� � {cleaned_file_path}")



#8#8#8#8#8#8#8#8#8#8#8#8#8#8#8#8#8#8#8#8#8#8
# �������� ������� TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# ���������� TF-IDF � ���������� ������
tfidf_matrix = tfidf_vectorizer.fit_transform([cleaned_lyrics])

# ��������� �������� � �������� TF-IDF
feature_names = tfidf_vectorizer.get_feature_names_out()
dense = tfidf_matrix.todense()
denselist = dense.tolist()
df_tfidf = pd.DataFrame(denselist, columns=feature_names)

# ����� TF-IDF ��������
print(df_tfidf)


# ������������ TF-IDF �������� �� ���� ������
tfidf_sum = df_tfidf.sum(axis=0)

# ���������� ���� �� �������� TF-IDF ��������
sorted_words = tfidf_sum.sort_values(ascending=False)

# ����� 10 �������� ������ ����
print("10 �������� ������ ����:")
print(sorted_words.head(10))


# ��������� WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(sorted_words)

# ������������ WordCloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


sentences = nltk.sent_tokenize(lyrics)  # ���������� �� �����������
tokenized_sentences = [word_tokenize(re.sub(r'\W+', ' ', sentence.lower())) for sentence in sentences]

# �������� ������ Word2Vec
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# ���������� ������
model.save(r"C:\Users\avazb\Desktop\BigData\Lab6\Lab6\kanye_word2vec.model")

print("������ Word2Vec ������� � ���������.")

# ��������� ������ ����
word = "test"
num_similar_words = 10

try:
    similar_words = model.wv.most_similar(word, topn=num_similar_words)
except KeyError:
    print(f"����� '{word}' ����������� � ������� ������.")
    similar_words = []

print(f"������ ����� � '{word}':")
for similar_word, similarity in similar_words:
    print(f"{similar_word}: {similarity:.4f}")




# ��������� 15 �������� ������ ���� �� ������
word_freq = Counter(model.wv.index_to_key)
most_common_words = word_freq.most_common(15)
words = [word for word, _ in most_common_words]

# ��������� �������� ��� �������� ������ ����
word_vectors = np.array([model.wv[word] for word in words])

# ���������� t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(words) - 1))
word_vectors_2d = tsne.fit_transform(word_vectors)

# ������������
plt.figure(figsize=(10, 6))
plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1])

# ���������� �����
for i, w in enumerate(words):
    plt.annotate(w, (word_vectors_2d[i, 0], word_vectors_2d[i, 1]))

plt.title("t-SNE ������������ 15 �������� ������ ����")
plt.xlabel("t-SNE ��������� 1")
plt.ylabel("t-SNE ��������� 2")
plt.grid()
plt.show()