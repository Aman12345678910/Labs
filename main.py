import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import numpy as np

# Загрузка модели
model = SentenceTransformer('sberbank-ai/sbert_large_nlu_ru')

# Загрузка данных
df = pd.read_csv('questions_clean.csv')

# Получение эмбеддингов
questions = df['query'].tolist()
embeddings = model.encode(questions, show_progress_bar=True)

# Нормализация векторов
normalized_embeddings = normalize(embeddings, norm='l2')

# Создание нового DataFrame с q_id и векторами в столбце query
result_df = pd.DataFrame()
result_df['q_id'] = df['q_id']

# Преобразуем векторы в списки и записываем в столбец query
result_df['query'] = normalized_embeddings.tolist()

# Сохранение результатов
result_df.to_csv('questions_with_vectors.csv', index=False)

print(f"Векторные представления сохранены в файл 'questions_with_vectors.csv'")
print(f"Размерность векторов: {normalized_embeddings.shape[1]}")
print(f"Количество вопросов: {len(questions)}")
print("Столбец 'query' теперь содержит векторные представления вместо текста")