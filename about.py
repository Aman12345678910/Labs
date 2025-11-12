import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# Загрузка модели
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Чтение данных
df = pd.read_csv('questions_clean.csv')

# Создание векторных представлений для текста
texts = df['query'].fillna('').astype(str).tolist()
embeddings = model.encode(texts)

# Преобразование векторов в строки с запятыми вместо пробелов
vector_strings = []
for vector in embeddings:                                    
    # Преобразуем numpy array в список, затем в строку с запятыми
    vector_list = vector.tolist()
    vector_str = '[' + ','.join(map(str, vector_list)) + ']'
    vector_strings.append(vector_str)

# Заменяем текст на векторные представления
df['query'] = vector_strings

# Сохранение результата
df.to_csv('questions_Vectorizes.csv', index=False)