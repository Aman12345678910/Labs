import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import os
import time
import torch
from concurrent.futures import ThreadPoolExecutor

def process_file_highly_optimized(filename):
    """Сильно оптимизированная версия для больших файлов"""
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Файл {filename} не найден")
    
    start_time = time.time()
    
    # Чтение файла с учетом уже разделенных абзацев
    print("Чтение файла с разделенными абзацами...")
    with open(filename, 'r', encoding='utf-8') as file:
        # Читаем все строки и сразу разделяем по пустым строкам
        content = file.read()
    
    # Разделяем на абзацы по пустым строкам и фильтруем
    raw_paragraphs = content.split('\n\n')
    paragraphs = []
    
    for i, paragraph in enumerate(raw_paragraphs):
        clean_paragraph = paragraph.strip()
        # Убираем лишние переносы внутри абзаца и оставляем только значимые абзацы
        if clean_paragraph and len(clean_paragraph) > 10:
            # Заменяем множественные переносы строк на пробелы внутри абзаца
            clean_paragraph = ' '.join(clean_paragraph.split('\n'))
            paragraphs.append(clean_paragraph)
    
    print(f"Найдено абзацев: {len(raw_paragraphs)}")
    print(f"Обработано значимых абзацев: {len(paragraphs)}")
    
    # Загрузка модели с оптимизациями
    print("Загрузка модели с оптимизациями...")
    model = SentenceTransformer(
        'sberbank-ai/sbert_large_nlu_ru',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Создание эмбеддингов
    print("Создание эмбеддингов...")
    
    # Разбиваем на батчи для лучшего управления памятью
    def process_batch(batch_texts):
        return model.encode(
            batch_texts,
            batch_size=64,  # Большой размер батча
            show_progress_bar=False,
            convert_to_numpy=True
        )
    
    # Обрабатываем файлы батчами
    batch_size = 1000
    all_embeddings = []
    
    total_batches = (len(paragraphs) - 1) // batch_size + 1
    
    for i in range(0, len(paragraphs), batch_size):
        batch = paragraphs[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"Обработка батча {batch_num}/{total_batches} ({len(batch)} абзацев)")
        
        batch_embeddings = process_batch(batch)
        all_embeddings.append(batch_embeddings)
    
    # Объединяем все эмбеддинги
    embeddings = np.vstack(all_embeddings)
    
    # Быстрая нормализация
    print("Нормализация векторов...")
    normalized_embeddings = normalize(embeddings, norm='l2', axis=1)
    
    # Оптимизированное создание строк с эмбеддингами
    print("Форматирование векторов...")
    
    # Используем многопоточность для форматирования больших наборов данных
    def format_embedding(embedding):
        return "[" + ", ".join(f"{x:.6f}" for x in embedding) + "]"
    
    # Определяем оптимальное количество потоков
    max_workers = min(32, (os.cpu_count() or 1) * 4)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        embedding_strings = list(executor.map(format_embedding, normalized_embeddings))
    
    # Быстрое создание DataFrame
    df = pd.DataFrame({
        'id': range(1, len(paragraphs) + 1),
        'embedding': embedding_strings
    })
    
    # Быстрое сохранение
    output_filename = 'embeddings_fast.csv'
    df.to_csv(output_filename, index=False, encoding='utf-8')
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Сохранено в {output_filename}")
    print(f"Общее время: {total_time:.2f} секунд")
    print(f"Обработано абзацев: {len(paragraphs)}")
    print(f"Скорость обработки: {len(paragraphs)/total_time:.2f} абзацев/секунду")
    print(f"Размерность эмбеддингов: {normalized_embeddings.shape}")
    print(f"Использовалось устройство: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    # Информация о памяти если используется GPU
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Использование GPU памяти: {memory_used:.2f} GB")
    
    return df, normalized_embeddings

def check_environment():
    """Проверка окружения и доступных ресурсов"""
    print("=" * 50)
    print("ПРОВЕРКА ОКРУЖЕНИЯ")
    print("=" * 50)
    
    # Проверка GPU
    if torch.cuda.is_available():
        print(f"✅ GPU доступен: {torch.cuda.get_device_name()}")
        print(f"   Количество GPU: {torch.cuda.device_count()}")
        print(f"   CUDA версия: {torch.version.cuda}")
    else:
        print("❌ GPU не доступен, используется CPU")
    
    # Проверка CPU
    print(f"✅ CPU ядер: {os.cpu_count()}")
    
    # Проверка памяти
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"✅ Оперативная память: {memory.total / 1024**3:.1f} GB")
        print(f"   Доступно: {memory.available / 1024**3:.1f} GB")
    except ImportError:
        print("ℹ️  Установите psutil для детальной информации о памяти: pip install psutil")
    
    print("=" * 50)

def analyze_file_structure(filename):
    """Анализ структуры файла для понимания разделения на абзацы"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Файл {filename} не найден")
    
    print("Анализ структуры файла...")
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Анализируем разделение на абзацы
    raw_paragraphs = content.split('\n\n')
    
    print(f"Всего блоков (разделенных пустыми строками): {len(raw_paragraphs)}")
    
    # Анализируем длину абзацев
    paragraph_lengths = [len(p.strip()) for p in raw_paragraphs if p.strip()]
    
    if paragraph_lengths:
        print(f"Длина абзацев: min={min(paragraph_lengths)}, max={max(paragraph_lengths)}, avg={np.mean(paragraph_lengths):.1f}")
        
        # Показываем примеры абзацев
        print("\nПримеры абзацев:")
        print("-" * 40)
        for i, p in enumerate(raw_paragraphs[:3]):
            if p.strip():
                preview = p.strip()[:80] + "..." if len(p.strip()) > 80 else p.strip()
                print(f"Абзац {i+1}: {preview}")
                print("-" * 40)

def main():
    """Основная функция"""
    
    # Проверяем окружение
    check_environment()
    
    # Файл для обработки
    input_file = 'GG.txt'
    
    try:
        # Анализируем структуру файла
        analyze_file_structure(input_file)
        
        print(f"\nНачинаем обработку файла: {input_file}")
        print("Это может занять некоторое время в зависимости от размера файла...")
        
        # Запускаем оптимизированную обработку
        df, embeddings = process_file_highly_optimized(input_file)
        
        # Выводим результаты
        print("\n" + "=" * 50)
        print("РЕЗУЛЬТАТЫ")
        print("=" * 50)
        print(f"✅ Успешно создано {len(df)} эмбеддингов")
        print(f"✅ Файл сохранен как: embeddings_fast.csv")
        print(f"✅ Размерность каждого вектора: {embeddings.shape[1]}")
        
        # Показываем примеры
        print("\nПримеры первых 3 записей:")
        print("-" * 30)
        for i in range(min(3, len(df))):
            print(f"ID: {df.iloc[i]['id']}")
            embedding_preview = df.iloc[i]['embedding'][:100] + "..." if len(df.iloc[i]['embedding']) > 100 else df.iloc[i]['embedding']
            print(f"Вектор: {embedding_preview}")
            print("-" * 30)
            
    except FileNotFoundError:
        print(f"❌ Ошибка: Файл {input_file} не найден!")
        print("Убедитесь, что файл находится в той же папке, что и скрипт.")
    except Exception as e:
        print(f"❌ Произошла ошибка: {e}")
        print("\nВозможные решения:")
        print("1. Проверьте, что установлены все зависимости: pip install sentence-transformers pandas scikit-learn torch")
        print("2. Убедитесь, что файл GG.txt существует")
        print("3. Проверьте доступную память")

# Дополнительные утилиты
def estimate_processing_time(filename):
    """Оценка времени обработки файла"""
    if not os.path.exists(filename):
        return "Файл не найден"
    
    # Получаем размер файла
    file_size = os.path.getsize(filename) / 1024 / 1024  # в MB
    
    # Более точная оценка на основе реальной структуры файла
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    
    raw_paragraphs = content.split('\n\n')
    estimated_paragraphs = len([p for p in raw_paragraphs if p.strip() and len(p.strip()) > 10])
    
    # Эмпирическая оценка времени
    if torch.cuda.is_available():
        estimated_time_seconds = estimated_paragraphs / 50  # ~50 абзацев/секунду на GPU
    else:
        estimated_time_seconds = estimated_paragraphs / 10  # ~10 абзацев/секунду на CPU
    
    estimated_time_minutes = estimated_time_seconds / 60
    
    print(f"Оценка обработки файла {filename}:")
    print(f"  Размер файла: {file_size:.1f} MB")
    print(f"  Примерное количество абзацев: {estimated_paragraphs}")
    print(f"  Оценочное время: {estimated_time_minutes:.1f} минут")
    
    return estimated_time_minutes

if __name__ == "__main__":
    # Оцениваем время обработки
    estimate_processing_time('GG.txt')
    
    # Запускаем основную обработку
    main()