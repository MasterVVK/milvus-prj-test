import dspy
import requests
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType, Index
from dspy.retrieve.milvus_rm import MilvusRM  # Используем Milvus как Retriever Model

# ========== 1. ПОДКЛЮЧЕНИЕ К MILVUS ==========
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "document_parts"

connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

# ========== 2. СОЗДАНИЕ КОЛЛЕКЦИИ ==========
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)
    print(f"❌ Коллекция {COLLECTION_NAME} удалена!")

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),  # Храним оригинальный текст
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)  # Размерность эмбеддинга
]

schema = CollectionSchema(fields, description="Коллекция документов")
collection = Collection(name=COLLECTION_NAME, schema=schema)

print(f"✅ Коллекция {COLLECTION_NAME} создана с размерностью 1024!")

# ========== 3. СОЗДАНИЕ ИНДЕКСА ==========
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128}
}

collection.create_index(field_name="embedding", index_params=index_params)
print(f"✅ Индекс создан для {COLLECTION_NAME}!")


# ========== 4. ФУНКЦИЯ ГЕНЕРАЦИИ ЭМБЕДДИНГОВ ==========
def generate_embedding(text):
    """Генерация 1024-мерного эмбеддинга через Ollama"""
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "mxbai-embed-large", "prompt": text}  # 💡 `prompt` теперь строка
        )
        response_data = response.json()

        if "embedding" not in response_data:
            print(f"❌ Ошибка! API Ollama не вернул 'embedding'. Полный ответ:\n{response_data}")
            return None

        embedding = response_data["embedding"]
        return embedding if isinstance(embedding, list) else None  # ✅ Убеждаемся, что это список float

    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка запроса к Ollama: {e}")
        return None
    except requests.exceptions.JSONDecodeError:
        print(f"❌ Ошибка JSON! Сервер Ollama вернул некорректный ответ:\n{response.text}")
        return None


# ========== 5. ФУНКЦИЯ ИНДЕКСАЦИИ ДОКУМЕНТОВ ==========
def index_documents(docs):
    """Добавление списка документов в Milvus"""
    embeddings = []
    texts = []

    for text in docs:
        embedding = generate_embedding(text)
        if embedding is not None:
            embeddings.append(embedding)
            texts.append(text)
        else:
            print(f"⚠️ Пропускаем документ: {text[:50]}... (не удалось сгенерировать эмбеддинг)")

    if not embeddings:
        print("❌ Ошибка: ни один документ не был добавлен.")
        return

    collection.insert([texts, embeddings])
    print(f"✅ {len(embeddings)} документов успешно добавлены в {COLLECTION_NAME}!")

    collection.load()
    print(f"📥 Коллекция {COLLECTION_NAME} загружена в память!")


# ========== 6. ФУНКЦИЯ ПОИСКА В MILVUS ==========
def search_documents(query):
    """Поиск в Milvus"""
    embedding = generate_embedding(query)
    if embedding is None:
        print("❌ Ошибка: не удалось сгенерировать эмбеддинг для запроса.")
        return

    collection.load()
    search_results = collection.search(
        data=[embedding],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=5
    )

    for i, result in enumerate(search_results[0]):
        print(f"🔍 Результат {i + 1}: {result.id}, расстояние: {result.distance}")


# ========== 7. НАСТРОЙКА LLM и RM ==========
lm = dspy.LM("openai/gemma:2b", api_base="http://localhost:11434")


class CustomMilvusRM(MilvusRM):
    """Кастомный Milvus Retriever с генерацией эмбеддингов через Ollama"""

    def __init__(self, collection_name):
        super().__init__(collection_name=collection_name)
        self.embedding_function = generate_embedding  # 💡 Указываем Ollama как источник эмбеддингов

    def forward(self, query):
        """Переопределяем forward, чтобы использовать Ollama"""
        embedding = self.embedding_function(query)
        if embedding is None:
            return []
        return super().forward(query)


milvus_retriever = CustomMilvusRM(collection_name=COLLECTION_NAME)

dspy.configure(lm=lm, rm=milvus_retriever)  # Настраиваем Milvus как Retriever Model


# ========== 8. DSPy КЛАСС ДЛЯ ИЗВЛЕЧЕНИЯ СОБЫТИЙ ==========
class Event(dspy.Signature):
    description = dspy.InputField(desc="Textual description of the event, including name, location, and dates")
    event_name = dspy.OutputField(desc="Name of the event")
    location = dspy.OutputField(desc="Location of the event")
    start_date = dspy.OutputField(desc="Start date of the event, YYYY-MM-DD")
    end_date = dspy.OutputField(desc="End date of the event, YYYY-MM-DD")


class EventExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retriever = dspy.Retrieve(k=3)  # Используем Milvus для извлечения документов
        self.predict = dspy.Predict(Event)

    def forward(self, query: str):
        results = self.retriever.forward(query)
        events = [self.predict(description=doc) for doc in results.passages]
        return events


# ========== 9. ТЕСТОВЫЕ ДАННЫЕ ==========
documents = [
    "Blockchain Expo Global, happening May 20-22, 2024, in Dubai, UAE, focuses on blockchain technology's applications.",
    "The AI Innovations Summit, scheduled for 15-17 September 2024 in London, UK, aims at professionals and researchers advancing AI.",
    "Berlin, Germany will host the CyberSecurity World Conference between November 5th and 7th, 2024."
]

# ========== 10. ТЕСТОВЫЙ ЗАПУСК ==========
if __name__ == "__main__":
    print("\n➡️ Индексируем тестовые документы...")
    index_documents(documents)

    print("\n🔍 Выполняем поиск по запросу...")
    search_documents("Latest blockchain trends in Europe")

    print("\n🔍 Выполняем поиск и извлечение событий...")
    extractor = EventExtractor()
    results = extractor.forward("Blockchain events close to Europe")
    print(results)
