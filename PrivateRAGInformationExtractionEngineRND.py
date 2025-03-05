from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import dspy
import numpy as np
import os
from typing import List, Dict, Any, Optional, Union


# Функция для загрузки документов из файла
def load_documents_from_file(file_path="documents.txt"):
    try:
        print(f"Загрузка документов из файла {file_path}")
        if not os.path.exists(file_path):
            print(f"Файл {file_path} не найден!")
            return []
        with open(file_path, 'r', encoding='utf-8') as f:
            documents = [line.strip() for line in f if line.strip()]
        print(f"Загружено {len(documents)} документов")
        return documents
    except Exception as e:
        print(f"Ошибка при загрузке документов: {e}")
        return []


# Функция для преобразования текста в эмбеддинги (заглушка)
def get_embeddings(texts, dim=1024):
    """Генерирует заглушку-эмбеддинг для текста заданной размерности"""
    return [np.random.rand(dim).tolist() for _ in texts]


# Создаем класс MilvusRM для интеграции Milvus с DSPy
class MilvusRM:
    def __init__(
            self,
            milvus_collection_name: str,
            milvus_host: str = "localhost",
            milvus_port: str = "19530",
            embedding_dim: int = 1024,
            metadata_field: str = "document",
            embedding_field: str = "embedding",
            id_field: str = "id",
            text_field: str = "text",
            recreate_collection: bool = True,
    ):
        """Конструктор для создания Milvus Retriever Module."""
        self.milvus_collection_name = milvus_collection_name
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.embedding_dim = embedding_dim
        self.metadata_field = metadata_field
        self.embedding_field = embedding_field
        self.id_field = id_field
        self.text_field = text_field
        self.recreate_collection = recreate_collection

        # Подключение к Milvus
        connections.connect(host=milvus_host, port=milvus_port)

        # Принудительно удаляем и пересоздаем коллекцию
        self._recreate_collection()

        # Получаем коллекцию
        self.collection = Collection(name=milvus_collection_name)
        self.collection.load()

    def _recreate_collection(self):
        """Удаление и пересоздание коллекции"""
        # Проверяем и удаляем существующую коллекцию
        try:
            if utility.has_collection(self.milvus_collection_name):
                print(f"Удаление существующей коллекции {self.milvus_collection_name}")
                utility.drop_collection(self.milvus_collection_name)
                print(f"Коллекция {self.milvus_collection_name} успешно удалена")
        except Exception as e:
            print(f"Ошибка при удалении коллекции: {e}")

        # Создаем новую коллекцию
        try:
            print(f"Создание коллекции {self.milvus_collection_name} с размерностью {self.embedding_dim}")

            # Определение схемы коллекции
            fields = [
                FieldSchema(name=self.id_field, dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name=self.text_field, dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name=self.metadata_field, dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name=self.embedding_field, dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
            ]
            schema = CollectionSchema(fields=fields)

            # Создание коллекции
            collection = Collection(name=self.milvus_collection_name, schema=schema)
            print(f"Коллекция {self.milvus_collection_name} успешно создана")

            # Создание индекса для быстрого поиска
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 8, "efConstruction": 64}
            }
            collection.create_index(field_name=self.embedding_field, index_params=index_params)
            print(f"Индекс создан для поля {self.embedding_field}")

        except Exception as e:
            print(f"Ошибка при создании коллекции: {e}")
            raise e

    def add(self, documents: List[str], embeddings: List[List[float]]):
        """Добавление документов и их эмбеддингов в Milvus."""
        if not documents or not embeddings:
            return

        # Проверка размерности
        if len(embeddings[0]) != self.embedding_dim:
            raise ValueError(
                f"Размерность эмбеддингов {len(embeddings[0])} не соответствует размерности коллекции {self.embedding_dim}")

        # Формируем данные для вставки в правильном формате для Milvus
        entities = []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            entity = {
                self.text_field: doc,
                self.metadata_field: doc,  # В данном случае используем сам документ как метаданные
                self.embedding_field: emb
            }
            entities.append(entity)

        self.collection.insert(entities)
        self.collection.flush()  # Гарантирует запись данных на диск
        print(f"Добавлены документы в коллекцию, размерность векторов: {self.embedding_dim}")

    def search(self, query_embedding: List[float], k: int = 3) -> List[Dict[str, Any]]:
        """Поиск ближайших документов к запросу."""
        # Проверка размерности запроса
        if len(query_embedding) != self.embedding_dim:
            raise ValueError(
                f"Размерность запроса {len(query_embedding)} не соответствует размерности коллекции {self.embedding_dim}")

        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}

        results = self.collection.search(
            data=[query_embedding],
            anns_field=self.embedding_field,
            param=search_params,
            limit=k,
            output_fields=[self.text_field]
        )

        retrieved = []
        for hits in results:
            for hit in hits:
                retrieved.append({
                    "score": hit.score,
                    "text": hit.entity.get(self.text_field),
                    "metadata": {}
                })

        return retrieved

    def __call__(self, query: str, k: int = 3):
        """Метод для интеграции с DSPy."""
        # Генерация эмбеддинга запроса с правильной размерностью
        query_embedding = np.random.rand(self.embedding_dim).tolist()
        print(f"Создан эмбеддинг запроса размерности {len(query_embedding)}")

        try:
            results = self.search(query_embedding, k)

            # Формат, ожидаемый DSPy
            class RetrieveResult:
                def __init__(self, passages):
                    self.passages = passages

            if not results:
                print("Поиск не дал результатов")
                return RetrieveResult([])

            return RetrieveResult([r["text"] for r in results])
        except Exception as e:
            print(f"Ошибка при поиске: {e}")
            # В случае ошибки возвращаем пустой результат или документы-заглушки
            print("Возвращаем заглушку вместо результатов поиска")
            # Используем первые k документов как заглушку
            docs = load_documents_from_file()
            return RetrieveResult(docs[:k] if docs and len(docs) >= k else [])


# === ОСНОВНОЙ КОД ===

# Загружаем документы из файла
documents = load_documents_from_file("documents.txt")

# Принудительное пересоздание Milvus коллекции
milvus_retriever = MilvusRM(
    milvus_collection_name="document_parts",
    milvus_host="localhost",
    milvus_port="19530",
    embedding_dim=1024,
    recreate_collection=True
)

# Всегда загружаем документы заново
print("\n=== ДОБАВЛЕНИЕ ДОКУМЕНТОВ В MILVUS ===")
document_embeddings = get_embeddings(documents, dim=1024)
print(f"Созданы эмбеддинги размерности {len(document_embeddings[0])}")
milvus_retriever.add(documents, document_embeddings)
print(f"Добавлено {len(documents)} документов в Milvus")

# Настройка языковой модели
gemma_model = dspy.LM('ollama/gemma:2b',
                      api_base='http://localhost:11434',
                      api_key='')

# Настройка DSPy
dspy.configure(lm=gemma_model, rm=milvus_retriever)


# Определение класса для извлечения событий
class Event(dspy.Signature):
    description = dspy.InputField(
        desc="Textual description of the event, including name, location and dates"
    )
    event_name = dspy.OutputField(desc="Name of the event")
    location = dspy.OutputField(desc="Location of the event")
    start_date = dspy.OutputField(desc="Start date of the event, YYYY-MM-DD")
    end_date = dspy.OutputField(desc="End date of the event, YYYY-MM-DD")


# Модуль для извлечения событий
class EventExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        # Используем наш MilvusRM напрямую
        self.retriever = milvus_retriever
        # Predict модуль для извлечения информации
        self.predict = dspy.Predict(Event)

    def forward(self, query: str):
        # Получение релевантных документов
        results = self.retriever(query, k=3)

        # Извлечение информации о событиях
        events = []
        for document in results.passages:
            print(f"Анализ документа: {document[:100]}...")
            event = self.predict(description=document)
            events.append(event)

        return events


# Инициализация и выполнение запроса
extractor = EventExtractor()
query = "Blockchain events close to Europe"
print(f"\nПоиск: {query}")
result = extractor.forward(query)

# Вывод результатов
print("\nИзвлеченные события:")
for i, event in enumerate(result, 1):
    print(f"\nСобытие {i}:")
    print(f"Название: {event.event_name}")
    print(f"Место: {event.location}")
    print(f"Дата начала: {event.start_date}")
    print(f"Дата окончания: {event.end_date}")