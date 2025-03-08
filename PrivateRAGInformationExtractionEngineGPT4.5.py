from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import dspy
import numpy as np
import os
import requests
import json
import time
from typing import List, Dict, Any

# Класс для создания эмбеддингов через Ollama с моделью BGE-M3
class OllamaBGEM3Embedder:
    def __init__(self, model_name='bge-m3:latest', base_url='http://localhost:11434'):
        self.model_name = model_name
        self.base_url = base_url
        self.embedding_endpoint = f"{base_url}/api/embeddings"
        self.embedding_dim = 1024

    def get_query_embedding(self, text: str) -> List[float]:
        payload = {"model": self.model_name, "prompt": text}
        response = requests.post(self.embedding_endpoint, json=payload)
        response.raise_for_status()
        data = response.json()
        return data['embedding']

# Функция для загрузки документов из файла
def load_documents_from_file(file_path="documents.txt"):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

# Milvus Retriever Module
class MilvusRM:
    def __init__(self, collection_name, embedder, host="localhost", port="19530"):
        connections.connect(host=host, port=port)
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)

        schema = CollectionSchema([
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("text", DataType.VARCHAR, max_length=65535),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=embedder.embedding_dim)
        ])

        self.collection = Collection(collection_name, schema)
        self.collection.create_index("embedding", {"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 8, "efConstruction": 64}})
        self.collection.load()
        self.embedder = embedder

    def add(self, documents: List[str]):
        embeddings = [self.embedder.get_query_embedding(doc) for doc in documents]
        entities = [{"text": doc, "embedding": emb} for doc, emb in zip(documents, embeddings)]
        self.collection.insert(entities)
        self.collection.flush()

    def __call__(self, query: str, k: int = 3):
        embedding = self.embedder.get_query_embedding(query)
        results = self.collection.search(
            data=[embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=k,
            output_fields=["text"]
        )
        class RetrieveResult:
            def __init__(self, passages):
                self.passages = passages
        return RetrieveResult([hit.entity.get("text") for hit in results[0]])

# Настройка
embedder = OllamaBGEM3Embedder()
documents = load_documents_from_file()
retriever = MilvusRM("document_parts", embedder)
retriever.add(documents)

lm = dspy.LM('ollama/gemma:2b', api_base='http://localhost:11434')
dspy.configure(lm=lm, rm=retriever)

# Определение модуля извлечения событий
class Event(dspy.Signature):
    description = dspy.InputField(desc="Описание события с названием, местом и датами")
    event_name = dspy.OutputField(desc="Название события")
    location = dspy.OutputField(desc="Место события")
    start_date = dspy.OutputField(desc="Дата начала YYYY-MM-DD")
    end_date = dspy.OutputField(desc="Дата окончания YYYY-MM-DD")

class EventExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(Event)

    def forward(self, query: str):
        results = retriever(query, k=3)
        return [self.predict(description=doc) for doc in results.passages]

# Извлечение событий
extractor = EventExtractor()
query = "Blockchain events close to Europe"
result = extractor.forward(query)

# Вывод результатов с проверкой
for event in result:
    if event.event_name and event.location and event.start_date and event.end_date:
        print({
            "name": event.event_name,
            "location": event.location,
            "start_date": event.start_date,
            "end_date": event.end_date
        })
    else:
        print("Событие не распознано или недостаточно информации.")