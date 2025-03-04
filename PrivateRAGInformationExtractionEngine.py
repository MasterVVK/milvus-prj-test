from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
import dspy
from dspy.retrieve.milvus_rm import MilvusRM
import numpy as np

# Подключение к Milvus
client = MilvusClient(uri="http://localhost:19530")

# Создание коллекции
collection_name = "document_parts"

# Определяем поля
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
    FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=65535),
]

# Проверяем, существует ли коллекция, и удаляем её
if client.has_collection(collection_name):
    client.drop_collection(collection_name)

# Создаём схему коллекции
schema = CollectionSchema(fields, description="Коллекция для хранения документов")

# Создаём коллекцию с правильной схемой
client.create_collection(
    collection_name=collection_name,
    schema=schema,
)

# Загрузка документов
documents = [
    "Taking place in San Francisco, USA, from the 10th to the 12th of June, 2024, the Global Developers Conference...",
    "The AI Innovations Summit, scheduled for 15-17 September 2024 in London, UK...",
    "Berlin, Germany will host the CyberSecurity World Conference between November 5th and 7th, 2024...",
]

# Преобразуем данные в float32
embeddings = np.array([[0.1] * 1536] * len(documents), dtype=np.float32)  # float32 для Milvus

# Подготовка данных для вставки
data = [
    {"embedding": emb, "document": doc}
    for emb, doc in zip(embeddings, documents)
]

# Вставляем данные в Milvus
client.insert(
    collection_name=collection_name,
    data=data,
)

# Используем новый способ подключения Ollama
gemma_model = dspy.LM(
    model="gemma:2b",
    api_base="http://localhost:11434",
    api_key=""
)

# Подключение нового ретривера
milvus_retriever = MilvusRM(
    collection_name=collection_name,
    client=client,
)

dspy.configure(lm=gemma_model, rm=milvus_retriever)

# Определяем класс Event
class Event(dspy.Signature):
    description = dspy.InputField(desc="Textual description of the event, including name, location and dates")
    event_name = dspy.OutputField(desc="Name of the event")
    location = dspy.OutputField(desc="Location of the event")
    start_date = dspy.OutputField(desc="Start date of the event, YYYY-MM-DD")
    end_date = dspy.OutputField(desc="End date of the event, YYYY-MM-DD")

# Определяем класс EventExtractor
class EventExtractor(dspy.Module):

    def __init__(self):
        super().__init__()
        self.retriever = dspy.Retrieve(k=3)  # Используем Milvus
        self.predict = dspy.Predict(Event)

    def forward(self, query: str):
        results = self.retriever.forward(query)

        events = []
        for document in results.passages:
            event = self.predict(description=document)
            events.append(event)

        return events

# Запуск модели
extractor = EventExtractor()
result = extractor.forward("Blockchain events close to Europe")
print(result)
