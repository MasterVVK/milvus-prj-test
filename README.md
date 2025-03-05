# PrivateRAG: Извлечение информации о событиях с использованием Milvus и Ollama

Система приватного Retrieval-Augmented Generation (RAG) для поиска и извлечения структурированной информации о событиях из текстовых документов. Использует векторную базу данных Milvus и локальные модели через Ollama для полностью автономной работы без необходимости подключения к облачным API.

##  Обзор

PrivateRAG объединяет следующие компоненты:

1. **Векторное хранилище Milvus**: быстрый, масштабируемый движок для семантического поиска
2. **Модель эмбеддингов BGE-M3**: создает высококачественные векторные представления текста
3. **Языковая модель Gemma 2B**: извлекает структурированную информацию из найденных текстов
4. **DSPy**: фреймворк для декларативной разработки приложений с использованием больших языковых моделей

Система работает полностью локально, защищая конфиденциальность данных и не требуя подключения к интернету для выполнения запросов.

## ✨ Особенности

- **Высококачественный семантический поиск**: использует модель BGE-M3 для создания контекстно-зависимых векторных эмбеддингов
- **Структурированное извлечение информации**: преобразует неструктурированный текст в структурированные данные о событиях
- **Полностью локальное выполнение**: все компоненты работают на локальной машине без передачи данных внешним сервисам
- **Масштабируемость**: Milvus позволяет эффективно работать с большими объемами документов
- **Поддержка автоматической загрузки моделей**: автоматически проверяет и загружает необходимые модели Ollama

## ️ Архитектура системы

Система состоит из следующих компонентов:

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│   Документы │───>│ BGE-M3 модель│───>│    Milvus   │<───│   Запрос    │
│  (из файла) │    │ (эмбеддинги) │    │  (хранение) │    │             │
└─────────────┘    └──────────────┘    └──────┬──────┘    └─────┬───────┘
                                              │                 │
                                              │                 │
                                              v                 v
                                       ┌──────────────────────────────┐
                                       │  Поиск релевантных документов │
                                       └─────────────┬─────────────────┘
                                                     │
                                                     v
                                              ┌─────────────┐
                                              │  Gemma 2B   │
                                              │  (анализ)   │
                                              └─────┬───────┘
                                                    │
                                                    v
                                       ┌──────────────────────────────┐
                                       │  Структурированная информация │
                                       │         о событиях            │
                                       └──────────────────────────────┘
```

##  Предварительные требования

- **Python 3.8+**
- **Docker** (для запуска Milvus)
- **Ollama** ([установка Ollama](https://github.com/ollama/ollama))
- **DSPy** (библиотека для программирования больших языковых моделей)
- **pymilvus** (Python-клиент для Milvus)
- **16 ГБ ОЗУ** (рекомендуется для работы всех компонентов)

## ️ Установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/MasterVVK/private-rag-milvus-ollama.git
cd private-rag-milvus-ollama
```

### 2. Создание виртуального окружения Python

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# или
.venv\Scripts\activate.bat  # Windows
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Запуск Milvus через Docker

```bash
docker run -d --name milvus_standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest standalone
```

### 5. Установка и запуск Ollama

Установите Ollama, следуя инструкциям с [официального сайта](https://github.com/ollama/ollama).

Запустите Ollama:
```bash
ollama serve
```

### 6. Загрузка моделей для Ollama

```bash
# Загрузка модели для эмбеддингов
ollama pull bge-m3:latest

# Загрузка языковой модели для анализа
ollama pull gemma:2b
```

##  Файлы и их назначение

- **PrivateRAGInformationExtractionEngine.py**: основной файл системы, содержащий логику RAG
- **requirements.txt**: список зависимостей Python
- **documents.txt**: пример файла с документами для обработки

##  Использование

### Подготовка данных

Создайте текстовый файл `documents.txt`, где каждая строка содержит отдельный документ:

```
"The Blockchain Expo Global, happening May 20-22, 2024, in Dubai, UAE, focuses on blockchain technology's applications, opportunities, and challenges for entrepreneurs, developers, and investors."
"Berlin, Germany will host the CyberSecurity World Conference between November 5th and 7th, 2024, serving as a key forum for cybersecurity professionals to exchange strategies and research on threat detection and mitigation."
...
```

### Запуск системы

```bash
python PrivateRAGInformationExtractionEngine.py
```

### Пример выполнения запроса

По умолчанию система выполнит поиск по запросу "Blockchain events close to Europe". Чтобы изменить запрос, отредактируйте переменную `query` в файле PrivateRAGInformationExtractionEngine.py:

```python
query = "AI conferences in North America"  # Измените на свой запрос
```

### Пример вывода системы

```
Загрузка документов из файла documents.txt
Загружено 30 документов
Проверка модели bge-m3:latest для эмбеддингов через Ollama
Модель bge-m3:latest уже доступна
Модель bge-m3:latest готова. Размерность эмбеддингов: 1024
Удаление существующей коллекции document_parts
Коллекция document_parts успешно удалена
Создание коллекции document_parts с размерностью 1024
Коллекция document_parts успешно создана
Индекс создан для поля embedding
=== ДОБАВЛЕНИЕ ДОКУМЕНТОВ В MILVUS ===
Создание эмбеддингов для документов через BGE-M3...
Обработка эмбеддингов: 0/30 до 10/30
Обработка эмбеддингов: 10/30 до 20/30
Обработка эмбеддингов: 20/30 до 30/30
Созданы эмбеддинги размерности 1024
Добавлены документы в коллекцию, размерность векторов: 1024
Добавлено 30 документов в Milvus
Поиск: Blockchain events close to Europe
Создание эмбеддинга для запроса: 'Blockchain events close to Europe'
Анализ документа: "The Blockchain Expo Global, happening May 20-22, 2024, in Dubai, UAE, focuses on blockchain technol...
Анализ документа: "Blockchain for Business Summit, happening in Singapore from 2024-05-02 to 2024-05-04, focuses on bl...
Анализ документа: "Scheduled for May 5-7, 2024, in London, UK, the Fintech Leaders Forum brings together experts to di...
Извлеченные события:
Событие 1:
Название: The Blockchain Expo Global
Место: Dubai, UAE
Дата начала: 2024-05-20
Дата окончания: 2024-05-22
Событие 2:
Название: Blockchain for Business Summit
Место: Singapore
Дата начала: 2024-05-02
Дата окончания: 2024-05-04
Событие 3:
Название: The Fintech Leaders Forum
Место: London, UK
Дата начала: 2024-05-05
Дата окончания: 2024-05-07
```

## 易 DSPy: Фреймворк для программирования ЛЛМ

[DSPy](https://github.com/stanfordnlp/dspy) - это декларативный фреймворк для разработки приложений с использованием больших языковых моделей (ЛЛМ). В нашем проекте DSPy используется для:

1. **Структурирования информации** через `dspy.Signature` и `dspy.InputField`/`dspy.OutputField`
2. **Интеграции языковых моделей** через `dspy.LM`
3. **Интеграции системы поиска** через систему Retrieval Module (RM)
4. **Создания конвейеров обработки данных** с помощью `dspy.Module`

### Основные компоненты DSPy в проекте

```python
# Определение структуры данных
class Event(dspy.Signature):
    description = dspy.InputField(...)
    event_name = dspy.OutputField(...)
    location = dspy.OutputField(...)
    # ...

# Подключение языковой модели
gemma_model = dspy.LM('ollama/gemma:2b', ...)

# Настройка DSPy с указанием ЛЛМ и модуля поиска
dspy.configure(lm=gemma_model, rm=milvus_retriever)

# Создание RAG модуля
class EventExtractor(dspy.Module):
    def __init__(self):
        self.retriever = milvus_retriever
        self.predict = dspy.Predict(Event)
        
    def forward(self, query):
        # Поиск + генерация
        # ...
```

### Возможности DSPy

- **Промптинг через типы данных**: вместо написания промптов вы определяете структуры данных
- **Модульность**: легко комбинировать и заменять компоненты системы
- **Оптимизация через демонстрации**: возможность обучения через примеры (few-shot learning)
- **Декларативный подход**: описываете что вы хотите получить, а не как это сделать

### Расширение DSPy-компонентов

Вы можете создавать собственные модули DSPy для более сложных задач:

```python
class ComplexEventExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retriever = milvus_retriever
        self.classifier = dspy.Predict(EventType)  # Определяет тип события
        self.extractor = dspy.Predict(Event)       # Извлекает информацию
        
    def forward(self, query):
        results = self.retriever(query)
        events = []
        
        for doc in results.passages:
            event_type = self.classifier(text=doc)
            event = self.extractor(description=doc, type=event_type.category)
            events.append(event)
            
        return events
```

Для более подробной информации о DSPy посетите [официальную документацию](https://github.com/stanfordnlp/dspy).

##  Расширение и кастомизация

### Использование других моделей для эмбеддингов

Измените имя модели в коде:

```python
embedder = OllamaBGEM3Embedder(model_name='другая-модель', base_url='http://localhost:11434')
```

### Использование другой языковой модели для анализа

```python
gemma_model = dspy.LM('ollama/другая-модель',
                      api_base='http://localhost:11434',
                      api_key='')
```

### Настройка схемы извлечения информации

Вы можете изменить структуру извлекаемой информации, модифицировав класс `Event`:

```python
class Event(dspy.Signature):
    description = dspy.InputField(desc="...")
    # Добавьте дополнительные поля
    organizer = dspy.OutputField(desc="Organizer of the event")
    website = dspy.OutputField(desc="Event website URL")
    # ...
```

## ️ Устранение неполадок

### Проблема: "Client error '404 Not Found' for url 'http://localhost:11434/api/chat'"

**Решение**: Убедитесь, что Ollama запущен командой `ollama serve`.

### Проблема: "vector dimension mismatch"

**Решение**: Проверьте, что размерность векторов в коллекции совпадает с размерностью эмбеддингов модели:

```python
# Убедитесь, что эти значения совпадают
milvus_retriever = MilvusRM(
    ...
    embedding_dim=1024,  # Должно соответствовать размерности модели
    ...
)
```

### Проблема: "field document not exist"

**Решение**: Убедитесь, что имена полей в поиске соответствуют созданной схеме.

##  Цитирование

Если вы используете этот проект в своих исследованиях или коммерческих решениях, пожалуйста, приведите ссылку:

```
@misc{privaterag2025,
  author = {Your Name},
  title = {PrivateRAG: Retrieval-Augmented Generation with Milvus and Ollama},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/private-rag-milvus-ollama}
}
```

##  Лицензия

Этот проект распространяется под лицензией MIT. См. файл LICENSE для получения подробной информации.

##  Благодарности

- [Milvus](https://milvus.io/) - за создание отличной векторной базы данных
- [Ollama](https://github.com/ollama/ollama) - за предоставление удобного способа запуска моделей локально
- [BAAI](https://www.baai.ac.cn/) - за создание модели BGE-M3
- [DSPy](https://github.com/stanfordnlp/dspy) - за удобный фреймворк для создания RAG-систем
