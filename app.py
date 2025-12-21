"""FastAPI приложение для RAG системы HIPAA регуляций."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging

from classifier import get_classifier, QuestionClassification
from embeddings import get_embedding_client
from retrievers import get_retriever_for_category

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HIPAA Regulations RAG API",
    description="API для поиска и генерации ответов по HIPAA регуляциям",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Модели запросов и ответов
class QuestionRequest(BaseModel):
    """Запрос с вопросом пользователя."""
    question: str = Field(..., description="Вопрос пользователя", min_length=1)
    max_results: Optional[int] = Field(5, ge=1, le=20, description="Максимальное количество результатов")


class RetrievedChunk(BaseModel):
    """Чанк, полученный из базы данных."""
    chunk_id: str
    section_number: str
    section_title: str
    text_raw: str
    similarity: float = Field(..., ge=0.0, le=1.0, description="Финальная оценка релевантности")
    anchor: Optional[str] = None
    chunk_kind: Optional[str] = None
    granularity: Optional[str] = None


class ClassificationResponse(BaseModel):
    """Ответ с классификацией вопроса."""
    category: str
    confidence: float
    reasoning: str


class SearchResponse(BaseModel):
    """Ответ с результатами поиска."""
    question: str
    classification: ClassificationResponse
    retrieved_chunks: List[RetrievedChunk]
    total_found: int


class AnswerResponse(BaseModel):
    """Полный ответ с генерацией."""
    question: str
    classification: ClassificationResponse
    retrieved_chunks: List[RetrievedChunk]
    answer: str
    sources: List[str]  # Список chunk_id или anchor для цитирования


# Глобальные объекты
classifier = None
embedding_client = None


@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске приложения."""
    global classifier, embedding_client
    logger.info("Инициализация приложения...")
    classifier = get_classifier()
    embedding_client = get_embedding_client()
    logger.info("✅ Приложение готово к работе")


@app.get("/")
async def root():
    """Корневой endpoint."""
    return {
        "message": "HIPAA Regulations RAG API",
        "version": "0.1.0",
        "endpoints": {
            "/classify": "Классификация вопроса",
            "/search": "Поиск релевантных чанков",
            "/answer": "Полный ответ с генерацией"
        }
    }


@app.get("/health")
async def health_check():
    """Проверка здоровья приложения."""
    return {
        "status": "healthy",
        "classifier": classifier is not None,
        "embedding_client": embedding_client is not None
    }


@app.post("/classify", response_model=ClassificationResponse)
async def classify_question(request: QuestionRequest):
    """
    Классифицирует вопрос пользователя.
    
    Определяет категорию вопроса для выбора подходящего ретривера.
    """
    try:
        logger.info(f"Классификация вопроса: {request.question[:100]}...")
        classification = classifier.classify(request.question)
        
        result = ClassificationResponse(
            category=classification.category,
            confidence=classification.confidence,
            reasoning=classification.reasoning
        )
        
        logger.info(f"Категория: {result.category} (уверенность: {result.confidence:.2%})")
        return result
        
    except Exception as e:
        logger.error(f"Ошибка классификации: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка классификации: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search_chunks(request: QuestionRequest):
    """
    Поиск релевантных чанков по вопросу.
    
    Включает классификацию вопроса и семантический поиск в базе данных.
    """
    try:
        logger.info(f"Поиск чанков для вопроса: {request.question[:100]}...")
        
        # 1. Классификация вопроса
        classification = classifier.classify(request.question)
        logger.info(f"Категория: {classification.category}")
        
        # 2. Создание эмбеддинга для вопроса
        question_embedding = embedding_client.create_embedding(request.question)
        logger.info(f"Эмбеддинг создан (размерность: {len(question_embedding)})")
        
        # 3. Поиск в базе данных с использованием ретривера
        retriever = get_retriever_for_category(
            classification.category,
            question=request.question
        )
        retrieved_chunks_raw = await retriever.retrieve(
            question_embedding=question_embedding,
            max_results=request.max_results,
            question=request.question
        )
        
        # Конвертируем в формат API
        # NavigationRetriever возвращает другую структуру (section_id вместо chunk_id, нет text_raw)
        retrieved_chunks = []
        for chunk in retrieved_chunks_raw:
            # Проверяем, это результат NavigationRetriever или обычного ретривера
            if "section_id" in chunk and "chunk_id" not in chunk:
                # NavigationRetriever результат - используем section_id как chunk_id
                retrieved_chunks.append(RetrievedChunk(
                    chunk_id=chunk["section_id"],  # Используем section_id как chunk_id
                    section_number=chunk["section_number"],
                    section_title=chunk["section_title"],
                    text_raw=f"Part {chunk.get('part', 'N/A')}, Section {chunk['section_number']}: {chunk['section_title']}",  # Формируем text_raw из метаданных
                    similarity=chunk["scores"]["final_score"],
                    anchor=chunk.get("anchor"),
                    chunk_kind=None
                ))
            else:
                # Обычный ретривер
                retrieved_chunks.append(RetrievedChunk(
                    chunk_id=chunk["chunk_id"],
                    section_number=chunk["section_number"],
                    section_title=chunk["section_title"],
                    text_raw=chunk["text_raw"],
                    similarity=chunk["scores"]["final_score"],
                    anchor=chunk.get("anchor"),
                    chunk_kind=chunk.get("chunk_kind")
                ))
        
        logger.info(f"Найдено чанков: {len(retrieved_chunks)}")
        
        return SearchResponse(
            question=request.question,
            classification=ClassificationResponse(
                category=classification.category,
                confidence=classification.confidence,
                reasoning=classification.reasoning
            ),
            retrieved_chunks=retrieved_chunks,
            total_found=len(retrieved_chunks)
        )
        
    except Exception as e:
        logger.error(f"Ошибка поиска: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка поиска: {str(e)}")


@app.post("/answer", response_model=AnswerResponse)
async def generate_answer(request: QuestionRequest):
    """
    Генерация полного ответа на вопрос.
    
    Включает классификацию, поиск релевантных чанков и генерацию ответа.
    """
    try:
        logger.info(f"Генерация ответа на вопрос: {request.question[:100]}...")
        
        # 1. Классификация вопроса
        classification = classifier.classify(request.question)
        logger.info(f"Категория: {classification.category}")
        
        # 2. Создание эмбеддинга для вопроса
        question_embedding = embedding_client.create_embedding(request.question)
        
        # 3. Поиск релевантных чанков с использованием ретривера
        retriever = get_retriever_for_category(
            classification.category,
            question=request.question
        )
        retrieved_chunks_raw = await retriever.retrieve(
            question_embedding=question_embedding,
            max_results=request.max_results,
            question=request.question
        )
        
        # Конвертируем в формат API
        # NavigationRetriever возвращает другую структуру (section_id вместо chunk_id, нет text_raw)
        retrieved_chunks = []
        for chunk in retrieved_chunks_raw:
            # Проверяем, это результат NavigationRetriever или обычного ретривера
            if "section_id" in chunk and "chunk_id" not in chunk:
                # NavigationRetriever результат - используем section_id как chunk_id
                retrieved_chunks.append(RetrievedChunk(
                    chunk_id=chunk["section_id"],  # Используем section_id как chunk_id
                    section_number=chunk["section_number"],
                    section_title=chunk["section_title"],
                    text_raw=f"Part {chunk.get('part', 'N/A')}, Section {chunk['section_number']}: {chunk['section_title']}",  # Формируем text_raw из метаданных
                    similarity=chunk["scores"]["final_score"],
                    anchor=chunk.get("anchor"),
                    chunk_kind=None
                ))
            else:
                # Обычный ретривер
                retrieved_chunks.append(RetrievedChunk(
                    chunk_id=chunk["chunk_id"],
                    section_number=chunk["section_number"],
                    section_title=chunk["section_title"],
                    text_raw=chunk["text_raw"],
                    similarity=chunk["scores"]["final_score"],
                    anchor=chunk.get("anchor"),
                    chunk_kind=chunk.get("chunk_kind")
                ))
        
        logger.info(f"Найдено чанков: {len(retrieved_chunks)}")
        
        # 4. Генерация ответа
        from generator import get_generator
        generator = get_generator()
        
        # Конвертируем chunks в формат для генератора
        chunks_for_generator = [
            {
                "chunk_id": chunk.chunk_id,
                "section_number": chunk.section_number,
                "section_title": chunk.section_title,
                "text_raw": chunk.text_raw,
                "anchor": chunk.anchor
            }
            for chunk in retrieved_chunks
        ]
        
        answer = await generator.generate(
            question=request.question,
            chunks=chunks_for_generator,
            classification=classification
        )
        
        # 5. Формирование списка источников
        sources = [chunk.chunk_id for chunk in retrieved_chunks[:3]]  # Топ-3 источника
        
        return AnswerResponse(
            question=request.question,
            classification=ClassificationResponse(
                category=classification.category,
                confidence=classification.confidence,
                reasoning=classification.reasoning
            ),
            retrieved_chunks=retrieved_chunks,
            answer=answer,
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Ошибка генерации ответа: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка генерации ответа: {str(e)}")




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
