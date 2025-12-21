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
    sources: List[str]  # Список anchor или chunk_id для цитирования (из citations)
    debug: Optional[Dict[str, Any]] = None  # Опциональные метаданные для отладки


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
        logger.info(f"Категория (before override): {classification.category}")
        
        # Применение post-classification override
        from classification_override import apply_classification_override
        classification = apply_classification_override(classification, request.question)
        
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
        initial_classification = classifier.classify(request.question)
        logger.info(f"Категория (before override): {initial_classification.category}")
        
        # 1.1. Применение post-classification override (доменные правила)
        from classification_override import apply_classification_override
        classification = apply_classification_override(initial_classification, request.question)
        if classification.category != initial_classification.category:
            logger.info(f"Категория (after override): {classification.category} (was: {initial_classification.category})")
        
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
            question=request.question,
            category=classification.category  # Передаем категорию для поддержки regulatory_principle
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
        
        # Извлекаем сигналы из результатов ретривера
        retriever_signals = {}
        
        # Для procedural: yesno_signal и yesno_rationale из первого элемента
        if classification.category == "procedural / best practices" and retrieved_chunks_raw:
            first_chunk = retrieved_chunks_raw[0]
            if "yesno_signal" in first_chunk:
                retriever_signals["yesno_signal"] = first_chunk.get("yesno_signal")
                retriever_signals["yesno_rationale"] = first_chunk.get("yesno_rationale", "")
                logger.info(f"Retriever yesno_signal: {retriever_signals['yesno_signal']}")
        
        # Для disclosure: policy_signal из любого элемента (обычно одинаковый)
        if classification.category == "permission / disclosure" and retrieved_chunks_raw:
            for chunk in retrieved_chunks_raw:
                if "policy_signal" in chunk:
                    retriever_signals["policy_signal"] = chunk.get("policy_signal")
                    logger.info(f"Retriever policy_signal: {retriever_signals['policy_signal']}")
                    break
        
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
        
        generation_result = await generator.generate(
            question=request.question,
            chunks=chunks_for_generator,
            classification=classification,
            retriever_signals=retriever_signals
        )
        
        # 5. Формирование списка источников из citations
        # Приоритет: anchor, fallback на chunk_id
        sources = []
        if generation_result.citations:
            # Используем citations из генератора (валидированные)
            for citation in generation_result.citations:
                if citation.anchor:
                    sources.append(citation.anchor)
                elif citation.chunk_id:
                    sources.append(citation.chunk_id)
        else:
            # Fallback: если citations нет, используем топ-3 chunk_id из retrieved_chunks
            sources = [chunk.chunk_id for chunk in retrieved_chunks[:3]]
            logger.warning("No citations in generation result, using fallback sources from retrieved_chunks")
        
        # Убираем дубликаты, сохраняя порядок
        seen = set()
        unique_sources = []
        for source in sources:
            if source not in seen:
                seen.add(source)
                unique_sources.append(source)
        sources = unique_sources
        
        # Формируем debug информацию (опционально)
        debug_info = None
        if generation_result.meta:
            debug_info = generation_result.meta
        
        return AnswerResponse(
            question=request.question,
            classification=ClassificationResponse(
                category=classification.category,
                confidence=classification.confidence,
                reasoning=classification.reasoning
            ),
            retrieved_chunks=retrieved_chunks,
            answer=generation_result.answer_text,  # Используем answer_text напрямую
            sources=sources,
            debug=debug_info
        )
        
    except Exception as e:
        logger.error(f"Ошибка генерации ответа: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка генерации ответа: {str(e)}")




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
