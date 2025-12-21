"""Основной класс генератора ответов."""

from typing import List, Optional, Dict, Any
import logging

from classifier import QuestionClassification

from .base import GenerationResult, ContextItem, Citation
from .policy import AnswerPolicy, PermissionPolicy, choose_policy, determine_permission_policy
from .context_builder import build_context
from .llm_client import LLMClient
from .citation_validator import parse_and_validate_citations
from .prompts import (
    build_strict_citation_prompt,
    build_navigation_prompt,
    build_summary_prompt,
    build_quoted_answer_prompt,
    build_listing_prompt,
)

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Генератор ответов с использованием LLM."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Инициализация генератора ответов.
        
        Args:
            api_key: API ключ (если не указан, LLMClient берет из переменных окружения)
            base_url: Базовый URL API (если не указан, LLMClient берет из переменных окружения)
            model: Модель для генерации (если не указана, LLMClient берет из переменных окружения)
        """
        # LLMClient сам обрабатывает все переменные окружения
        self.llm_client = LLMClient(api_key=api_key, base_url=base_url, model=model)
    
    def _generate_strict_citation_answer(
        self,
        context_items: List[ContextItem],
        question: str
    ) -> GenerationResult:
        """
        Генерирует ответ для STRICT_CITATION policy без использования LLM.
        
        Возвращает чистые цитаты в формате bullet points: anchor - text_raw
        
        Args:
            context_items: Элементы контекста
            question: Вопрос пользователя (для метаданных)
        
        Returns:
            GenerationResult с ответом в виде цитат
        """
        if not context_items:
            return GenerationResult(
                answer_text="No relevant citations found in the regulations.",
                citations=[],
                policy="",
                meta={"error": "no_context_items", "policy": "strict_citation"}
            )
        
        # Формируем ответ в виде bullet points: anchor - text_raw
        answer_lines = []
        citations = []
        
        for item in context_items:
            if item.anchor and item.text_raw:
                # Формат: anchor - text_raw
                answer_lines.append(f"{item.anchor} - {item.text_raw}")
                
                # Создаем citation
                citations.append(Citation(
                    anchor=item.anchor,
                    quote=item.text_raw,
                    chunk_id=item.chunk_id
                ))
        
        answer_text = "\n".join(answer_lines) if answer_lines else "No citations with anchors found."
        
        # Метаданные
        meta = {
            "policy": "strict_citation",
            "llm_skipped": True,
            "citations_count": len(citations),
            "context_items_count": len(context_items),
        }
        
        return GenerationResult(
            answer_text=answer_text,
            citations=citations,
            policy="",
            meta=meta
        )
    
    def _extract_citations(self, answer_text: str, context_items: List[ContextItem]) -> List[Citation]:
        """
        Извлекает цитаты из ответа (базовая реализация).
        
        Пока просто создает цитаты на основе найденных чанков с anchor.
        В следующих шагах будет улучшено для извлечения цитат из текста ответа.
        
        Args:
            answer_text: Текст ответа
            context_items: Элементы контекста
        
        Returns:
            Список цитат
        """
        citations = []
        
        # Пока создаем цитаты на основе топ-чанков с anchor
        for item in context_items[:3]:  # Топ-3 чанка
            if item.anchor:
                citations.append(Citation(
                    anchor=item.anchor,
                    quote=item.text_raw[:200] + "..." if len(item.text_raw) > 200 else item.text_raw,
                    chunk_id=item.chunk_id
                ))
        
        return citations
    
    def _get_prompt_builder(self, policy: AnswerPolicy):
        """
        Возвращает функцию построения промпта для политики.
        
        Args:
            policy: Политика генерации ответа
        
        Returns:
            Функция build_prompt(question, context, meta) -> (system, user, temperature, max_tokens)
        """
        prompt_builders = {
            AnswerPolicy.STRICT_CITATION: build_strict_citation_prompt,
            AnswerPolicy.NAVIGATION: build_navigation_prompt,
            AnswerPolicy.SUMMARY: build_summary_prompt,
            AnswerPolicy.QUOTED_ANSWER: build_quoted_answer_prompt,
            AnswerPolicy.LISTING: build_listing_prompt,
        }
        
        return prompt_builders.get(policy, build_summary_prompt)
    
    async def generate(
        self,
        question: str,
        chunks: List[dict],
        classification: QuestionClassification,
        retriever_signals: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        """
        Генерирует ответ на основе найденных чанков.
        
        Args:
            question: Вопрос пользователя
            chunks: Список найденных релевантных чанков
            classification: Классификация вопроса
            retriever_signals: Сигналы от ретривера (policy_signal, yesno_signal и т.д.)
        
        Returns:
            GenerationResult со структурированным ответом, цитатами и метаданными
        """
        if not chunks:
            return GenerationResult(
                answer_text="No relevant information found in HIPAA regulations database.",
                citations=[],
                policy="",
                meta={"error": "no_chunks_found"}
            )
        
        # Выбираем политику генерации ответа (нужна для построения контекста)
        answer_policy = choose_policy(
            category=classification.category,
            classification_confidence=classification.confidence,
            signals=retriever_signals or {},
            question=question
        )
        
        # Преобразуем чанки в элементы контекста с учетом политики
        context_items = build_context(chunks, answer_policy)
        
        # Определяем политику разрешения/запрета (для permission/disclosure вопросов)
        permission_policy_enum = determine_permission_policy(classification.category, retriever_signals or {})
        permission_policy = permission_policy_enum.value if permission_policy_enum != PermissionPolicy.NONE else ""
        
        # Подготавливаем метаданные для промпта (включая сигналы от ретривера)
        prompt_meta = {
            "category": classification.category,
            "confidence": classification.confidence,
            "permission_policy": permission_policy,
        }
        
        # Добавляем сигналы от ретривера в meta для промпта
        if retriever_signals:
            if "yesno_signal" in retriever_signals:
                prompt_meta["yesno_signal"] = retriever_signals["yesno_signal"]
                prompt_meta["yesno_rationale"] = retriever_signals.get("yesno_rationale", "")
                logger.info(f"Using retriever yesno_signal: {prompt_meta['yesno_signal']}")
            
            if "policy_signal" in retriever_signals:
                prompt_meta["policy_signal"] = retriever_signals["policy_signal"]
                logger.info(f"Using retriever policy_signal: {prompt_meta['policy_signal']}")
        
        # Специальная обработка для STRICT_CITATION - без LLM
        if answer_policy == AnswerPolicy.STRICT_CITATION:
            logger.info("LLM skipped for STRICT_CITATION policy - returning raw citations")
            return self._generate_strict_citation_answer(context_items, question)
        
        # Выбираем функцию построения промпта по политике
        prompt_builder_func = self._get_prompt_builder(answer_policy)
        logger.info(f"Using prompt template: {prompt_builder_func.__name__} for policy: {answer_policy.value}")
        
        # Строим промпт
        system_message, user_prompt, temperature, max_tokens = prompt_builder_func(
            question=question,
            context=context_items,
            meta=prompt_meta
        )
        
        # Генерируем ответ через LLM клиент
        llm_response = await self.llm_client.complete(
            system=system_message,
            user=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Для QUOTED_ANSWER и LISTING парсим JSON и валидируем citations
        if answer_policy in (AnswerPolicy.QUOTED_ANSWER, AnswerPolicy.LISTING):
            # Определяем, обязательны ли citations
            require_citations = classification.category in ("definition", "regulatory_principle", "procedural / best practices")
            
            answer_text, citations = parse_and_validate_citations(
                llm_response=llm_response,
                context_items=context_items,
                require_citations=require_citations,
                auto_fix_quote=True  # Явно включаем auto-fix для quote при валидном anchor
            )
        else:
            # Для остальных политик используем старый метод
            answer_text = llm_response
            citations = self._extract_citations(answer_text, context_items)
        
        # Метаданные для отладки
        meta = {
            "model": self.llm_client.model,
            "chunks_count": len(chunks),
            "category": classification.category,
            "confidence": classification.confidence,
            "prompt_template": prompt_builder_func.__name__,
            "answer_policy": answer_policy.value,
            "permission_policy": permission_policy,
            "citations_validated": answer_policy in (AnswerPolicy.QUOTED_ANSWER, AnswerPolicy.LISTING),
            "valid_citations_count": len(citations),
        }
        
        # Добавляем сигналы от ретривера в финальные метаданные
        if retriever_signals:
            if "yesno_signal" in retriever_signals:
                meta["retriever_yesno_signal"] = retriever_signals["yesno_signal"]
                meta["retriever_yesno_rationale"] = retriever_signals.get("yesno_rationale", "")
            if "policy_signal" in retriever_signals:
                meta["retriever_policy_signal"] = retriever_signals["policy_signal"]
        
        return GenerationResult(
            answer_text=answer_text,
            citations=citations,
            policy=permission_policy,  # Сохраняем permission_policy в поле policy
            meta=meta
        )
