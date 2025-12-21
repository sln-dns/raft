"""Пакет ретриверов для поиска релевантных чанков."""

# Импортируем базовые классы из нового модуля
from .base import BaseRetriever, CandidateChunk, NavigationHit

# Импортируем функцию выбора ретривера из registry
from .registry import get_retriever_for_category

# Импортируем все ретриверы из новых модулей
from .overview_purpose import OverviewPurposeRetriever
from .navigation import NavigationRetriever
from .definition import DefinitionRetriever
from .scope import ScopeRetriever
from .penalties import PenaltiesRetriever
from .procedural import ProceduralRetriever
from .permission_disclosure import PermissionDisclosureRetriever
from .citation import CitationRetriever
from .general import GeneralRetriever
from .semantic import SemanticRetriever

# Экспортируем все публичные классы и функции
__all__ = [
    # Базовые классы
    "BaseRetriever",
    "CandidateChunk",
    "NavigationHit",
    # Функции
    "get_retriever_for_category",
    # Ретриверы
    "OverviewPurposeRetriever",
    "NavigationRetriever",
    "DefinitionRetriever",
    "ScopeRetriever",
    "PenaltiesRetriever",
    "ProceduralRetriever",
    "PermissionDisclosureRetriever",
    "CitationRetriever",
    "GeneralRetriever",
    "SemanticRetriever",
]
