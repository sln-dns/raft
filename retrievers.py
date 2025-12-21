"""Shim-модуль для обратной совместимости с монолитным retrievers.py.

Все ретриверы перенесены в пакет retrievers/.
Этот файл оставлен только для обратной совместимости со старыми импортами.
"""

from typing import Optional

# Импортируем базовые классы из нового пакета
from retrievers.base import BaseRetriever, CandidateChunk, NavigationHit

# Импортируем функцию выбора ретривера из нового пакета
from retrievers.registry import get_retriever_for_category

# Импортируем все ретриверы из нового пакета
from retrievers.overview_purpose import OverviewPurposeRetriever
from retrievers.navigation import NavigationRetriever
from retrievers.definition import DefinitionRetriever
from retrievers.scope import ScopeRetriever
from retrievers.penalties import PenaltiesRetriever
from retrievers.procedural import ProceduralRetriever
from retrievers.permission_disclosure import PermissionDisclosureRetriever
from retrievers.citation import CitationRetriever
from retrievers.general import GeneralRetriever

# Импортируем утилиты из нового пакета
from retrievers.utils import get_db_connection

# Импортируем SemanticRetriever из нового модуля
from retrievers.semantic import SemanticRetriever


# Экспортируем все для обратной совместимости
__all__ = [
    # Базовые классы
    "BaseRetriever",
    "CandidateChunk",
    "NavigationHit",
    # Функции
    "get_retriever_for_category",
    "get_db_connection",
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
