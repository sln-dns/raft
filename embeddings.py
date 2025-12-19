import os
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class EmbeddingClient:
    """Клиент для работы с эмбеддингами через VseGPT API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.vsegpt.ru/v1",
        model: Optional[str] = None,
    ):
        """
        Инициализация клиента для работы с эмбеддингами.
        
        Args:
            api_key: API ключ VseGPT (если не указан, берется из VSEGPT_API_KEY)
            base_url: Базовый URL API (по умолчанию VseGPT)
            model: Модель для эмбеддингов (если не указана, берется из EMBEDDING_MODEL)
        """
        self.api_key = api_key or os.getenv("VSEGPT_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API ключ не найден. Укажите VSEGPT_API_KEY в .env файле или передайте api_key напрямую."
            )
        
        self.base_url = base_url
        self.model = model or os.getenv("EMBEDDING_MODEL", "emb-qwen/qwen3-embedding-8b")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
    
    def create_embedding(
        self,
        text: str,
        encoding_format: str = "float",
    ) -> List[float]:
        """
        Создает эмбеддинг для одного текста.
        
        Args:
            text: Текст для создания эмбеддинга
            encoding_format: Формат кодирования ('float' или 'base64')
        
        Returns:
            Список чисел (вектор эмбеддинга)
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format=encoding_format,
        )
        return response.data[0].embedding
    
    def create_embeddings_batch(
        self,
        texts: List[str],
        encoding_format: str = "float",
    ) -> List[List[float]]:
        """
        Создает эмбеддинги для списка текстов.
        
        Args:
            texts: Список текстов для создания эмбеддингов
            encoding_format: Формат кодирования ('float' или 'base64')
        
        Returns:
            Список векторов эмбеддингов
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format=encoding_format,
        )
        return [item.embedding for item in response.data]
    
    def get_embedding_dimension(self) -> int:
        """
        Возвращает размерность эмбеддингов для текущей модели.
        
        Returns:
            Размерность вектора эмбеддинга
        """
        # Создаем тестовый эмбеддинг для определения размерности
        test_embedding = self.create_embedding("test")
        return len(test_embedding)


# Глобальный экземпляр клиента (создается при первом использовании)
_client: Optional[EmbeddingClient] = None


def get_embedding_client() -> EmbeddingClient:
    """
    Возвращает глобальный экземпляр клиента для эмбеддингов.
    
    Returns:
        Экземпляр EmbeddingClient
    """
    global _client
    if _client is None:
        _client = EmbeddingClient()
    return _client


if __name__ == "__main__":
    # Пример использования
    client = EmbeddingClient()
    
    # Создание эмбеддинга для одного текста
    text = "The food was delicious and the waiter..."
    embedding = client.create_embedding(text)
    print(f"Размерность эмбеддинга: {len(embedding)}")
    print(f"Первые 5 значений: {embedding[:5]}")
    
    # Создание эмбеддингов для нескольких текстов
    texts = [
        "First text to embed",
        "Second text to embed",
    ]
    embeddings = client.create_embeddings_batch(texts)
    print(f"\nСоздано {len(embeddings)} эмбеддингов")
