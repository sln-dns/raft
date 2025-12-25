"""Минималистичный Streamlit UI для HIPAA RAG API."""

import streamlit as st
import requests
import os
from typing import Optional, Tuple
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация API
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def post_json(path: str, payload: dict) -> Tuple[int, Optional[dict]]:
    """
    Выполняет POST запрос к API и возвращает статус код и JSON ответ.
    
    Args:
        path: Путь к эндпоинту (например, "/answer")
        payload: Тело запроса (словарь)
    
    Returns:
        Tuple[status_code, json_response] или (status_code, None) при ошибке
    """
    url = f"{API_BASE_URL}{path}"
    
    try:
        session = requests.Session()
        session.trust_env = False  # Отключаем системные прокси
        
        response = session.post(
            url,
            json=payload,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.status_code, response.json()
        else:
            logger.error(f"API returned status {response.status_code}: {response.text}")
            return response.status_code, None
    
    except requests.exceptions.Timeout:
        logger.error(f"Request timeout for {url}")
        return 0, None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        return 0, None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 0, None


def main():
    """Главная функция приложения."""
    
    st.set_page_config(
        page_title="HIPAA Regulations",
        page_icon=None,
        layout="centered"
    )
    
    st.title("HIPAA Regulations")
    
    # Поле для ввода вопроса
    question = st.text_area(
        "Question",
        height=100,
        placeholder="Enter your question about HIPAA regulations..."
    )
    
    # Кнопка для получения ответа
    if st.button("Get Answer", type="primary"):
        if not question or not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Processing..."):
                status_code, response = post_json("/answer", {
                    "question": question.strip(),
                    "max_results": 8
                })
                
                if status_code == 200 and response:
                    # Сохраняем ответ в session_state
                    st.session_state.answer = response.get("answer", "")
                    st.session_state.citations = response.get("sources", [])
                else:
                    st.error("Failed to get answer from API.")
                    st.session_state.answer = None
                    st.session_state.citations = None
    
    # Вывод ответа
    if "answer" in st.session_state and st.session_state.answer:
        st.divider()
        st.subheader("Answer")
        st.write(st.session_state.answer)
    
    # Вывод citations (proof)
    if "citations" in st.session_state and st.session_state.citations:
        st.divider()
        st.subheader("Proof")
        for citation in st.session_state.citations:
            st.write(citation)


if __name__ == "__main__":
    main()
