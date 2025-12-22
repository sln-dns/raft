"""Streamlit UI для HIPAA Regulations RAG API.

Позволяет:
- Ввести вопрос
- Выбрать режим вызова (classify/search/answer)
- Увидеть результат и источники
- Видеть debug-информацию в отдельной панели
"""

import streamlit as st
import requests
import json
import time
from typing import Optional, Tuple, Dict, Any

# Page configuration
st.set_page_config(
    page_title="HIPAA Regulations RAG",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# API Client Layer
# ============================================================================

def post_json(path: str, payload: Dict[str, Any], base_url: str, timeout: int = 30) -> Tuple[int, Optional[Dict[str, Any]], Optional[str]]:
    """
    Send POST request to API endpoint with JSON payload.
    
    Args:
        path: API endpoint path (e.g., "/classify", "/search", "/answer")
        payload: JSON payload as dictionary
        base_url: Base URL of the API
        timeout: Request timeout in seconds (default: 30)
    
    Returns:
        Tuple of (status_code, json_data, error_message)
        - If success: (200, json_data, None)
        - If error: (status_code, None, error_message)
    """
    if not base_url:
        return 0, None, "API Base URL is not set"
    
    # Remove trailing slash from base_url and leading slash from path
    base_url = base_url.rstrip('/')
    path = path.lstrip('/')
    url = f"{base_url}/{path}"
    
    # Create session that doesn't trust environment proxy settings
    session = requests.Session()
    session.trust_env = False
    
    try:
        response = session.post(
            url,
            json=payload,
            timeout=timeout,
            headers={
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        )
        
        # Check status code
        if response.status_code == 200:
            # Try to parse JSON
            try:
                json_data = response.json()
                return 200, json_data, None
            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON response: {str(e)[:200]}"
                return 200, None, error_msg
        else:
            # Non-200 status code
            try:
                error_data = response.json()
                error_msg = error_data.get('detail', f"HTTP {response.status_code}: {response.text[:200]}")
            except:
                error_msg = f"HTTP {response.status_code}: {response.text[:200] if response.text else 'No response body'}"
            return response.status_code, None, error_msg
            
    except requests.exceptions.Timeout:
        return 0, None, f"Request timeout after {timeout} seconds"
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Connection error: Cannot connect to {url}"
        if "Connection refused" in str(e):
            error_msg += " (API may not be running)"
        return 0, None, error_msg
    except requests.exceptions.RequestException as e:
        return 0, None, f"Request error: {str(e)[:200]}"
    except Exception as e:
        return 0, None, f"Unexpected error: {type(e).__name__}: {str(e)[:200]}"


def display_api_error(status_code: int, error_message: str, show_details: bool = True):
    """
    Display API error in a user-friendly way.
    
    Args:
        status_code: HTTP status code (0 if connection error)
        error_message: Error message
        show_details: Whether to show detailed error information
    """
    if status_code == 0:
        st.error(f"🔌 **Connection Error**")
        st.error(error_message)
    elif status_code == 422:
        st.error(f"❌ **Validation Error** (HTTP {status_code})")
        st.error(error_message)
        if show_details:
            st.info("💡 Check that your question is not empty and parameters are valid")
    elif status_code == 500:
        st.error(f"❌ **Server Error** (HTTP {status_code})")
        st.error(error_message)
        if show_details:
            st.info("💡 The API server encountered an error. Check server logs.")
    elif status_code == 502:
        st.error(f"❌ **Bad Gateway** (HTTP {status_code})")
        st.error(error_message)
        if show_details:
            st.info("💡 API server may be behind a proxy or not responding correctly")
    elif status_code == 503:
        st.error(f"❌ **Service Unavailable** (HTTP {status_code})")
        st.error(error_message)
        if show_details:
            st.info("💡 API server is temporarily unavailable")
    else:
        st.error(f"❌ **Error** (HTTP {status_code})")
        st.error(error_message)

# ============================================================================
# Main Application
# ============================================================================

# Header
st.title("📋 HIPAA Regulations RAG System")

# Sidebar - API Configuration
with st.sidebar:
    st.header("⚙️ API Configuration")
    
    # API Base URL input
    api_base_url = st.text_input(
        "API Base URL",
        value="http://localhost:8000",
        help="Base URL of the FastAPI server"
    )
    
    # Remove trailing slash if present
    if api_base_url:
        api_base_url = api_base_url.rstrip('/')
    
    # Health check function
    def check_api_health(base_url: str) -> Tuple[bool, Optional[str]]:
        """Check API health.
        
        Returns:
            Tuple of (is_healthy, error_message)
        """
        if not base_url:
            return False, "API URL is empty"
        
        # Create session that doesn't trust environment proxy settings
        # This prevents issues when system proxy is configured
        session = requests.Session()
        session.trust_env = False
        
        health_url = f"{base_url}/health"
        try:
            response = session.get(health_url, timeout=3)
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get("status") == "healthy":
                        return True, None
                    else:
                        return False, f"API status: {data.get('status', 'unknown')}"
                except:
                    return False, "Invalid JSON response from /health"
            else:
                # Try root endpoint as fallback
                try:
                    root_response = session.get(base_url, timeout=3)
                    if root_response.status_code == 200:
                        return True, "API is reachable (health endpoint not available)"
                    else:
                        return False, f"HTTP {root_response.status_code}"
                except:
                    return False, f"HTTP {response.status_code} from /health"
        except requests.exceptions.Timeout:
            return False, "Connection timeout"
        except requests.exceptions.ConnectionError:
            return False, "Connection refused - API may not be running"
        except Exception as e:
            return False, f"Error: {str(e)[:100]}"
    
    # Ping button
    if st.button("🔍 Ping API", use_container_width=True):
        is_healthy, error_msg = check_api_health(api_base_url)
        st.session_state.api_health_status = ("OK", None) if is_healthy else ("ERROR", error_msg)
    
    # Display API status
    if "api_health_status" in st.session_state:
        status, message = st.session_state.api_health_status
        if status == "OK":
            st.success(f"✅ API Status: {status}")
            if message:
                st.info(message)
        else:
            st.error(f"❌ API Status: {status}")
            if message:
                st.error(f"Error: {message}")
    else:
        st.info("ℹ️ Click 'Ping API' to check status")
    
    st.divider()
    
    # Debug Panel
    # Use session_state to access debug_mode (defined later in Parameters section)
    debug_mode_value = st.session_state.get("debug_mode", True)
    if debug_mode_value:
        st.header("🔍 Debug Panel")
        
        # Classification debug
        if "last_classification" in st.session_state:
            with st.expander("📊 Last Classification", expanded=False):
                last_class = st.session_state.last_classification
                st.write("**Question:**", last_class.get("question", "N/A"))
                st.write("**Category:**", last_class.get("category", "N/A"))
                st.write("**Confidence:**", f"{last_class.get('confidence', 0.0) * 100:.1f}%")
                reasoning = last_class.get("reasoning")
                if reasoning:
                    st.write("**Reasoning:**")
                    st.caption(reasoning)
                st.write("**Latency:**", f"{last_class.get('latency', 0.0):.3f}s")
                
                show_raw_json_value = st.session_state.get("show_raw_json", False)
                if show_raw_json_value:
                    st.markdown("**Raw Response:**")
                    st.json(last_class.get("raw_response", {}))
        
        # Search debug
        if "last_search" in st.session_state:
            with st.expander("🔎 Last Search", expanded=False):
                last_search = st.session_state.last_search
                st.write("**Question:**", last_search.get("question", "N/A"))
                
                classification = last_search.get("classification", {})
                if classification:
                    st.write("**Category:**", classification.get("category", "N/A"))
                    st.write("**Confidence:**", f"{classification.get('confidence', 0.0) * 100:.1f}%")
                
                st.write("**Total Found:**", last_search.get("total_found", 0))
                st.write("**Latency:**", f"{last_search.get('latency', 0.0):.3f}s")
                
                retrieved_chunks = last_search.get("retrieved_chunks", [])
                if retrieved_chunks:
                    st.write(f"**Chunks Count:** {len(retrieved_chunks)}")
                    # Show first chunk details
                    if retrieved_chunks:
                        first_chunk = retrieved_chunks[0]
                        st.caption(f"First chunk: {first_chunk.get('anchor', first_chunk.get('chunk_id', 'N/A'))}")
                        st.caption(f"Score: {first_chunk.get('similarity', 0.0):.4f}")
                
                show_raw_json_value = st.session_state.get("show_raw_json", False)
                if show_raw_json_value:
                    st.markdown("**Raw Response:**")
                    st.json(last_search.get("raw_response", {}))
        
        # Answer debug
        if "last_answer" in st.session_state:
            with st.expander("💬 Last Answer", expanded=False):
                last_answer = st.session_state.last_answer
                st.write("**Question:**", last_answer.get("question", "N/A"))
                
                classification = last_answer.get("classification", {})
                if classification:
                    st.write("**Category:**", classification.get("category", "N/A"))
                    st.write("**Confidence:**", f"{classification.get('confidence', 0.0) * 100:.1f}%")
                
                debug_info = last_answer.get("debug", {})
                if debug_info:
                    st.markdown("**Debug Info:**")
                    
                    answer_policy = debug_info.get("answer_policy") or debug_info.get("policy", "N/A")
                    st.write(f"**Answer Policy:** {answer_policy}")
                    
                    prompt_template = debug_info.get("prompt_template", "N/A")
                    st.write(f"**Prompt Template:** {prompt_template}")
                    
                    chunks_count = debug_info.get("chunks_count") or debug_info.get("context_items_count", "N/A")
                    st.write(f"**Context Size:** {chunks_count} chunks")
                    
                    model = debug_info.get("model", "N/A")
                    st.write(f"**Model:** {model}")
                    
                    citations_validated = debug_info.get("citations_validated", False)
                    valid_citations_count = debug_info.get("valid_citations_count") or debug_info.get("citations_count", 0)
                    st.write(f"**Citations Validated:** {'✅ Yes' if citations_validated else '❌ No'}")
                    st.write(f"**Valid Citations:** {valid_citations_count}")
                    
                    llm_skipped = debug_info.get("llm_skipped", False)
                    if llm_skipped:
                        st.warning("⚠️ LLM was skipped")
                    
                    retriever_yesno_signal = debug_info.get("retriever_yesno_signal")
                    if retriever_yesno_signal:
                        st.write(f"**Yes/No Signal:** {retriever_yesno_signal}")
                    
                    retriever_policy_signal = debug_info.get("retriever_policy_signal")
                    permission_policy = debug_info.get("permission_policy")
                    if retriever_policy_signal or permission_policy:
                        st.write(f"**Policy Signal:** {retriever_policy_signal or permission_policy}")
                
                st.write("**Latency:**", f"{last_answer.get('latency', 0.0):.3f}s")
                st.write("**Sources Count:**", len(last_answer.get("sources", [])))
                
                # Override information
                if classification:
                    reasoning = classification.get("reasoning", "")
                    if "Override:" in reasoning or "override" in reasoning.lower():
                        st.markdown("**Classification Override:**")
                        if "Override:" in reasoning:
                            override_reason = reasoning.split("Override:")[1].split("Original:")[0].strip()
                            st.caption(f"Reason: {override_reason}")
                
                show_raw_json_value = st.session_state.get("show_raw_json", False)
                if show_raw_json_value:
                    st.markdown("**Raw Response:**")
                    st.json(last_answer.get("raw_response", {}))
        
        # Show message if no debug data available
        if ("last_classification" not in st.session_state and 
            "last_search" not in st.session_state and 
            "last_answer" not in st.session_state):
            st.info("ℹ️ No debug data available yet. Run Classify, Search, or Answer to see debug information.")

# Store API base URL in session state for use in main content
st.session_state.api_base_url = api_base_url

# Main content area
st.markdown("---")

# Question input
st.subheader("📝 Question")
question = st.text_area(
    "Enter your question about HIPAA regulations:",
    height=120,
    placeholder="E.g., What does 'minimum necessary' mean in HIPAA terminology?",
    help="Enter your question about HIPAA regulations",
    key="question_input"
)

# Action buttons and parameters
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    classify_button = st.button("🔍 Classify", use_container_width=True, type="secondary")

with col2:
    search_button = st.button("🔎 Search", use_container_width=True, type="secondary")

with col3:
    answer_button = st.button("💬 Answer", use_container_width=True, type="primary")

# Parameters section
st.markdown("---")
st.subheader("⚙️ Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    max_results = st.slider(
        "Top K",
        min_value=1,
        max_value=15,
        value=8,
        help="Maximum number of results to retrieve"
    )

with col2:
    debug_mode = st.checkbox(
        "Debug mode",
        value=True,
        help="Show debug information"
    )

with col3:
    show_raw_json = st.checkbox(
        "Show raw JSON",
        value=False,
        help="Display raw JSON response from API"
    )

# Store parameters in session state
st.session_state.max_results = max_results
st.session_state.debug_mode = debug_mode
st.session_state.show_raw_json = show_raw_json

# Handle button clicks
if classify_button:
    if not question.strip():
        st.warning("⚠️ Please enter a question")
    else:
        # Prepare payload for classification
        payload = {
            "question": question.strip()
        }
        
        # Measure latency
        start_time = time.time()
        
        # Show loading state
        with st.spinner("🔄 Classifying question..."):
            # Call API
            status_code, json_data, error_message = post_json(
                path="/classify",
                payload=payload,
                base_url=api_base_url,
                timeout=30
            )
        
        latency = time.time() - start_time
        
        # Display results or errors
        if status_code == 200 and json_data:
            # Save to session state
            st.session_state.last_classification = {
                "question": question.strip(),
                "category": json_data.get("category"),
                "confidence": json_data.get("confidence"),
                "reasoning": json_data.get("reasoning"),
                "latency": latency,
                "raw_response": json_data
            }
            
            # Display main results
            st.success("✅ Classification completed successfully")
            st.markdown("---")
            
            # Category and confidence
            col1, col2 = st.columns(2)
            
            with col1:
                category = json_data.get("category", "unknown")
                category_emoji = {
                    "definition": "📖",
                    "overview / purpose": "📋",
                    "overview / navigation": "🗺️",
                    "scope / applicability": "🎯",
                    "penalties": "⚖️",
                    "procedural / best practices": "🔧",
                    "permission / disclosure": "🔓",
                    "citation-required": "📝",
                    "regulatory_principle": "⚖️",
                    "other": "❓"
                }.get(category, "❓")
                
                st.metric(
                    label="Category",
                    value=f"{category_emoji} {category}"
                )
            
            with col2:
                confidence = json_data.get("confidence", 0.0)
                confidence_pct = f"{confidence * 100:.1f}%"
                st.metric(
                    label="Confidence",
                    value=confidence_pct
                )
            
            # Reasoning (if available)
            reasoning = json_data.get("reasoning")
            if reasoning:
                st.markdown("---")
                st.subheader("💭 Reasoning")
                st.info(reasoning)
        else:
            # Display error
            display_api_error(status_code, error_message or "Unknown error", debug_mode)

elif search_button:
    if not question.strip():
        st.warning("⚠️ Please enter a question")
    else:
        # Prepare payload for search
        payload = {
            "question": question.strip(),
            "max_results": max_results
        }
        
        # Add category if available from last classification
        if "last_classification" in st.session_state:
            last_cat = st.session_state.last_classification.get("category")
            if last_cat:
                # Note: API doesn't accept category in payload, but we can use it for display
                pass
        
        # Measure latency
        start_time = time.time()
        
        # Show loading state
        with st.spinner("🔄 Searching for relevant chunks..."):
            # Call API
            status_code, json_data, error_message = post_json(
                path="/search",
                payload=payload,
                base_url=api_base_url,
                timeout=60
            )
        
        latency = time.time() - start_time
        
        # Display results or errors
        if status_code == 200 and json_data:
            # Save to session state
            st.session_state.last_search = {
                "question": question.strip(),
                "classification": json_data.get("classification", {}),
                "retrieved_chunks": json_data.get("retrieved_chunks", []),
                "total_found": json_data.get("total_found", 0),
                "latency": latency,
                "raw_response": json_data
            }
            
            # Display main results
            st.success("✅ Search completed successfully")
            st.markdown("---")
            
            # Show classification info (if available)
            classification = json_data.get("classification", {})
            if classification:
                col1, col2 = st.columns(2)
                with col1:
                    category = classification.get("category", "unknown")
                    category_emoji = {
                        "definition": "📖",
                        "overview / purpose": "📋",
                        "overview / navigation": "🗺️",
                        "scope / applicability": "🎯",
                        "penalties": "⚖️",
                        "procedural / best practices": "🔧",
                        "permission / disclosure": "🔓",
                        "citation-required": "📝",
                        "regulatory_principle": "⚖️",
                        "other": "❓"
                    }.get(category, "❓")
                    st.write(f"**Category:** {category_emoji} {category}")
                with col2:
                    confidence = classification.get("confidence", 0.0)
                    st.write(f"**Confidence:** {confidence * 100:.1f}%")
            
            # Show total found
            total_found = json_data.get("total_found", 0)
            st.markdown(f"**Found:** {total_found} chunk{'s' if total_found != 1 else ''}")
            st.markdown("---")
            
            # Display chunks
            retrieved_chunks = json_data.get("retrieved_chunks", [])
            if retrieved_chunks:
                st.subheader("📄 Retrieved Chunks")
                
                for idx, chunk in enumerate(retrieved_chunks, 1):
                    with st.container():
                        # Chunk header
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            anchor = chunk.get("anchor", "N/A")
                            section_number = chunk.get("section_number", "N/A")
                            section_title = chunk.get("section_title", "N/A")
                            
                            if anchor and anchor != "N/A":
                                st.markdown(f"**{idx}. {anchor}**")
                            else:
                                chunk_id = chunk.get("chunk_id", "N/A")
                                st.markdown(f"**{idx}. {chunk_id}**")
                            
                            st.caption(f"§{section_number} - {section_title}")
                        
                        with col2:
                            score = chunk.get("similarity", 0.0)
                            st.metric("Score", f"{score:.3f}")
                        
                        with col3:
                            chunk_kind = chunk.get("chunk_kind")
                            granularity = chunk.get("granularity")
                            if chunk_kind or granularity:
                                kind_text = chunk_kind or "N/A"
                                gran_text = granularity or "N/A"
                                st.caption(f"Kind: {kind_text}")
                                st.caption(f"Gran: {gran_text}")
                        
                        # Text raw in expander
                        text_raw = chunk.get("text_raw", "")
                        if text_raw:
                            with st.expander(f"📝 View text (chunk {idx})"):
                                st.text(text_raw)
                        
                        st.divider()
            else:
                st.info("No chunks found for this question.")
        else:
            # Display error
            display_api_error(status_code, error_message or "Unknown error", debug_mode)

elif answer_button:
    if not question.strip():
        st.warning("⚠️ Please enter a question")
    else:
        # Prepare payload for answer generation
        payload = {
            "question": question.strip(),
            "max_results": max_results
        }
        
        # Measure latency
        start_time = time.time()
        
        # Show loading state
        with st.spinner("🔄 Generating answer..."):
            # Call API
            status_code, json_data, error_message = post_json(
                path="/answer",
                payload=payload,
                base_url=api_base_url,
                timeout=120  # Longer timeout for LLM generation
            )
        
        latency = time.time() - start_time
        
        # Display results or errors
        if status_code == 200 and json_data:
            # Save to session state
            st.session_state.last_answer = {
                "question": question.strip(),
                "classification": json_data.get("classification", {}),
                "answer": json_data.get("answer", ""),
                "sources": json_data.get("sources", []),
                "retrieved_chunks": json_data.get("retrieved_chunks", []),
                "debug": json_data.get("debug", {}),
                "latency": latency,
                "raw_response": json_data
            }
            
            # Display main results
            st.success("✅ Answer generated successfully")
            st.markdown("---")
            
            # Show classification info (if available)
            classification = json_data.get("classification", {})
            if classification:
                col1, col2 = st.columns(2)
                with col1:
                    category = classification.get("category", "unknown")
                    category_emoji = {
                        "definition": "📖",
                        "overview / purpose": "📋",
                        "overview / navigation": "🗺️",
                        "scope / applicability": "🎯",
                        "penalties": "⚖️",
                        "procedural / best practices": "🔧",
                        "permission / disclosure": "🔓",
                        "citation-required": "📝",
                        "regulatory_principle": "⚖️",
                        "other": "❓"
                    }.get(category, "❓")
                    st.write(f"**Category:** {category_emoji} {category}")
                with col2:
                    confidence = classification.get("confidence", 0.0)
                    st.write(f"**Confidence:** {confidence * 100:.1f}%")
            
            st.markdown("---")
            
            # Main answer text
            answer_text = json_data.get("answer", "")
            if answer_text:
                st.subheader("💬 Answer")
                # Format answer text (preserve line breaks)
                st.markdown(answer_text.replace("\n", "\n\n"))
            else:
                st.warning("⚠️ No answer generated")
            
            # Sources and Citations
            sources = json_data.get("sources", [])
            retrieved_chunks = json_data.get("retrieved_chunks", [])
            
            # Build citations from sources and retrieved_chunks
            citations = []
            if sources:
                # Create a map of anchor -> chunk for quick lookup
                anchor_to_chunk = {}
                for chunk in retrieved_chunks:
                    anchor = chunk.get("anchor")
                    if anchor:
                        anchor_to_chunk[anchor] = chunk
                    # Also index by chunk_id as fallback
                    chunk_id = chunk.get("chunk_id")
                    if chunk_id:
                        anchor_to_chunk[chunk_id] = chunk
                
                # Build citations from sources
                for source in sources:
                    chunk = anchor_to_chunk.get(source)
                    if chunk:
                        citations.append({
                            "anchor": source,
                            "quote": chunk.get("text_raw", ""),
                            "section_number": chunk.get("section_number", ""),
                            "section_title": chunk.get("section_title", "")
                        })
                    else:
                        # Source not found in chunks, add as anchor only
                        citations.append({
                            "anchor": source,
                            "quote": "",
                            "section_number": "",
                            "section_title": ""
                        })
            
            # Display citations
            if citations:
                st.markdown("---")
                st.subheader("📚 Citations")
                
                for idx, citation in enumerate(citations, 1):
                    with st.container():
                        anchor = citation.get("anchor", "N/A")
                        quote = citation.get("quote", "")
                        section_number = citation.get("section_number", "")
                        section_title = citation.get("section_title", "")
                        
                        # Citation header
                        if section_number and section_title:
                            st.markdown(f"**{idx}. {anchor}** - §{section_number} - {section_title}")
                        else:
                            st.markdown(f"**{idx}. {anchor}**")
                        
                        # Quote in expander
                        if quote:
                            with st.expander(f"📝 View quote (citation {idx})"):
                                st.text(quote)
                        
                        st.divider()
            elif sources:
                # Show sources as simple list if no quotes available
                st.markdown("---")
                st.subheader("📚 Sources")
                for idx, source in enumerate(sources, 1):
                    st.markdown(f"{idx}. {source}")
        else:
            # Display error
            display_api_error(status_code, error_message or "Unknown error", debug_mode)
