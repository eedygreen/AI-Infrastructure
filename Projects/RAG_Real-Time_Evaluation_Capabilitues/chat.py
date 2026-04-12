#!/usr/bin/env python3
"""
NASA RAG Chat with RAGAS Evaluation Integration

Enhanced version of the simple RAG chat that includes real-time evaluation
and feedback collection for continuous improvement.
"""

import streamlit as st
import os
import json
import tempfile
import pandas as pd
import asyncio

import ragas_evaluator
import rag_client
import llm_client

from pathlib import Path
from typing import Dict, List, Optional

# RAGAS imports
try:
    from ragas import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    st.warning("RAGAS not available. Install with: pip install ragas")

# Page configuration
st.set_page_config(
    page_title="NASA RAG Chat with Evaluation",
    page_icon="🚀",
    layout="wide"
)

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""

    return rag_client.discover_chroma_backends()

@st.cache_resource
def initialize_rag_system(_chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""

    collection = rag_client.initialize_rag_system(_chroma_dir, collection_name)

    return collection

def retrieve_documents(
    chromadb: str,
    collection_name, 
    query: str, 
    n_results: int = 3, 
    mission_filter: Optional[str] = None
) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""
    collection = initialize_rag_system(chromadb, collection_name)

    if collection is None:
        st.error("RAG initialization failed - check logs for details.")
        return None

    return rag_client.retrieve_documents(
        collection,
        query, 
        n_results, 
        mission_filter,
    )


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    
    return rag_client.format_context(documents, metadatas)

def generate_response(
    openai_key, 
    user_message: str, 
    context: str, 
    conversation_history: List[Dict], 
    model: str = "gpt-3.5-turbo"
) -> str:
    """Generate response using OpenAI with context"""
    try:
        return llm_client.generate_response(
            openai_key, user_message, context, conversation_history, model
        )
    except Exception as e:
        return f"Error generating response: {e}"

def evaluate_response_quality(
    question: str, 
    answer: str, 
    contexts: List[str], 
    ground_truth: str | None = None
) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    try:
        return asyncio.run(ragas_evaluator.evaluate_response_quality(question, answer, contexts, ground_truth))
    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}

def display_evaluation_metrics(scores: Dict[str, float]):
    """Display evaluation metrics in the sidebar"""

    if "error" in scores:
        st.sidebar.error(f"Evaluation Error: {scores['error']}")
        return
    
    st.sidebar.subheader("📊 Response Quality")
    
    for metric_name, score in scores.items():
        if isinstance(score, (int, float)):
            # Color code based on score
            
            if score >= 0.8:
                color = "green"
            elif score >= 0.6:
                color = "orange"
            else:
                color = "red"
            
            st.sidebar.metric(
                label=metric_name.replace('_', ' ').title(),
                value=f"{score:.3f}",
                delta=None
            )
            
            # Add progress bar
            st.sidebar.progress(float(score))

def cleanup_tmp(path: Optional[str]) -> None:
    """Delete a temp file, ignoring errors if it no longer exists"""
    if path:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass

def save_upload_to_tmp(uploaded_file) -> Optional[str]:
    """Write user uploaded file (st.UploadFile) to a temp location and return its path."""
    suffix = None
    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix if uploaded_file.name else ".json"
    
        if not suffix:
                suffix = ".json"

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=suffix,
                prefix="rag_eval_",
                delete=False
            ) as tmp:
                tmp.write(uploaded_file.getvalue())
                return tmp.name

        except OSError as e:
            st.error(f"Could not save Uploaded file: {e}")
            return None
        
def build_aggregate(aggregate: dict) -> pd.DataFrame:
    """Convert aggregate dictionary into DataFrame."""

    rows = [
        {
            "Metric": m.replace("_", " ").title(),
            "Mean": round(s["mean"], 4),
            "Min": round(s["min"], 4),
            "Max": round(s["max"], 4),
            "Stdev": round(s["stdev"], 4),
            "Count": s["count"],
        }
        for m, s in aggregate.items()
    ]

    return pd.DataFrame(rows)

def build_per_question(per_questions: list[dict]) -> pd.DataFrame:
    """Convert per_question list into DataFrame"""
    rows: list = []

    for pq in per_questions:
        row = {
            "ID": pq.get("id", "-"),
            "Category": pq.get("category", "-").replace("_", " ").title(),
            "Question": pq.get("question", "")[:80],
        }

        scores = pq.get("scores", {})
        if "error" in scores:
            row["Status"] = "❌ {scores['error']}"
        else:
            row["Status"] = "✅"
            for metric, value in scores.items():
                if isinstance(value, (int, float)):
                    row[metric.replace("_", " ").title()] = round(value, 4)
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    st.title("🚀 NASA Space Mission Chat with Evaluation")
    st.markdown("Chat with AI about NASA space missions with real-time quality evaluation")

    chat_tab, eval_tab = st.tabs(["Chat", "Evaluation"])  

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_backend" not in st.session_state:
        st.session_state.current_backend = None
    if "last_evaluation" not in st.session_state:
        st.session_state.last_evaluation = None
    if "last_contexts" not in st.session_state:
        st.session_state.last_contexts = []
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Discover available backends
        with st.spinner("Discovering ChromaDB backends..."):
            available_backends = discover_chroma_backends()
        
        if not available_backends:
            st.error("No ChromaDB backends found!")
            st.info("Please run the embedding pipeline first:\n`python run_text_embedding.py`")
            st.stop()
        
        # Backend selection
        st.subheader("📊 ChromaDB Backend")
        backend_options = {k: v["display_name"] for k, v in available_backends.items()}
        
        selected_backend_key = st.selectbox(
            "Select Document Collection",
            options=list(backend_options.keys()),
            format_func=lambda x: backend_options[x],
            help="Choose which document collection to use for retrieval"
        )
        
        selected_backend = available_backends[selected_backend_key]
        
        # API Key input
        st.subheader("🔑 OpenAI Settings")
        openai_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Enter your OpenAI API key"
        )
        
        if not openai_key:
            st.warning("Please enter your OpenAI API key")
            st.stop()
        
        # Model selection
        model_choice = st.selectbox(
            "OpenAI Model",
            options=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
            help="Choose the OpenAI model for responses"
        )
        
        # Retrieval settings
        st.subheader("🔍 Retrieval Settings")
        n_docs = st.slider("Documents to retrieve", 1, 10, 3)
        
        # Evaluation settings
        st.subheader("📊 Evaluation Settings")

        enable_evaluation = st.checkbox(
            "Enable RAGAS Evaluation", 
            value=RAGAS_AVAILABLE,
            disabled=not RAGAS_AVAILABLE
        )

        if st.session_state.get("last_evaluation"):
            st.divider()
            display_evaluation_metrics(st.session_state.last_evaluation)
        
        # Initialize RAG system when backend changes
        if (st.session_state.current_backend != selected_backend_key):
            st.session_state.current_backend = selected_backend_key
            # Clear cache to force reinitialization
            st.cache_resource.clear()
    
    # Initialize RAG system
    with st.spinner("Initializing RAG system..."):
        collection = initialize_rag_system(
            selected_backend["path"], 
            selected_backend["collection"]
        )
    
    if not collection:
        st.error(f"Failed to initialize RAG system - check logs for details")
        st.stop()

    # Display chat messages
    with chat_tab:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input

        if prompt := st.chat_input("Ask about NASA space missions..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            try:
                gd = json.loads(prompt)
                ground_truth = gd.get("ground_truth")
            except json.JSONDecodeError:
                ground_truth = prompt

            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("Searching documents and generating response..."):
                    # Retrieve relevant documents
                    docs_result = retrieve_documents(
                        selected_backend["path"],
                        selected_backend["collection"],
                        prompt, 
                        n_docs
                    )
                    
                    # Format context
                    context = ""
                    contexts_list = []
                    if docs_result and docs_result.get("documents"):
                        context = format_context(docs_result["documents"][0], docs_result["metadatas"][0])
                        contexts_list = docs_result["documents"][0]
                        st.session_state.last_contexts = contexts_list
                    
                    # Generate response
                    response = generate_response(
                        openai_key, 
                        prompt, 
                        context, 
                        st.session_state.messages[:-1],
                        model_choice
                    )
                    st.markdown(response)
                    
                    # Evaluate response quality if enabled
                    if not RAGAS_AVAILABLE:
                        st.warning("RAGAS not available. Install with: pip install ragas")
                    
                    ragas_ready = enable_evaluation and RAGAS_AVAILABLE

                    if ragas_ready:
                        with st.spinner("Evaluating response quality..."):
                            evaluation_scores = evaluate_response_quality(
                                prompt, 
                                response, 
                                contexts_list,
                                ground_truth=ground_truth
                            )
                        st.session_state.last_evaluation = evaluation_scores    
                    else:
                        st.info("Enable RAGAS Evaluation to see scores")
                                    # Display evaluation metrics if available

            st.session_state.messages.append({"role": "assistant", "content": response})

            st.rerun()

            
    # Evaluation: Add assistant response to chat history
    instruction_format = """
        Load Evaluation question from a JSON or plain-text file

        JSON format (test_questions.json):
            {"questions": [{
                "id": "q1",
                "category": "overview",
                "question": "...",
                "contexts": ["...", "..."],
                "answer": "...",
                "ground_truth": "..."   ← optional
            }, ...]}

        Plain-text format (evaluation_dataset.txt):
            One JSON object per line (JSON-Lines), same fields as above.
    """
    
    with eval_tab:
        for key in ["eval_results", "eval_tmp_path"]:
            if key not in st.session_state:
                st.session_state[key] = None
        
        st.markdown(
        "Upload a **`test_questions.json`** or **`evaluation_dataset.txt`** file.  \n"
        "The evaluation runs over every question and displays  \n"
        "per-question scores and aggregate statistics.\n\n"
        "**Accepted formats:**\n"
        "- `.json` — `{\"questions\": [{\"question\", \"contexts\", \"answer\", "
        "\"ground_truth\" (optional)}]}`\n"
        "- `.txt` / `.jsonl` — one JSON object per line"
        )

        st.write("### Upload JSON file")
        uploaded_file = st.file_uploader("Choose a file", type=["json", "txt", "jsonl"],
        help=f"test_questions.json or evaluation_dataset.txt. {instruction_format}",)

        if uploaded_file is not None:
            st.success(f"{uploaded_file.name} uploaded successfully!")   
        
        use_live_rag = st.checkbox(
            "Generate answers live using the RAG pipeline",
            value=False,
            help =(
                "When enabled: the RAG system retrieves documents and calls the LLM "
                "to generate an answer for each question before scoring.  "
                "When disabled: pre-written answers in the dataset file are used"
            ),
        )
        if use_live_rag:
            st.info("✅ Live RAG mode: answers will be generated for each question.")

        run_clicked = st.button(
        "▶ Run Evaluation",
        type="primary",
        disabled=(uploaded_file is None or not RAGAS_AVAILABLE),
        )   

        if not RAGAS_AVAILABLE:
            st.warning("RAGAS is not installed. Install with: `pip install ragas`")

        if run_clicked and uploaded_file  is not None:
            cleanup_tmp(st.session_state.eval_tmp_path)

            tmp_path = save_upload_to_tmp(uploaded_file)
            if tmp_path is None:
                return
            
            st.session_state.eval_tmp_path = tmp_path

            rag_fn = None
            if use_live_rag:
                collection = initialize_rag_system(
                    selected_backend["path"],
                    selected_backend["collection"]
                )

                if collection is None:
                    st.error("RAG initialization failed. Falling back to pre-written answers.")
                else:
                    def rag_fn(question: str, context: list[str]) -> str:
                        docs = retrieve_documents(
                        selected_backend["path"],
                        selected_backend["collection"],
                        question,
                        n_docs,
                    )
                        live_context = ""
                        if docs and docs.get("documents"):
                            live_context = format_context(
                                docs["documents"][0], 
                                docs["metadatas"][0]
                            )
                        try:
                            return llm_client.generate_response(
                                openai_key, question, live_context, [], model_choice
                            )
                        except Exception as e:
                            return f"[generation error: {e}]"
                        
            with st.spinner(f"Running RAGAS evaluation on {uploaded_file.name}..."):
                results = ragas_evaluator.evaluate_dataset(
                    filepath=tmp_path,
                    rag_fn=rag_fn
                )
            st.session_state.eval_results = results

        results = st.session_state.eval_results 
        if results is None:
            st.info("Upload a dataset file and click **▶ Run Evaluation** to see results.")
            return
        
        summary = results.get("summary", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Questions", summary.get("total", 0))
        c2.metric("Scored", summary.get("scored", 0))
        c3.metric("Failed", summary.get("Failed", 0))

        if summary.get("Failed", 0) > 0:
            st.warning(f"{summary['failed']} questions(s) failed - check logs for details.")

        st.divider()

        aggregate = results.get("aggregate", {})
        if aggregate:
            st.subheader("📈 Aggregate Statistics")
            st.dataframe(build_aggregate(aggregate), use_container_width=True, hide_index=False)
            chat_data = pd.DataFrame([
                {
                    "Metric": m.replace("_", " ").title(),
                    "Mean": s["mean"],
                    "Min": s["min"],
                    "Max": s["max"],
                }
                for m, s in aggregate.items()
            ]).set_index("Metric")
            st.bar_chart(chat_data)

        st.divider()

        per_question = results.get("per_question", [])
        if per_question:
            st.subheader("🔍 Per-Question Results")
            st.dataframe(build_per_question(per_question), use_container_width=True, hide_index=False)

            for pq in per_question:
                if not isinstance(pq, dict):
                    st.warning(f"'{pq}' is of type {type(pq).__name__}, skipping...")
                    continue
                scores = pq.get("scores", {})
                icon = "✅" if "error" not in scores else "❌"
                with st.expander(f"{icon} [{pq.get('id', '?')}] {pq.get('question', ' ')[:80]}"):
                    st.markdown(f"**Category:** {pq.get('category', '-').replace('_', ' ').title()}")
                    st.markdown(f"**Question:** {pq.get('question', ' ')}")

                    if "error" in scores:
                        st.error(scores["error"])
                    else:
                        for metric, value in scores.items():
                            if isinstance(value, (int, float)):
                                st.metric(metric.replace("_", " ").title(), f"{value:.4f}")
                                st.progress(float(value))

        st.divider()
        st.download_button(
            label="⬇ Download full results as JSON",
            data=json.dumps(results, indent=2),
            file_name="evaluation_results.json",
            mime="application/json",
        )
        

if __name__ == "__main__":
    main()