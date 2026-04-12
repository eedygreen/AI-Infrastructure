
from utils import logger
from typing import Optional, Callable
import statistics
import os, json
from pathlib import Path
import asyncio

# RAGAS imports
try:
    from ragas.llms import llm_factory 
    from ragas.embeddings.base import embedding_factory 
    from openai import AsyncOpenAI
    from ragas.metrics.collections import (
        BleuScore, 
        Faithfulness, 
        RougeScore,
        ContextRecall,
        ContextPrecision,
        AnswerRelevancy
    )
    
    RAGAS_AVAILABLE = True
    logger.info("RAGAS imported successfully")

except ImportError as _ragas_import_err:
    RAGAS_AVAILABLE = False
    logger.error(f"RAGAS not available - evaluation disabled. Reason: {_ragas_import_err}", exc_info=True)


def _validate_entry(entry: dict) -> list[str]:
    """
        Return a list of human readable validation errors for one dataset entry
    """
    errors: list[str] = []

    for required in ("question", "contexts", "answer"):
        if not entry.get(required):
            errors.append(f"missing or empty required field: '{required}'")
    
    contexts = entry.get("contexts")
    if contexts is not None:
        if not isinstance(contexts, list):
            errors.append("'context' must be a list of strings, not a single string")
        elif not all(isinstance(c, str) for c in contexts):
            errors.append("Every item in 'contexts' must be a plain string")

    return errors


async def score_metrics(
    metric, 
    user_input,  
    response,
    retrieved_contexts, 
    reference                               
):
    """
        Filter arguments dyanmically based on metric type
    """
    kwargs = {}

    if isinstance(metric, Faithfulness):
        kwargs.update({"user_input": user_input, "response": response, "retrieved_contexts": retrieved_contexts})

    elif isinstance(metric, (ContextRecall, ContextPrecision)):
        kwargs.update({"user_input": user_input, "retrieved_contexts": retrieved_contexts, "reference": reference})
    
    elif isinstance(metric, AnswerRelevancy):
        kwargs.update({"user_input": user_input, "response": response})

    elif isinstance(metric, (BleuScore, RougeScore)):
        kwargs.update({"reference": reference, "response": response})
    
    return await metric.ascore(**kwargs)

async def evaluate_response_quality(
        question: str, 
        answer: str, 
        contexts: list[str],
        ground_truth: Optional[str] = None
        ) -> dict[str, float]:
    """
        Evaluate RAG response quality using RAGAS metrics
        Metrics without ground_truth: Faithfulness, ResponseRelevancy
        Metrics with ground_truth: + BleuScore, RougeScore, NonLLMContextPrecisionWithReference
    
        Args:
            question: the user query
            answer: The generated response
            contexts: Retrieved context chunks passed to the LLM
            ground_truth: Optional reference answer for reference-based metrics
    """

    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    
    missing: list = []
    
    if not question:
        missing.append("question")
    
    if not answer:
        missing.append("answer")
    
    if not contexts:
        missing.append("contexts")

    if missing:
        return {"error": f"Required inputs are empty or missing: {','.join(missing)}"}
    
    if not isinstance(contexts, list):
        return {"error": "'contexts' must be a list of strings, not a single string"}
    
    try:
    # Create evaluator LLM with model gpt-3.5-turbo
        openai_api_key = os.getenv("OPENAI_API_KEY")
    
        if openai_api_key.startswith("voc-"):
            client = AsyncOpenAI(
                api_key=openai_api_key,
                base_url="https://openai.vocareum.com/v1" 
            )
        else:
            client = AsyncOpenAI(api_key=openai_api_key)

        kwargs_llm = dict(
            model_llm="gpt-3.5-turbo",
            model_emb="text-embedding-3-small",
        )

        evaluator_llm = llm_factory(kwargs_llm["model_llm"], client=client)
    
    #  Create evaluator_embeddings with model test-embedding-3-small
        eval_embedding = embedding_factory(
            "openai",
            model=kwargs_llm['model_emb'],
            client=client,
            interface='modern'
        )

        if evaluator_llm is None:
            logger.error(f"evaluator_llm is: {evaluator_llm}", exc_info=True)
            raise ValueError("evlulator_llm is not initialized")
        
        if eval_embedding is None:
            logger.error(f"evaluator_embedding is: {eval_embedding}", exc_info=True)
            raise ValueError(f"evaluator_embedding is not initialized")
    # Define an instance for each metric to evaluate
        # metrics without ground_truth (No reference answer)
        metrics = [
            Faithfulness(llm=evaluator_llm), 
            AnswerRelevancy(llm=evaluator_llm, embeddings=eval_embedding),
            ContextRecall(llm=evaluator_llm),
            ContextPrecision(llm=evaluator_llm)    
        ]

        if ground_truth:
            metrics += [
                BleuScore(),                                # Non-llm
                RougeScore()
            ]

        invalid = [i for i, m in enumerate(metrics) if m is None]
        if invalid:
            raise ValueError(f"Metrics at indices {invalid} are None = check initialization")

        results = await asyncio.gather(*[
            score_metrics(
                metric,
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=ground_truth
            )
            for metric in metrics
        ])    

        scores: dict[str, float] = {
           type(metric).__name__: result.value for metric, result in zip(metrics, results)      
        }

        logger.info(f"RAGAS evaluation complete: {scores}")
        return scores                       # Return the evaluation results
    
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}", exc_info=True)
        return {"error": str(e)}
    
def _aggregate_scores(per_question_scores: list[dict[str, float]]) -> dict[str, dict]:
    """
        Compute per-metric mean, min, max, stdev and sample count.
    """
    metric_values: dict[str, list[float]] = {}

    for scores in per_question_scores:
        for metric, value in scores.items():
            if isinstance(value, (int, float)):
                metric_values.setdefault(metric, []).append(float(value))

    aggregate: dict[str, dict] = {}
    for metric, values in metric_values.items():
        aggregate[metric] = {
            "mean": round(statistics.mean(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            "count": len(values),
        }
    return aggregate

def load_evaluation_dataset(filepath: str) -> list[dict]:
    """
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

        Returns:
            List of validated question dicts.  Entries that fail validation are
            skipped with a logged warning rather than crashing the whole load.
    """

    path = Path(filepath)

    if not path.exists():
        logger.error(f"Dataset file not found: {filepath}", exc_info=True)
        return []
    
    suffix = path.suffix.lower()

    try:
        if suffix == ".json":
            with open(path, 'r') as ds:
                raw = json.load(ds)
            entries = raw.get("questions", []) if isinstance(raw, dict) else raw

        elif suffix in (".txt", ".jsonl"):
            with open(path, 'r') as ds:
                entries = [
                    json.loads(line)
                    for line in ds
                    if line.strip() and not line.startswith("#")
                ]
        else:
            logger.error(f"Unsupported file format: '{suffix}'. Use .json or .txt/.jsonl", exc_info=True)
            return []

    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to load dataset from {filepath}: {e}", exc_info=True)
        return []
    
    valid_entries: list[dict] = []
    
    for idx, entry in enumerate(entries):
        errors = _validate_entry(entry)
        if errors:
            entry_id = entry.get("id", f"index {idx}")
            logger.warning(f"Skipping entry '{entry_id}': {';'.join(errors)}")
        else:
            valid_entries.append(entry)

    logger.info(f"Loaded {len(valid_entries)}/{len(entries)} valid entries from {filepath}")
    return valid_entries

def evaluate_dataset(
    filepath: str, 
    rag_fn: Optional[Callable[[str, list[str]], str]]=None
) -> dict:
    """
        Run end-to-end evaluation over all questions in a dataset file.

        Args:
            filepath: Path to test_questions.json or evaluation_dataset.txt.
            rag_fn:   Optional callable(question, contexts) → answer.
                    When provided, answers are generated live by your RAG pipeline
                    instead of using the pre-written answers in the dataset.
                    Signature: rag_fn(question: str, contexts: List[str]) -> str

        Returns:
            {
            "per_question": [{"id": ..., "category": ..., "scores": {...}}, ...],
            "aggregate":    {"faithfulness": {"mean": ..., "min": ..., ...}, ...},
            "summary":      {"total": N, "failed": M}
            }
    """

    entries = load_evaluation_dataset(filepath)

    if not entries:
        return {"error": f"No valide entries found in {filepath}"}
    
    per_question_results: list[dict] = []
    per_question_scores: list[dict[str, float]] = []
    failed = 0

    for entry in entries:
        question = entry["question"]
        contexts = entry["contexts"]
        ground_truth = entry.get("ground_truth")

        if rag_fn is not None:
            try:
                answer = rag_fn(question, contexts)
            except Exception as e:
                logger.error(f"rag_fn failed for '{entry.get('id')}': {e}", exc_info=True)
                answer = entry.get("answer", "")

        else:
            answer = entry.get("answer", "")

        scores = asyncio.run(evaluate_response_quality(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth
        ))

        if "error" in scores:
            logger.error(f"Evaluation failed for entry '{entry.get('id')}': {scores['error']}", exc_info=True)
            failed += 1

        per_question_results.append({
            "id": entry.get("id", "unknown"),
            "category": entry.get("category", "uncategorized"),
            "question": question,
            "scores": scores
        })

        numeric_scores = {metric: value for metric, value in scores.items() if isinstance(value, (int, float))}
        if numeric_scores:
            per_question_scores.append(numeric_scores)

    aggregate = _aggregate_scores(per_question_scores)

    return {
        "per_question": per_question_results,
        "aggregate": aggregate,
        "summary" : {
            "total": len(entries),
            "failed": failed,
            "scored": len(per_question_scores),
        },
    }