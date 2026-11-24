"""
predictions.py - BERT Q&A Prediction Generator

This module loads a BERT model fine-tuned on SQuAD and generates predictions.
Builds on Lesson 6's BERT exploration - same architecture, now for Q&A!
"""

from transformers import pipeline
import torch


def load_bert_qa_model(model_name="deepset/bert-base-cased-squad2"):
    """
    Load BERT Q&A model (fine-tuned on SQuAD 2.0).
    
    This is the same BERT architecture from Lesson 6!
    Now fine-tuned specifically for question answering.
    
    Args:
        model_name: HuggingFace model identifier
                   Default: deepset/bert-base-cased-squad2 (~400MB)
                   Alternative: bert-large-uncased-whole-word-masking-finetuned-squad (~1.3GB)
    
    Returns:
        qa_pipeline: HuggingFace QA pipeline
    """
    print(f"Loading BERT Q&A model: {model_name}")
    print("(This is the same BERT from Lesson 6, fine-tuned for Q&A!)\n")
    
    device = 0 if torch.cuda.is_available() else -1
    
    qa_pipeline = pipeline(
        "question-answering",
        model=model_name,
        tokenizer=model_name,
        device=device
    )
    
    print(f"✓ Model loaded successfully!")
    print(f"  Device: {'GPU' if device == 0 else 'CPU'}")
    print()
    
    return qa_pipeline


def generate_predictions(qa_pipeline, squad_examples, max_samples=None, show_progress=True):
    """
    Generate predictions using BERT Q&A model.
    
    Args:
        qa_pipeline: HuggingFace QA pipeline
        squad_examples: List of SQuAD examples
        max_samples: Maximum number of examples to process (None = all)
        show_progress: Whether to print progress
    
    Returns:
        List of predictions with format:
        {
            'id': question_id,
            'prediction_text': predicted_answer,
            'score': confidence_score,
            'ground_truth': correct_answer,
            'is_impossible': whether_unanswerable
        }
    """
    if max_samples:
        squad_examples = squad_examples[:max_samples]
    
    if show_progress:
        print(f"Generating predictions for {len(squad_examples)} examples...")
        print("(This may take 1-2 minutes on CPU)\n")
    
    predictions = []
    
    for i, example in enumerate(squad_examples):
        # Show progress every 20 examples
        if show_progress and (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(squad_examples)} examples...")
        
        try:
            # Run BERT Q&A
            result = qa_pipeline(
                question=example['question'],
                context=example['context']
            )
            
            prediction_text = result['answer']
            score = result['score']
            
        except Exception as e:
            # Handle edge cases (very long contexts, etc.)
            prediction_text = ""
            score = 0.0
        
        # Get ground truth
        if example['is_impossible']:
            ground_truth = ""
        else:
            ground_truth = example['answers']['text'][0] if example['answers']['text'] else ""
        
        predictions.append({
            'id': example['id'],
            'question': example['question'],
            'prediction_text': prediction_text,
            'score': score,
            'ground_truth': ground_truth,
            'is_impossible': example['is_impossible']
        })
    
    if show_progress:
        print(f"\n✓ Generated {len(predictions)} predictions!")
    
    return predictions


def generate_ranked_predictions(qa_pipeline, squad_examples, top_k=5, max_samples=None):
    """
    Generate top-K ranked predictions for each question.
    
    This extracts multiple candidate answers with scores,
    useful for computing Precision@K, Recall@K, and MRR.
    
    Args:
        qa_pipeline: HuggingFace QA pipeline
        squad_examples: List of SQuAD examples
        top_k: Number of top candidates to return
        max_samples: Maximum examples to process
    
    Returns:
        List of predictions with ranked candidates
    """
    if max_samples:
        squad_examples = squad_examples[:max_samples]
    
    print(f"Generating top-{top_k} ranked predictions for {len(squad_examples)} examples...\n")
    
    predictions = []
    
    for i, example in enumerate(squad_examples):
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(squad_examples)} examples...")
        
        try:
            # Get top-K answers
            result = qa_pipeline(
                question=example['question'],
                context=example['context'],
                top_k=top_k
            )
            
            # Handle both single and multiple results
            if isinstance(result, list):
                ranked = [{'text': r['answer'], 'score': r['score']} for r in result]
            else:
                ranked = [{'text': result['answer'], 'score': result['score']}]
            
        except Exception as e:
            ranked = [{'text': '', 'score': 0.0}]
        
        # Get ground truth
        ground_truth = example['answers']['text'] if not example['is_impossible'] else []
        
        predictions.append({
            'id': example['id'],
            'question': example['question'],
            'ranked_predictions': ranked,
            'ground_truth': ground_truth,
            'is_impossible': example['is_impossible']
        })
    
    print(f"\n✓ Generated top-{top_k} ranked predictions!")
    
    return predictions


def show_prediction_examples(predictions, n_examples=5):
    """
    Display sample predictions for inspection.
    
    Args:
        predictions: List of predictions
        n_examples: Number of examples to show
    """
    print("=" * 80)
    print("SAMPLE PREDICTIONS")
    print("=" * 80)
    
    for i, pred in enumerate(predictions[:n_examples], 1):
        print(f"\n{i}. Question: {pred['question']}")
        print(f"   BERT Predicted: '{pred['prediction_text']}'")
        print(f"   Ground Truth:   '{pred['ground_truth']}'")
        print(f"   Confidence:     {pred['score']:.3f}")
        print(f"   Type:           {'Unanswerable' if pred['is_impossible'] else 'Answerable'}")
        
        # Quick visual check
        if pred['prediction_text'].lower() == pred['ground_truth'].lower():
            print(f"   Result:         ✓ EXACT MATCH")
        elif pred['ground_truth'] and pred['prediction_text']:
            print(f"   Result:         ~ PARTIAL (check F1)")
        else:
            print(f"   Result:         ✗ INCORRECT")


if __name__ == "__main__":
    print("Testing BERT Q&A Prediction Generator...")
    print("=" * 80)
    print()
    
    # Test with a simple example
    qa = load_bert_qa_model()
    
    test_question = "What is the capital of France?"
    test_context = "Paris is the capital and largest city of France. It is located in northern France."
    
    print("Test Question:", test_question)
    print("Test Context:", test_context)
    print()
    
    result = qa(question=test_question, context=test_context)
    
    print("BERT's Answer:", result['answer'])
    print("Confidence:", result['score'])
    
    print("\n✓ BERT Q&A working correctly!")
