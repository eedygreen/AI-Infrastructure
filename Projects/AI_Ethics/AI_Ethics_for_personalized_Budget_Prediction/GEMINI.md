# GEMINI.md - Ethics in AI Project Context

## Project Overview
This project, titled **"Bias Evaluation and Ethical Audit,"** is focused on identifying and mitigating bias in an AI model designed for a hypothetical mobile app called **IDOOU**. The app is an "activity recommender," and the specific task is to predict a user's budget based on features like gender, age, and education level.

### Key Objectives
*   **Bias Detection:** Determine if certain groups (e.g., users with higher education) are privileged in the model's budget predictions.
*   **Ethical Audit:** Perform a systematic evaluation of the model's behavior, risks, and limitations.
*   **Explainability:** Apply techniques like LIME, Captum, and prompt sensitivity testing to understand model decisions.
*   **Mitigation:** Develop a plan to address identified biases before deployment.

### Main Technologies
*   **Languages:** Python
*   **Frameworks/Libraries:** 
    *   `AIF360` (AI Fairness 360) for fairness metrics.
    *   `PyTorch` & `Transformers` (Hugging Face) for model handling.
    *   `TensorFlow` & `Keras` (used in some parts of the environment).
    *   `Pandas`, `NumPy`, `Seaborn`, `Matplotlib` for data analysis and visualization.
    *   `LIME`, `Captum` for AI explainability.

## Building and Running

### Environment Setup
The project uses several dependency files. It is recommended to use a virtual environment.
```bash
# Using requirements.txt
pip install -r requirements.txt

# Or using required.txt (which contains more specific versions)
pip install -r required.txt
```
*Note: The project seems to have been developed in a environment with Python 3.7 or similar, though `pyproject.toml` mentions `>=3.14` (which might be a placeholder or for future-proofing).*

### Running the Analysis
The primary workspace is the Jupyter Notebook:
*   `AI Ethics Project -- STARTER.ipynb`: Contains the full lifecycle from data loading to bias evaluation.

To start the notebook:
```bash
jupyter lab AI\ Ethics\ Project\ --\ STARTER.ipynb
```

### Data
*   `udacity_ai_ethics_project_data.csv`: The synthetic dataset containing ~300,000 participant records.

## Development Conventions

### Workflow
1.  **Exploration:** Use the Jupyter Notebook to explore `udacity_ai_ethics_project_data.csv`.
2.  **Model Loading:** Load the fine-tuned model and tokenizer (usually from Hugging Face or a local path specified in the notebook).
3.  **Evaluation:** Run the fairness evaluation cells using `AIF360` metrics.
4.  **Audit:** Document findings in an **Ethical Audit Report**.
5.  **Mitigation:** Propose safeguards in a **Comprehensive Mitigation Plan**.

### Project Deliverables (Inferred from README)
*   **Ethical Audit Report:** Evidence-based observations of model behavior.
*   **Comprehensive Mitigation Plan:** Actionable safeguards.
*   **Ethics Committee Presentation:** High-level summary for non-technical stakeholders.

### Coding Style
*   Follow standard Python (PEP 8) conventions within scripts like `main.py`.
*   Maintain clear, documented cells in the Jupyter Notebook for reproducibility.
*   Prioritize transparency and explainability in all model evaluations.
