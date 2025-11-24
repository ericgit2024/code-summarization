# SP-RAG Code Summarization Walkthrough

This guide provides step-by-step instructions to set up and run the SP-RAG (Structural Prompting + Retrieval-Augmented Generation) Code Summarization system.

## 1. Prerequisites

Ensure you have the following installed on your system:
*   **Python 3.8+**
*   **pip** (Python package installer)
*   **Git**
*   **Graphviz** (Required for dependency graph visualization)
    *   Ubuntu/Debian: `sudo apt-get install graphviz libgraphviz-dev`
    *   MacOS: `brew install graphviz`

## 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install Python Dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate

    # Install dependencies
    pip install -r requirements.txt
    ```

## 3. Data Preparation & Indexing

Before running inference or training, you need to build the vector index for Retrieval-Augmented Generation (RAG). This script loads the dataset, encodes code snippets, and builds a FAISS index.

```bash
python3 -m src.scripts.index_codebase
```
*   **Output:** Creates a `rag_index.pkl` file in the root directory.

## 4. Model Fine-tuning (Optional)

To fine-tune the LLM (Gemma or DeepSeek-Coder) on the dataset using QLoRA (4-bit quantization + LoRA):

```bash
python3 -m src.model.trainer
```
*   **Configuration:** You can modify `src/model/trainer.py` to adjust epochs, batch size, or the model ID.
*   **Output:** Saves the fine-tuned LoRA adapter to the `gemma_lora_finetuned/` directory.

## 5. Benchmarking

To evaluate the system's performance (BLEU, ROUGE, METEOR, Semantic Similarity) on a test set:

```bash
python3 -m src.scripts.benchmark
```
*   **Note:** This script uses the trained adapter if found; otherwise, it evaluates the base model.

## 6. Running the Interactive UI

The project includes a Streamlit application for easy interaction.

```bash
streamlit run src/ui/app.py
```
*   Open your browser at `http://localhost:8501`.
*   **Usage:**
    1.  Paste a Python code snippet or provide a GitHub Repository URL.
    2.  Click "Generate Summary".
    3.  View the generated summary and structural analysis placeholders.

## 7. Model Access & Configuration

### Setting up HuggingFace Authentication

The system uses the **Gemma model** (`google/gemma-2b`), which is a gated model on Hugging Face. You need to authenticate using an environment variable:

1. **Get your HuggingFace token:**
   - Go to [HuggingFace Settings > Access Tokens](https://huggingface.co/settings/tokens)
   - Create a new token or copy an existing one
   - Accept the Gemma model license at [google/gemma-2b](https://huggingface.co/google/gemma-2b)

2. **Set the environment variable:**
   
   **Windows PowerShell:**
   ```powershell
   $env:HF_TOKEN="your_token_here"
   ```
   
   **Linux/Mac:**
   ```bash
   export HF_TOKEN="your_token_here"
   ```
   
   **For persistent setup (recommended):**
   - Windows: Add to your PowerShell profile or set as a system environment variable
   - Linux/Mac: Add the export command to your `~/.bashrc` or `~/.zshrc`

3. **Verify the token is set:**
   ```bash
   # Windows PowerShell
   echo $env:HF_TOKEN
   
   # Linux/Mac
   echo $HF_TOKEN
   ```

## 8. Project Structure

*   `src/data/`: Dataset loading (`dataset.py`) and Prompt construction (`prompt.py`).
*   `src/structure/`: Tools to extract AST (`ast_utils.py`) and CFG (`graph_utils.py`).
*   `src/retrieval/`: RAG system logic (`rag.py`).
*   `src/model/`: Model loading (`model_loader.py`), training (`trainer.py`), and inference (`inference.py`).
*   `src/ui/`: Streamlit app (`app.py`) and visualization (`visualization.py`).
*   `src/utils/`: Helper scripts like metrics (`metrics.py`) and repo analysis (`repo_analysis.py`).

## 9. Known Limitations

*   **PDG Extraction:** The Program Dependence Graph (PDG) extraction is currently a placeholder. Full PDG extraction requires complex static analysis tools which are beyond the scope of this prototype. The system relies primarily on AST and CFG for structural prompting.
