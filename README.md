# Schema-Aware Text-to-SQL with LoRA and RAG

This project evaluates two open-source instruction models for translating
natural-language questions into executable SQLite queries:

- CodeLlama-7B with a LoRA adapter on the Chinook database
- 4-bit Mistral-7B on Chinook and Spider
- FAISS retrieval using `all-MiniLM-L6-v2` embeddings
- exact-match and execution-match evaluation

The implementation is organized into reusable Python classes, concise Colab
notebooks, fixed dataset resolvers, and automated tests.

## Results

| Experiment | Test questions | Exact match | Execution match |
|---|---:|---:|---:|
| CodeLlama + LoRA + RAG, Chinook | 30 | 16.67% | 40.00% |
| Mistral baseline, Chinook | 30 | 20.00% | 56.67% |
| Mistral + RAG, Chinook | 30 | 26.67% | 66.67% |
| Mistral baseline, Spider | 108 | 4.63% | 22.22% |
| Mistral + corrected RAG, Spider | 108 | 12.96% | 29.63% |

Execution match is the primary metric because differently written SQL queries
can return the same correct rows.

## Repository structure

```text
Text-to-SQL-RAG-LoRA/
├── notebooks/      # Three runnable Colab experiments
├── src/            # Data, model, retrieval, generation and evaluation classes
├── datasets/       # Chinook and selected Spider SQLite databases
├── knowledge/      # Retrieval knowledge chunks
├── prompts/        # Mistral evaluation prompts
├── models/         # CodeLlama LoRA adapter
├── tests/          # Unit and integrity tests
└── results/        # Generated result JSON files
```

## Run in Colab

1. Upload and extract the project ZIP under `/content/project`.
2. Open a notebook from `notebooks/`.
3. Run the setup and dependency cells.
4. Start Mistral notebooks with `LIMIT = 3`; use `LIMIT = None` for the final run.
5. Generated JSON results are written to `results/`.

## Notebooks

- `01_CodeLlama_Chinook.ipynb`
- `02_Mistral_Chinook.ipynb`
- `03_Mistral_Spider.ipynb`

## Key engineering improvements

- Replaced repeated notebook functions with reusable classes.
- Removed hard-coded Google Drive paths.
- Added robust project and database path resolution.
- Standardized SQL extraction before evaluation.
- Corrected Spider retrieval to index complete knowledge chunks.
- Added smoke-test limits and automated validation.

## Notes

- CodeLlama training is optional because the LoRA adapter is included.
- Full-FP16 CodeLlama fine-tuning generally requires more memory than a free T4.
- Generated outputs can vary slightly with GPU, CUDA and library versions.

## Retriever evaluation metrics

The existing SQL-generation and RAG flow is unchanged. The notebooks now include a separate component-level retriever evaluation after the normal RAG evaluation.

Because the repository does not contain manual chunk-relevance labels, retrieval quality is measured against the table names required by each held-out gold SQL query:

- **Table Recall@K** — fraction of gold-required tables present in the Top-K retrieved context.
- **Table Precision@K** — fraction of distinct retrieved tables that are gold-required.
- **MRR** — reciprocal rank of the first retrieved chunk containing any gold-required table.

These are intentionally reported as table-grounded retrieval metrics rather than generic human-relevance metrics. Existing Exact Match and Execution Match remain the SQL-generation/end-to-end metrics.
