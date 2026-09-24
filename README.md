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

## Comprehensive RAG evaluation added

The project now reports a broader deterministic evaluation suite **without changing the original retriever, prompts, model, or generation logic**. This makes it suitable for ablation studies of the improvement ideas used in this project.

### Retrieval quality

The old FAISS + `all-MiniLM-L6-v2` retriever can now be evaluated with:

- **Table Recall@K** — required-table coverage.
- **Table Precision@K** — how much retrieved table context is actually required.
- **Table F1@K** — balance of table precision and recall.
- **Hit Rate@K** — whether at least one relevant chunk is retrieved.
- **Full Table Coverage@K** — whether all required tables are retrieved.
- **MRR** — rank of the first relevant chunk.
- **MAP@K** — precision across relevant chunk ranks.
- **nDCG@K** — rank-sensitive retrieval relevance.

The chunk relevance label is deterministic: a chunk is relevant when it contains at least one table required by the held-out gold SQL. This is a table-grounded proxy because the repository has no manual chunk-relevance annotations.

### Generation / end-to-end quality

In addition to the original metrics:

- **Exact Match Accuracy**
- **Execution Match Accuracy**

both SQL evaluators now report:

- **SQL Validity Rate** — generated SQL executes successfully.
- **SQL Extraction Success Rate** — a non-empty SQL statement was extracted.
- **Single-Statement Compliance Rate** — output follows the one-query format expected by the prompt.
- **Mean Result-Set Precision / Recall / F1** — partial credit when the result is close but not an exact execution match.
- **Mean Table Selection Precision / Recall / F1** — whether generation selected the gold-required tables.

### Grounding / faithfulness proxies

For RAG runs the exact retrieved context used to generate each SQL query is retained in the result row. This enables:

- **Mean Context Grounding Score** — fraction of tables used by generated SQL that were present in retrieved context.
- **Fully Grounded Query Rate** — percentage of generated queries whose referenced tables are all supported by retrieved context.
- **Schema Compliance Rate** — generated SQL references only tables that exist in the target SQLite schema.

These are deterministic Text-to-SQL grounding proxies. They avoid using an LLM-as-a-judge for claims that can be verified directly from SQL and the database.

### Safety / guardrail metrics

For this project, classic natural-language toxicity is not a useful primary metric because the desired output is SQL. The relevant guardrail metrics are therefore:

- **Read-Only Compliance Rate** — percentage of generations restricted to `SELECT`/`WITH` queries.
- **Unsafe SQL Rate** — percentage containing mutating/destructive commands such as `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, or `PRAGMA`.

### Scope adherence

`src/scope_evaluation.py` adds metrics for a future scope classifier/router:

- **Scope Accuracy**
- **In-Scope Recall**
- **Out-of-Scope Detection Recall**
- **Scope Macro F1**
- **Over-Refusal Rate**
- **Under-Refusal Rate**

The current repository does **not** contain a labeled in-scope/out-of-scope benchmark, so the project intentionally does not fabricate numerical scope results. Add rows containing `expected_in_scope` and `predicted_in_scope` to evaluate a router when one is introduced.

### Mapping to the original improvement plan

- **Chunking / embeddings / reranking / K tuning:** compare Recall, Precision, F1, Hit Rate, Full Coverage, MRR, MAP, and nDCG.
- **Generator-model or system-prompt changes:** compare Exact Match, Execution Match, SQL Validity, result-set F1, extraction success, and single-statement compliance.
- **Guardrails / toxicity-related robustness:** compare Read-Only Compliance and Unsafe SQL Rate.
- **Scope classifier/router or query decomposition:** compare the optional scope metrics plus Schema Compliance and grounding metrics.

## One-command full evaluation of the old Chinook RAG

After installing `requirements.txt` on a CUDA Colab runtime, run:

```bash
python scripts/evaluate_old_mistral_chinook_full.py
```

For a quick smoke test first:

```bash
python scripts/evaluate_old_mistral_chinook_full.py --limit 3
```

The full run keeps the original Mistral + MiniLM + FAISS-L2 + `top_k=3` pipeline and writes all retrieval, generation, grounding, format, and guardrail metrics to:

```text
results/mistral_chinook_old_rag_complete_metrics.json
```

<!-- EXPANDED_EVAL_START -->

## Comprehensive RAG Evaluation

The final **Mistral-7B-Instruct-v0.2 + RAG** pipeline was evaluated on
**30 held-out Chinook Text-to-SQL questions**.

### Retrieval Metrics

| Metric | Score |
|---|---:|
| Table Recall@3 | **96.67%** |
| Table Precision@3 | **54.33%** |
| Table F1@3 | **66.48%** |
| Hit Rate@3 | **96.67%** |
| Full Table Coverage@3 | **96.67%** |
| MRR | **95.00%** |
| MAP@3 | **85.37%** |
| nDCG@3 | **88.44%** |

### Text-to-SQL Generation Metrics

| Metric | Score |
|---|---:|
| Exact Match Accuracy | **30.00%** |
| Execution Match Accuracy | **66.67%** |
| SQL Validity Rate | **96.67%** |
| SQL Extraction Success Rate | **100.00%** |
| Schema Compliance Rate | **100.00%** |
| Table Selection Precision | **86.67%** |
| Table Selection Recall | **96.67%** |
| Table Selection F1 | **89.89%** |

### Grounding and Safety Metrics

| Metric | Score |
|---|---:|
| Context Grounding Score | **95.00%** |
| Fully Grounded Query Rate | **93.33%** |
| Single-Statement Compliance | **100.00%** |
| Read-Only Compliance | **100.00%** |
| Unsafe SQL Rate | **0.00%** |

**Execution Match Accuracy (66.67%)** is used as the primary end-to-end
measure of generated SQL correctness.

The retrieval results show strong table coverage and ranking quality,
while generation remains the main source of remaining end-to-end errors.

<!-- EXPANDED_EVAL_END -->
