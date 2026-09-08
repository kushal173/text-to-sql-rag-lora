import json
import unittest
from pathlib import Path

from src.data import ChinookData
from src.evaluation import SQLEvaluator
from src.mistral import MistralPromptBuilder, MistralRunner
from src.mistral_evaluation import MistralSQLEvaluator
from src.prompts import PromptBuilder
from src.rag import ChinookRetriever


ROOT = Path(__file__).parents[1]


class ProjectTests(unittest.TestCase):
    def test_chinook_assets(self):
        data = ChinookData(ROOT / "datasets/chinook")
        self.assertEqual(len(data.load_train()), 88)
        self.assertEqual(len(data.load_test()), 30)
        self.assertGreater(
            (ROOT / "models/codellama_chinook_lora/adapter_model.safetensors").stat().st_size,
            1_000_000,
        )

    def test_codellama_prompts(self):
        self.assertTrue(PromptBuilder.finetune("Question?").endswith("### Response:\n"))
        self.assertTrue(
            PromptBuilder.schema_or_rag("Question?", "schema").endswith("### SQL:")
        )

    def test_codellama_corpus(self):
        data = ChinookData(ROOT / "datasets/chinook")
        corpus = ChinookRetriever.build_corpus(
            data.schema_from_training(), data.load_train()
        )
        self.assertEqual(corpus[0], "Schema:")
        self.assertEqual(len(corpus), 62)

    def test_sql_extraction_and_execution(self):
        raw = "SELECT ArtistId FROM Album GROUP BY ArtistId; explanation"
        clean = SQLEvaluator.extract_first_statement(raw)
        self.assertEqual(clean, "SELECT ArtistId FROM Album GROUP BY ArtistId;")
        result = SQLEvaluator(ROOT / "datasets/chinook/chinook.db").compare(
            clean, "SELECT ArtistId FROM Album GROUP BY ArtistId;"
        )
        self.assertTrue(result["execution_match"])

    def test_mistral_prompt_and_extractor(self):
        prompt = MistralPromptBuilder.baseline("schema", "question")
        self.assertTrue(prompt.endswith("\n ### SQL:"))
        raw = prompt + "\nSELECT COUNT(*) FROM Track; explanation"
        self.assertEqual(
            MistralRunner.extract_sql(raw), "SELECT COUNT(*) FROM Track;"
        )

    def test_spider_assets(self):
        evaluator = MistralSQLEvaluator(ROOT / "datasets")
        prompts = json.loads(
            (ROOT / "prompts/Mistral7b/spider_prompts.json").read_text()
        )
        self.assertEqual(len(prompts), 108)
        for db_id in {row["db_id"] for row in prompts}:
            self.assertTrue(evaluator.db_path("spider", db_id).is_file())

    def test_metrics_summary(self):
        metrics = json.loads((ROOT / "results/metrics_summary.json").read_text())
        self.assertEqual(metrics["Mistral_Chinook_RAG"]["n"], 30)
        self.assertEqual(metrics["Mistral_Spider_RAG"]["n"], 108)


    def test_retrieval_table_extraction_and_metrics(self):
        from src.retrieval_evaluation import evaluate_retriever, extract_sql_tables

        self.assertEqual(
            extract_sql_tables(
                "SELECT a.Name FROM Artist a JOIN Album b ON a.ArtistId=b.ArtistId"
            ),
            {"artist", "album"},
        )

        class DummyRetriever:
            def retrieve(self, question, top_k=3):
                return [
                    "Table: Artist\nColumns: ArtistId, Name",
                    "Table: Customer\nColumns: CustomerId, Country",
                    "Table: Album\nColumns: AlbumId, ArtistId",
                ][:top_k]

        examples = [{
            "question": "Which artist has the most albums?",
            "gold_sql": (
                "SELECT Artist.Name FROM Artist JOIN Album "
                "ON Artist.ArtistId=Album.ArtistId"
            ),
        }]
        rows, metrics = evaluate_retriever(examples, DummyRetriever(), top_k=3)
        self.assertEqual(metrics["Table Recall@3"], 1.0)
        self.assertEqual(metrics["Table Precision@3"], round(2 / 3, 4))
        self.assertEqual(metrics["MRR"], 1.0)
        self.assertEqual(rows[0]["gold_tables"], ["album", "artist"])


if __name__ == "__main__":
    unittest.main()
