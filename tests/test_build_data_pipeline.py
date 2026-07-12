from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.build_data_pipeline import build_manifest, build_pipeline, load_config, scope_config


class BuildDataPipelineTest(unittest.TestCase):
    def test_load_config_reports_missing_file(self) -> None:
        with self.assertRaises(SystemExit) as context:
            load_config(Path("/tmp/missing_data_sources_config.json"))

        self.assertIn("Data pipeline config is missing", str(context.exception))

    def test_load_config_reports_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{bad", encoding="utf-8")

            with self.assertRaises(SystemExit) as context:
                load_config(path)

        self.assertIn("not valid JSON", str(context.exception))

    def test_load_config_requires_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "list.json"
            path.write_text("[]", encoding="utf-8")

            with self.assertRaises(SystemExit) as context:
                load_config(path)

        self.assertIn("must be a JSON object", str(context.exception))

    def test_build_pipeline_adds_hf_download_and_normalize(self) -> None:
        pipeline = build_pipeline(sample_config())
        stages = pipeline["stages"]
        self.assertEqual(stages[0]["type"], "download")
        self.assertEqual(stages[0]["args"]["preset"], "rewardbench")
        self.assertEqual(stages[1]["type"], "profile_data")
        self.assertEqual(stages[2]["type"], "normalize")
        self.assertEqual(stages[2]["args"]["target"], "preference")

    def test_build_pipeline_adds_url_download(self) -> None:
        config = sample_config()
        config["datasets"][0]["source"] = {
            "type": "url",
            "url": "https://example.com/rewardbench.jsonl",
            "output": "data/raw/rewardbench.jsonl",
            "limit": 10,
        }
        pipeline = build_pipeline(config)
        stage = pipeline["stages"][0]

        self.assertEqual(stage["type"], "download")
        self.assertEqual(stage["args"]["url"], "https://example.com/rewardbench.jsonl")
        self.assertEqual(stage["args"]["output"], "data/raw/rewardbench.jsonl")
        self.assertEqual(stage["args"]["limit"], 10)

    def test_build_pipeline_skips_manual_download(self) -> None:
        pipeline = build_pipeline(sample_config())
        manual_stages = [stage for stage in pipeline["stages"] if "rubricbench" in stage["name"]]
        self.assertEqual(len(manual_stages), 2)
        self.assertEqual(manual_stages[0]["type"], "profile_data")
        self.assertEqual(manual_stages[1]["type"], "normalize")

    def test_build_pipeline_can_add_explicit_manual_download(self) -> None:
        config = sample_config()
        config["datasets"][1]["source"]["official_url"] = "https://huggingface.co/datasets/org/researchrubrics/resolve/main/data.jsonl"
        config["datasets"][1]["source"]["download_enabled"] = True

        pipeline = build_pipeline(config)
        stages = [stage for stage in pipeline["stages"] if "rubricbench" in stage["name"]]

        self.assertEqual(stages[0]["type"], "download")
        self.assertEqual(
            stages[0]["args"]["url"],
            "https://huggingface.co/datasets/org/researchrubrics/resolve/main/data.jsonl",
        )
        self.assertEqual(stages[0]["args"]["output"], "data/raw/rubricbench.jsonl")

    def test_scope_config_filters_datasets_and_sets_required_scope(self) -> None:
        config = sample_config()
        config["local_config_init"] = {"enabled": True, "template": "configs/data_sources_real.template.json"}
        config["post_download_local_config_init"] = {
            "enabled": True,
            "template": "configs/data_sources_real.template.json",
        }
        config["pre_normalization_source_report"] = {"enabled": True}
        config["source_report"] = {"enabled": True}
        config["datasets"][1]["name"] = "researchrubrics"
        config["datasets"][1]["source"]["official_url"] = "https://huggingface.co/datasets/org/researchrubrics/resolve/main/data.jsonl"
        config["datasets"][1]["source"]["download_enabled"] = True

        scoped = scope_config(config, ["researchrubrics"])
        pipeline = build_pipeline(scoped)
        names = [stage["name"] for stage in pipeline["stages"]]

        self.assertEqual([dataset["name"] for dataset in scoped["datasets"]], ["researchrubrics"])
        self.assertIn("download_researchrubrics", names)
        self.assertNotIn("download_rewardbench", names)
        self.assertNotIn("profile_rubricbench", names)
        init_stage = next(stage for stage in pipeline["stages"] if stage["name"] == "init_data_source_local_config")
        post_stage = next(stage for stage in pipeline["stages"] if stage["name"] == "init_data_source_local_config_post_download")
        pre_report_stage = next(
            stage for stage in pipeline["stages"] if stage["name"] == "data_source_report_pre_normalization"
        )
        report_stage = next(stage for stage in pipeline["stages"] if stage["name"] == "data_source_report")
        self.assertEqual(init_stage["args"]["required_dataset"], ["researchrubrics"])
        self.assertEqual(post_stage["args"]["required_dataset"], ["researchrubrics"])
        self.assertEqual(pre_report_stage["args"]["required_dataset"], ["researchrubrics"])
        self.assertEqual(report_stage["args"]["required_dataset"], ["researchrubrics"])
        self.assertEqual(
            init_stage["args"]["report_json"],
            "outputs/data_sources/local_config_init_researchrubrics.json",
        )
        self.assertEqual(
            post_stage["args"]["report_json"],
            "outputs/data_sources/local_config_post_download_researchrubrics.json",
        )
        self.assertEqual(
            pre_report_stage["args"]["output_json"],
            "outputs/data_sources/source_report_pre_normalization_researchrubrics.json",
        )
        self.assertEqual(
            report_stage["args"]["output_json"],
            "outputs/data_sources/source_report_researchrubrics.json",
        )

    def test_scope_config_reports_missing_dataset(self) -> None:
        with self.assertRaises(SystemExit) as context:
            scope_config(sample_config(), ["missing"])

        self.assertIn("missing requested dataset", str(context.exception))

    def test_build_manifest_contains_raw_and_processed_files(self) -> None:
        manifest = build_manifest(sample_config())
        self.assertIn("data/raw/rewardbench.jsonl", manifest["required_files"])
        self.assertIn("outputs/data_profiles/rewardbench.json", manifest["required_files"])
        self.assertIn("data/processed/rewardbench_pref.jsonl", manifest["required_files"])
        self.assertIn("data/raw/rubricbench.jsonl", manifest["required_files"])

    def test_build_pipeline_can_add_source_report_stage(self) -> None:
        config = sample_config()
        config["source_report"] = {
            "enabled": True,
            "config": "configs/data_sources.local.json",
            "output_json": "outputs/source_report.json",
            "output_md": "outputs/source_report.md",
            "required_datasets": ["rubricbench"],
            "strict": True,
        }
        pipeline = build_pipeline(config)
        names = [stage["name"] for stage in pipeline["stages"]]
        source_idx = names.index("data_source_report")
        self.assertLess(names.index("download_rewardbench"), source_idx)
        self.assertLess(names.index("profile_rewardbench"), source_idx)
        stage = pipeline["stages"][source_idx]
        self.assertEqual(stage["args"]["config"], "configs/data_sources.local.json")
        self.assertEqual(stage["args"]["required_dataset"], ["rubricbench"])
        self.assertTrue(stage["args"]["strict"])

    def test_build_pipeline_can_add_pre_normalization_source_report_stage(self) -> None:
        config = sample_config()
        config["pre_normalization_source_report"] = {
            "enabled": True,
            "config": "configs/data_sources.local.json",
            "output_json": "outputs/source_report_pre.json",
            "output_md": "outputs/source_report_pre.md",
            "required_datasets": ["rubricbench"],
            "strict": True,
        }

        pipeline = build_pipeline(config)
        names = [stage["name"] for stage in pipeline["stages"]]
        source_idx = names.index("data_source_report_pre_normalization")
        self.assertLess(names.index("download_rewardbench"), source_idx)
        self.assertLess(source_idx, names.index("profile_rewardbench"))
        stage = pipeline["stages"][source_idx]
        self.assertEqual(stage["args"]["output_json"], "outputs/source_report_pre.json")
        self.assertEqual(stage["args"]["required_dataset"], ["rubricbench"])

        manifest = build_manifest(config)
        self.assertIn("outputs/source_report_pre.json", manifest["required_files"])

    def test_build_pipeline_can_add_local_config_init_stage(self) -> None:
        config = sample_config()
        config["local_config_init"] = {
            "enabled": True,
            "template": "configs/data_sources_real.template.json",
            "output": "configs/data_sources_real.local.json",
            "report_json": "outputs/data_sources/local_config_init.json",
            "report_md": "outputs/data_sources/local_config_init.md",
            "required_datasets": ["rubricbench"],
        }
        pipeline = build_pipeline(config)

        self.assertEqual(pipeline["stages"][0]["type"], "init_data_source_config")
        self.assertEqual(pipeline["stages"][0]["args"]["template"], "configs/data_sources_real.template.json")
        self.assertEqual(pipeline["stages"][0]["args"]["report_json"], "outputs/data_sources/local_config_init.json")
        self.assertEqual(pipeline["stages"][0]["args"]["required_dataset"], ["rubricbench"])

        manifest = build_manifest(config)
        self.assertIn("configs/data_sources_real.local.json", manifest["required_files"])
        self.assertIn("outputs/data_sources/local_config_init.json", manifest["required_files"])

    def test_build_pipeline_can_add_post_download_local_config_init_stage(self) -> None:
        config = sample_config()
        config["local_config_init"] = {
            "enabled": True,
            "template": "configs/data_sources_real.template.json",
            "output": "configs/data_sources_real.local.json",
            "report_json": "outputs/data_sources/local_config_init.json",
        }
        config["post_download_local_config_init"] = {
            "enabled": True,
            "template": "configs/data_sources_real.template.json",
            "output": "configs/data_sources_real.local.json",
            "report_json": "outputs/data_sources/local_config_post_download.json",
            "fill_present_sha256": True,
            "update_existing": True,
        }
        config["source_report"] = {"enabled": True}

        pipeline = build_pipeline(config)
        names = [stage["name"] for stage in pipeline["stages"]]
        post_idx = names.index("init_data_source_local_config_post_download")
        self.assertLess(names.index("download_rewardbench"), post_idx)
        self.assertLess(post_idx, names.index("data_source_report"))
        post_stage = pipeline["stages"][post_idx]
        self.assertTrue(post_stage["args"]["fill_present_sha256"])
        self.assertTrue(post_stage["args"]["update_existing"])

        manifest = build_manifest(config)
        self.assertIn("outputs/data_sources/local_config_post_download.json", manifest["required_files"])

    def test_build_manifest_contains_source_report_output(self) -> None:
        config = sample_config()
        config["source_report"] = {
            "enabled": True,
            "output_json": "outputs/source_report.json",
        }
        manifest = build_manifest(config)

        self.assertIn("outputs/source_report.json", manifest["required_files"])

    def test_build_pipeline_can_add_gold_validation_stage(self) -> None:
        config = sample_config()
        config["datasets"][1]["source"]["official_url"] = "https://official.example/rubricbench.jsonl"
        config["datasets"][1]["source"]["require_official_url"] = True
        config["datasets"][1]["normalizations"][0]["validation"] = {
            "enabled": True,
            "min_records": 10,
            "require_provenance": True,
            "required_data_source": ["rubricbench"],
            "forbidden_data_source": ["toy"],
            "output_json": "outputs/validation/rubricbench_gold.json",
            "strict": True,
        }
        pipeline = build_pipeline(config)
        stages = [stage for stage in pipeline["stages"] if "rubricbench" in stage["name"]]
        self.assertEqual(stages[-1]["type"], "validate_gold")
        self.assertEqual(stages[-1]["args"]["min_records"], 10)
        self.assertTrue(stages[-1]["args"]["require_provenance"])
        self.assertEqual(stages[-1]["args"]["required_data_source"], ["rubricbench"])
        self.assertEqual(stages[-1]["args"]["forbidden_data_source"], ["toy"])
        self.assertEqual(
            stages[-1]["args"]["required_provenance"],
            [
                "paper_url=https://arxiv.org/abs/2603.01562",
                "source_url=https://official.example/rubricbench.jsonl",
            ],
        )
        self.assertTrue(stages[-1]["args"]["strict"])

        manifest = build_manifest(config)
        self.assertIn("outputs/validation/rubricbench_gold.json", manifest["required_files"])

    def test_build_pipeline_passes_gold_provenance_args_to_normalizer(self) -> None:
        config = sample_config()
        config["datasets"][1]["source"]["official_url"] = "https://official.example/rubricbench.jsonl"
        config["datasets"][1]["source"]["require_official_url"] = True
        pipeline = build_pipeline(config)
        stage = next(item for item in pipeline["stages"] if item["name"] == "normalize_rubricbench_gold_1")

        self.assertEqual(stage["args"]["source_url"], "https://official.example/rubricbench.jsonl")
        self.assertEqual(stage["args"]["paper_url"], "https://arxiv.org/abs/2603.01562")
        self.assertEqual(stage["args"]["dataset_version"], "v1")

    def test_build_pipeline_can_add_query_disjoint_split_stage(self) -> None:
        config = sample_config()
        config["datasets"][1]["normalizations"][0]["split"] = {
            "enabled": True,
            "output_dir": "data/processed/splits",
            "output_prefix": "rubricbench_gold_",
            "manifest": "outputs/data_splits/rubricbench_gold_split.json",
            "splits": ["train_seed:50", "dev:20", "test_main:rest"],
            "group_key": "query",
            "stratify_key": "data_source",
            "gold_type": "human_gold",
            "main_eval_split": ["test_main"],
            "seed": 13,
            "outputs": [
                "data/processed/splits/rubricbench_gold_train_seed.jsonl",
                "data/processed/splits/rubricbench_gold_dev.jsonl",
                "data/processed/splits/rubricbench_gold_test_main.jsonl",
            ],
        }

        pipeline = build_pipeline(config)
        names = [stage["name"] for stage in pipeline["stages"]]
        normalize_stage = pipeline["stages"][names.index("normalize_rubricbench_gold_1")]
        split_stage = pipeline["stages"][names.index("split_rubricbench_gold_1")]

        self.assertLess(names.index("normalize_rubricbench_gold_1"), names.index("split_rubricbench_gold_1"))
        self.assertNotIn("split", normalize_stage["args"])
        self.assertEqual(split_stage["type"], "split_dataset")
        self.assertEqual(split_stage["args"]["input"], "data/processed/rubricbench_gold.jsonl")
        self.assertEqual(split_stage["args"]["split"], ["train_seed:50", "dev:20", "test_main:rest"])
        self.assertEqual(split_stage["args"]["output_dir"], "data/processed/splits")
        self.assertEqual(split_stage["args"]["output_prefix"], "rubricbench_gold_")
        self.assertEqual(split_stage["args"]["manifest"], "outputs/data_splits/rubricbench_gold_split.json")
        self.assertEqual(split_stage["args"]["main_eval_split"], ["test_main"])

        manifest = build_manifest(config)
        self.assertIn("outputs/data_splits/rubricbench_gold_split.json", manifest["required_files"])
        self.assertIn("data/processed/splits/rubricbench_gold_train_seed.jsonl", manifest["required_files"])
        self.assertIn("data/processed/splits/rubricbench_gold_test_main.jsonl", manifest["required_files"])

    def test_build_pipeline_passes_query_pool_provenance_args_to_normalizer(self) -> None:
        config = sample_config()
        config["datasets"][1]["source"]["official_url"] = "https://official.example/rubricbench.jsonl"
        config["datasets"][1]["source"]["require_official_url"] = True
        config["datasets"][1]["source"]["paper_url"] = "https://arxiv.org/abs/2603.01562"
        config["datasets"][1]["normalizations"].append(
            {
                "target": "query_pool",
                "output": "data/processed/rubricbench_queries.jsonl",
                "data_source": "rubricbench",
                "dedupe_query": True,
            }
        )
        pipeline = build_pipeline(config)
        stage = next(item for item in pipeline["stages"] if item["name"] == "normalize_rubricbench_query_pool_2")

        self.assertEqual(stage["args"]["source_url"], "https://official.example/rubricbench.jsonl")
        self.assertEqual(stage["args"]["paper_url"], "https://arxiv.org/abs/2603.01562")

    def test_build_pipeline_can_add_query_pool_validation_stage(self) -> None:
        config = sample_config()
        config["datasets"][1]["source"]["official_url"] = "https://official.example/rubricbench.jsonl"
        config["datasets"][1]["source"]["require_official_url"] = True
        config["datasets"][1]["source"]["paper_url"] = "https://arxiv.org/abs/2603.01562"
        config["datasets"][1]["normalizations"].append(
            {
                "target": "query_pool",
                "output": "data/processed/rubricbench_queries.jsonl",
                "data_source": "rubricbench",
                "validation": {
                    "enabled": True,
                    "min_records": 10,
                    "require_provenance": True,
                    "required_data_source": ["rubricbench"],
                    "forbidden_data_source": ["toy", "proxy"],
                    "output_json": "outputs/validation/rubricbench_queries.json",
                    "strict": True,
                },
            }
        )
        pipeline = build_pipeline(config)
        stages = [stage for stage in pipeline["stages"] if "rubricbench_query_pool" in stage["name"]]

        self.assertEqual(stages[-1]["type"], "validate_gold")
        self.assertEqual(stages[-1]["args"]["target"], "query_pool")
        self.assertEqual(stages[-1]["args"]["min_records"], 10)
        self.assertEqual(stages[-1]["args"]["required_data_source"], ["rubricbench"])
        self.assertEqual(stages[-1]["args"]["forbidden_data_source"], ["toy", "proxy"])
        self.assertEqual(
            stages[-1]["args"]["required_provenance"],
            [
                "paper_url=https://arxiv.org/abs/2603.01562",
                "source_url=https://official.example/rubricbench.jsonl",
            ],
        )
        self.assertTrue(stages[-1]["args"]["strict"])

        manifest = build_manifest(config)
        self.assertIn("outputs/validation/rubricbench_queries.json", manifest["required_files"])

    def test_query_pool_validation_defaults_use_queries_output_stem(self) -> None:
        config = sample_config()
        config["datasets"][1]["normalizations"].append(
            {
                "target": "query_pool",
                "output": "data/processed/rubricbench_queries.jsonl",
                "data_source": "rubricbench",
                "validation": {
                    "enabled": True,
                    "min_records": 10,
                },
            }
        )
        pipeline = build_pipeline(config)
        stage = next(item for item in pipeline["stages"] if item["name"] == "validate_rubricbench_query_pool_2")

        self.assertEqual(stage["args"]["target"], "query_pool")
        self.assertEqual(stage["args"]["output_json"], "outputs/data_validation/rubricbench_queries.json")
        self.assertEqual(stage["args"]["output_md"], "outputs/data_validation/rubricbench_queries.md")

        manifest = build_manifest(config)
        self.assertIn("outputs/data_validation/rubricbench_queries.json", manifest["required_files"])
        self.assertIn("outputs/data_validation/rubricbench_queries.md", manifest["required_files"])

    def test_build_pipeline_keeps_explicit_source_url_over_source_config(self) -> None:
        config = sample_config()
        config["datasets"][1]["source"]["official_url"] = "https://official.example/rubricbench.jsonl"
        config["datasets"][1]["source"]["require_official_url"] = True
        config["datasets"][1]["normalizations"][0]["source_url"] = "https://mirror.example/rubricbench.jsonl"
        pipeline = build_pipeline(config)
        stage = next(item for item in pipeline["stages"] if item["name"] == "normalize_rubricbench_gold_1")

        self.assertEqual(stage["args"]["source_url"], "https://mirror.example/rubricbench.jsonl")
        self.assertEqual(stage["args"]["paper_url"], "https://arxiv.org/abs/2603.01562")
        self.assertEqual(stage["args"]["dataset_version"], "v1")

    def test_build_pipeline_can_add_schema_contract_stage(self) -> None:
        config = sample_config()
        config["datasets"][0]["schema_contract"] = {
            "target": ["preference", "multicandidate"],
            "output_json": "outputs/schema/rewardbench.json",
            "output_md": "outputs/schema/rewardbench.md",
            "min_records": 10,
        }

        pipeline = build_pipeline(config)
        stage = next(item for item in pipeline["stages"] if item["type"] == "schema_contract")
        self.assertEqual(stage["args"]["input"], "data/raw/rewardbench.jsonl")
        self.assertEqual(stage["args"]["target"], ["preference", "multicandidate"])
        self.assertEqual(stage["args"]["min_records"], 10)

        manifest = build_manifest(config)
        self.assertIn("outputs/schema/rewardbench.json", manifest["required_files"])


def sample_config():
    return {
        "datasets": [
            {
                "name": "rewardbench",
                "source": {
                    "type": "hf",
                    "preset": "rewardbench",
                    "output": "data/raw/rewardbench.jsonl",
                },
                "profile": {"output": "outputs/data_profiles/rewardbench.json"},
                "normalizations": [
                    {
                        "target": "preference",
                        "output": "data/processed/rewardbench_pref.jsonl",
                        "dedupe_query": True,
                    }
                ],
            },
            {
                "name": "rubricbench",
                "source": {
                    "type": "manual",
                    "raw_path": "data/raw/rubricbench.jsonl",
                },
                "profile": {"output": "outputs/data_profiles/rubricbench.json"},
                "normalizations": [
                    {
                        "target": "gold",
                        "output": "data/processed/rubricbench_gold.jsonl",
                        "paper_url": "https://arxiv.org/abs/2603.01562",
                        "dataset_version": "v1",
                    }
                ],
            },
        ]
    }


if __name__ == "__main__":
    unittest.main()
