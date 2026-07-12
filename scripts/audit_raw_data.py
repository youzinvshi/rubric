import json
import hashlib
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def audit_rubricbench(path: Path) -> dict:
    if not path.exists():
        return {"status": "missing"}
    num_rows = 0
    missing_fields = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            num_rows += 1
            record = json.loads(line)
            # Check fields: case_id, instruction, rubrics
            has_id = "case_id" in record or "id" in record
            has_query = "instruction" in record or "query" in record or "prompt" in record
            has_gold = "rubrics" in record or "gold_rubrics" in record
            
            if not (has_id and has_query and has_gold):
                missing_fields += 1
            
            # Strict gate: if gold evaluation dimensions are missing, raise error
            if not has_gold:
                raise ValueError(f"Strict Gate Failed: RubricBench row {i} missing gold evaluation dimensions!")
            
            # Verify we can extract gold evaluation dimensions
            rubrics = record.get("rubrics") or record.get("gold_rubrics")
            if isinstance(rubrics, str):
                parsed = [r.strip() for r in rubrics.split("\n") if r.strip()]
                if not parsed:
                    raise ValueError(f"Strict Gate Failed: RubricBench row {i} gold evaluation dimensions extracted empty!")

    return {
        "status": "ok",
        "num_rows": num_rows,
        "missing_fields": missing_fields
    }

def audit_researchrubrics(path: Path) -> dict:
    if not path.exists():
        return {"status": "missing"}
    num_rows = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            num_rows += 1
            record = json.loads(line)
            has_query = any(k in record for k in ["instruction", "query", "prompt"])
            has_gold = any(k in record for k in ["rubrics", "gold_rubrics"])
            if not (has_query and has_gold):
                # We might log warning, but let's just count
                pass
    return {"status": "ok", "num_rows": num_rows}

def audit_downstream(path: Path) -> dict:
    if not path.exists():
        return {"status": "missing"}
    num_rows = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            num_rows += 1
            record = json.loads(line)
            # just ensuring it parses and has basic fields
    return {"status": "ok", "num_rows": num_rows}

SPLITS_DIR = Path("outputs/data_splits")


# Per-dataset governance describing how each source may enter the layered corpus.
# gold_type:                human_gold | proxy_teacher | (future) expert_gold | synthetic
# use_for:                  layered-corpus roles the source is allowed to fill
# allowed_in_main_bsc_eval: only clean human-gold may back the main BSC table
# contamination_risk:       reviewer-facing risk that this source leaks into eval
# split_policy:             holdout_test_main | non_overlap | full_holdout | none
# split_manifest:           split_dataset.py manifest that documents group-disjointness
DATASETS = {
    "rubricbench": {
        "file": "rubricbench_raw.jsonl",
        "has_hard_gold": True,
        "usage": ["bsc_eval", "rl_gold"],
        "auditor": audit_rubricbench,
        "gold_type": "human_gold",
        "use_for": ["main_bsc_eval", "rl_gold", "sft_seed"],
        "allowed_in_main_bsc_eval": True,
        "contamination_risk": "low",
        "split_policy": "holdout_test_main",
        "split_manifest": "rubricbench_gold_split.json",
        "notes": "Human-written atomic rubrics. Split into test_main holdout / dev / train_seed so the main BSC table is never trained on.",
    },
    "researchrubrics": {
        "file": "researchrubrics_raw.jsonl",
        "has_hard_gold": True,
        "usage": ["sft", "generalization"],
        "auditor": audit_researchrubrics,
        "gold_type": "human_gold",
        "use_for": ["generalization_eval", "sft_seed"],
        "allowed_in_main_bsc_eval": False,
        "contamination_risk": "low",
        "split_policy": "non_overlap",
        "split_manifest": "researchrubrics_gold_split.json",
        "notes": "High-quality research rubrics. Query-disjoint 60/41 split into train_seed and dev_test for complex-task generalization without leaking the evaluation subset into SFT seed data.",
    },
    "rewardbench": {
        "file": "rewardbench_filtered.jsonl",
        "has_hard_gold": False,
        "usage": ["sft_proxy_train", "downstream_holdout"],
        "auditor": audit_downstream,
        "gold_type": "proxy_teacher",
        "use_for": ["sft_proxy_train", "downstream_holdout"],
        "allowed_in_main_bsc_eval": False,
        "contamination_risk": "medium",
        "split_policy": "non_overlap",
        "split_manifest": "rewardbench_pref_split.json",
        "notes": "Pairwise preference prompts converted into evaluation-criteria elicitation tasks via multi-teacher generation + verifier filtering. 60/20/20 non-overlap: proxy train / dev / downstream holdout.",
    },
    "rewardbench2": {
        "file": "rewardbench2_test.jsonl",
        "has_hard_gold": False,
        "usage": ["downstream_holdout"],
        "auditor": audit_downstream,
        "gold_type": "proxy_teacher",
        "use_for": ["downstream_holdout"],
        "allowed_in_main_bsc_eval": False,
        "contamination_risk": "high",
        "split_policy": "full_holdout",
        "split_manifest": None,
        "notes": "Test-only split kept fully as downstream holdout; not used for training to avoid contaminating downstream judge evaluation.",
    },
    "judgebench": {
        "file": "judgebench_test.jsonl",
        "has_hard_gold": False,
        "usage": ["downstream_eval_hard"],
        "auditor": audit_downstream,
        "gold_type": "proxy_teacher",
        "use_for": ["downstream_holdout"],
        "allowed_in_main_bsc_eval": False,
        "contamination_risk": "high",
        "split_policy": "full_holdout",
        "split_manifest": None,
        "notes": "Hard judge cases kept fully as downstream holdout; not used for training.",
    },
}


def load_split_counts(split_manifest_name):
    """Read split_dataset.py manifest and return {split_name: {groups, records}}."""
    if not split_manifest_name:
        return {}
    path = SPLITS_DIR / split_manifest_name
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    counts = {}
    for name, info in data.get("splits", {}).items():
        counts[name] = {"groups": info.get("groups", 0), "records": info.get("records", 0)}
    counts["_group_disjoint"] = bool(data.get("group_disjoint", False))
    return counts


def assert_contamination_policy(manifest):
    """Fail-closed guard: human-gold sources may back the main BSC eval; proxy may not."""
    violations = []
    for name, entry in manifest.items():
        gold_type = entry.get("gold_type")
        allowed = entry.get("allowed_in_main_bsc_eval")
        if gold_type == "proxy_teacher" and allowed:
            violations.append(f"{name}: proxy_teacher must not be allowed_in_main_bsc_eval")
        if gold_type == "synthetic" and allowed:
            violations.append(f"{name}: synthetic must not be allowed_in_main_bsc_eval")
        if entry.get("has_hard_gold") and gold_type != "human_gold":
            violations.append(f"{name}: has_hard_gold implies gold_type=human_gold, got {gold_type}")
    if violations:
        raise ValueError("Contamination policy violated:\n  - " + "\n  - ".join(violations))


def main():
    manifest = {}
    report_lines = ["# Data Audit Report\n"]

    for name, info in DATASETS.items():
        path = RAW_DIR / info["file"]
        sha256 = sha256_file(path)
        audit_res = info["auditor"](path)
        split_counts = load_split_counts(info.get("split_manifest"))

        manifest[name] = {
            "path": str(path),
            "num_rows": audit_res.get("num_rows", 0),
            "sha256": sha256,
            "has_hard_gold": info["has_hard_gold"],
            "usage": info["usage"],
            "status": audit_res["status"],
            "gold_type": info["gold_type"],
            "use_for": info["use_for"],
            "allowed_in_main_bsc_eval": info["allowed_in_main_bsc_eval"],
            "contamination_risk": info["contamination_risk"],
            "split_policy": info["split_policy"],
            "split_manifest": (str(SPLITS_DIR / info["split_manifest"]) if info.get("split_manifest") else None),
            "split_counts": split_counts,
            "notes": info["notes"],
        }

        report_lines.append(f"## {name}")
        report_lines.append(f"- **Path**: `{path}`")
        report_lines.append(f"- **Status**: {audit_res['status']}")
        report_lines.append(f"- **Rows**: {audit_res.get('num_rows', 0)}")
        report_lines.append(f"- **Gold type**: {info['gold_type']}")
        report_lines.append(f"- **Use for**: {', '.join(info['use_for'])}")
        report_lines.append(f"- **Allowed in main BSC eval**: {info['allowed_in_main_bsc_eval']}")
        report_lines.append(f"- **Contamination risk**: {info['contamination_risk']}")
        report_lines.append(f"- **Split policy**: {info['split_policy']}")
        if split_counts:
            parts = [f"{k}={v['records']}r/{v['groups']}g" for k, v in split_counts.items() if not k.startswith("_")]
            report_lines.append(f"- **Splits**: {', '.join(parts)} (group_disjoint={split_counts.get('_group_disjoint')})")
        report_lines.append(f"- **SHA256**: `{sha256}`\n")

    # Fail-closed: refuse to emit a manifest that would let proxy data pose as human-gold.
    assert_contamination_policy(manifest)

    with open(PROCESSED_DIR / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(PROCESSED_DIR / "data_audit_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("Data audit completed successfully!")

if __name__ == "__main__":
    main()
