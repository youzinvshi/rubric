import json
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def convert_rubricbench():
    input_path = RAW_DIR / "rubricbench_raw.jsonl"
    output_path = PROCESSED_DIR / "rubricbench_bsc_eval.jsonl"
    if not input_path.exists():
        return

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if not line.strip(): continue
            record = json.loads(line)
            
            # Extract ID
            case_id = record.get("case_id") or record.get("id") or f"rubricbench_{i:06d}"
            
            # Extract Query
            query = record.get("instruction") or record.get("query") or record.get("prompt")
            
            # Extract Rubrics
            raw_rubrics = record.get("rubrics") or record.get("gold_rubrics")
            if isinstance(raw_rubrics, str):
                gold_rubrics = [r.strip() for r in raw_rubrics.split("\n") if r.strip()]
            elif isinstance(raw_rubrics, list):
                gold_rubrics = [str(r).strip() for r in raw_rubrics if str(r).strip()]
            else:
                gold_rubrics = []
                
            out_record = {
                "id": case_id,
                "source": "rubricbench",
                "query": query,
                "gold_rubrics": gold_rubrics,
                "split": "eval",
                "gold_type": "human"
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")

def convert_researchrubrics():
    input_path = RAW_DIR / "researchrubrics_raw.jsonl"
    output_path = PROCESSED_DIR / "researchrubrics_bsc_eval.jsonl"
    if not input_path.exists():
        return

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if not line.strip(): continue
            record = json.loads(line)
            
            case_id = record.get("case_id") or record.get("id") or f"researchrubrics_{i:06d}"
            query = record.get("instruction") or record.get("query") or record.get("prompt")
            
            raw_rubrics = record.get("rubrics") or record.get("gold_rubrics")
            if isinstance(raw_rubrics, str):
                gold_rubrics = [r.strip() for r in raw_rubrics.split("\n") if r.strip()]
            elif isinstance(raw_rubrics, list):
                gold_rubrics = [str(r).strip() for r in raw_rubrics if str(r).strip()]
            else:
                gold_rubrics = []
                
            out_record = {
                "id": case_id,
                "source": "researchrubrics",
                "query": query,
                "gold_rubrics": gold_rubrics,
                "split": "eval",
                "gold_type": "human"
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    convert_rubricbench()
    convert_researchrubrics()
    print("Converted RubricBench and ResearchRubrics to bsc_eval format.")
