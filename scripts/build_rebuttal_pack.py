#!/usr/bin/env python3
"""Generate a reviewer-facing rebuttal pack from Evidence Matrix outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from scripts.build_result_card import build_claim_ladder_status
except ModuleNotFoundError:  # pragma: no cover - exercised by direct script execution
    from build_result_card import build_claim_ladder_status


DEFENSE_STATUS_KEYS = [
    "answer_ready",
    "needs_readiness",
    "needs_evidence",
    "cannot_claim",
    "missing_claim_mapping",
]


DEFAULT_CONCERNS = [
    {
        "id": "R1",
        "topic": "Motivation",
        "question": "How do we know a single-model evaluation-criteria policy really has blind spots?",
        "claim_sections": ["Motivation"],
        "fallback": "Present this as a hypothesis until hard-gold RubricBench evidence passes.",
    },
    {
        "id": "R2",
        "topic": "Metric Robustness",
        "question": "Is BSC sensitive to the semantic matching threshold, or is the embedding threshold arbitrary?",
        "claim_sections": ["Robustness"],
        "fallback": (
            "Do not tune thresholds on RubricBench test_main; report the fixed BGE protocol "
            "(coverage_tau=0.75, redundancy_tau=0.85), threshold sweeps, paired bootstrap CIs, "
            "and C6 strict-gate human-audit summaries before treating BSC alignment as validated."
        ),
    },
    {
        "id": "R3",
        "topic": "Reward Hacking",
        "question": "Is the RL-stage coverage change still reportable under redundancy and hallucination controls?",
        "claim_sections": ["Main Results"],
        "keywords": ["redundancy", "hallucination", "simply increasing"],
        "fallback": (
            "Keep reward-hacking mitigation as method design unless Red/Hall evidence passes; "
            "BSC-only coverage changes are metric results, not judge-utility claims."
        ),
    },
    {
        "id": "R4",
        "topic": "Downstream Utility",
        "question": "Is higher evaluation-dimension coverage supported by held-out preference-judging utility?",
        "claim_sections": ["Downstream"],
        "fallback": (
            "Frame downstream utility as unclaimed unless RewardBench, JudgeBench, and RewardBench-2 "
            "pass with fixed API/model scorer outputs, paper_claim_eligible summaries, and SHA-bound "
            "input/provider/budget contracts."
        ),
    },
    {
        "id": "R5",
        "topic": "Multi-Teacher Data",
        "question": "Is multi-teacher union empirically supported over the strongest single teacher?",
        "claim_sections": ["Ablation"],
        "keywords": ["Multi-teacher", "single teacher", "union"],
        "fallback": (
            "Use multi-teacher union as an uncontrolled data-construction choice until C5 supports it over "
            "the strongest single teacher under the same hard-gold BSC protocol."
        ),
    },
    {
        "id": "R6",
        "topic": "RL Stage Necessity",
        "question": "Does GRPO/RLVR add anything beyond SFT-only imitation?",
        "claim_sections": ["Ablation"],
        "keywords": ["SFT-only", "SFT+GRPO", "RL-stage"],
        "fallback": (
            "Report SFT-only parity as evidence that the RL stage is not supported unless C14 "
            "passes under the same BGE, verifier, threshold, and bootstrap-CI protocol."
        ),
    },
    {
        "id": "R7",
        "topic": "BSC Validity",
        "question": "Is BSC a self-defined metric that only optimizes the authors' own score?",
        "claim_sections": ["Robustness", "Downstream"],
        "fallback": (
            "Treat BSC as a proposed semantic diagnostic until it is bound to human-gold dimensions, "
            "fixed BGE/threshold protocols, C6-passing matched/unmatched human-audit summaries, "
            "and held-out RewardBench/JudgeBench/RewardBench-2 utility evidence."
        ),
    },
    {
        "id": "R8",
        "topic": "Data Contamination",
        "question": "Could training, proxy-gold construction, verifier calibration, or checkpoint selection contaminate evaluation?",
        "claim_sections": ["Data Hygiene"],
        "fallback": (
            "Do not answer contamination concerns from narrative alone; require C0 safe_to_claim with "
            "RubricBench test_main hard-gold holdout, query-disjoint proxy/SFT/RL artifacts, downstream "
            "holdout audits, and SHA-bound training/evaluation provenance."
        ),
    },
    {
        "id": "R9",
        "topic": "Verifier Bias",
        "question": "Does the verifier control which criteria survive and therefore determine the result?",
        "claim_sections": ["Ablation"],
        "keywords": ["verifier"],
        "fallback": (
            "Keep verifier effects as a method-risk discussion until C7 passes with rule/API verifier "
            "evidence, no-verifier ablation, and C6-gated human-audit checks for matched/unmatched BSC pairs."
        ),
    },
    {
        "id": "R10",
        "topic": "Human Audit",
        "question": "Can automated BSC matches be trusted without completed human validation?",
        "claim_sections": ["Robustness"],
        "keywords": ["auditable"],
        "fallback": (
            "Report annotation-pack readiness only as preparation; do not call BSC alignment validated until "
            "human-audit summaries pass C6 strict gates for enough labels, invalid labels, uncertain "
            "rate, auto-matched agreement, and auto-unmatched confirmation."
        ),
    },
    {
        "id": "R11",
        "topic": "Open-Ended RLVR Transfer",
        "question": "What makes this evidence for RLVR on open-ended semantic criteria elicitation rather than prompt tuning or SFT imitation?",
        "claim_sections": ["Main Results", "Ablation"],
        "keywords": ["GRPO", "RLVR", "SFT-only"],
        "fallback": (
            "Keep the RLVR-transfer claim as a testable hypothesis until C14 supports SFT+GRPO over "
            "SFT-only under the same BGE/verifier/threshold/bootstrap protocol and C7 confirms the "
            "relevant trained reward-component ablations."
        ),
    },
]


def main() -> None:
    args = parse_args()
    evidence = load_json(args.evidence_matrix)
    readiness = load_json(args.readiness_report) if args.readiness_report else {}
    concerns = load_json(args.concerns) if args.concerns else DEFAULT_CONCERNS
    rows = build_rebuttal_pack(evidence, concerns, readiness)
    claim_ladder = claim_ladder_from_evidence_rows(evidence)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "rebuttal_pack.json"
    output_md = args.output_dir / "rebuttal_pack.md"
    output_manifest = args.output_dir / "rebuttal_pack_manifest.json"
    output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(to_markdown(rows, readiness, claim_ladder), encoding="utf-8")
    output_manifest.write_text(
        json.dumps(
            build_rebuttal_manifest(
                rows=rows,
                concerns=concerns,
                claim_ladder=claim_ladder,
                concerns_path=args.concerns,
                evidence_matrix=args.evidence_matrix,
                readiness_report=args.readiness_report,
                output_json=output_json,
                output_md=output_md,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} rebuttal entries to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a rebuttal pack from claim evidence.")
    parser.add_argument("--evidence-matrix", required=True, type=Path)
    parser.add_argument("--readiness-report", type=Path)
    parser.add_argument("--concerns", type=Path, help="Optional JSON list of reviewer concern templates.")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_rebuttal_pack(
    evidence_rows: list[dict[str, Any]],
    concerns: list[dict[str, Any]],
    readiness: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    entries = []
    readiness_supplied = readiness is not None
    readiness_ok = bool((readiness or {}).get("ok", False))
    for concern in concerns:
        matches = match_claims(evidence_rows, concern)
        statuses = [row.get("status", "missing_evidence") for row in matches]
        if matches and all(status == "safe_to_claim" for status in statuses):
            defense_status = "answer_ready" if not readiness_supplied or readiness_ok else "needs_readiness"
        elif any(status == "contradicted" for status in statuses):
            defense_status = "cannot_claim"
        elif matches:
            defense_status = "needs_evidence"
        else:
            defense_status = "missing_claim_mapping"

        entries.append(
            {
                "id": concern["id"],
                "topic": concern["topic"],
                "question": concern["question"],
                "defense_status": defense_status,
                "recommended_position": recommended_position(defense_status, concern),
                "matched_claims": [
                    {
                        "claim_id": row.get("claim_id"),
                        "section": row.get("paper_section"),
                        "status": row.get("status"),
                        "claim": row.get("claim"),
                        "evidence": row.get("evidence"),
                    }
                    for row in matches
                ],
                "readiness_ok": readiness_ok,
            }
        )
    return entries


def build_rebuttal_manifest(
    *,
    rows: list[dict[str, Any]],
    concerns: list[dict[str, Any]],
    claim_ladder: list[dict[str, Any]] | None = None,
    concerns_path: Path | None,
    evidence_matrix: Path,
    readiness_report: Path | None,
    output_json: Path,
    output_md: Path,
) -> dict[str, Any]:
    status_counts: dict[str, int] = {key: 0 for key in DEFENSE_STATUS_KEYS}
    claim_ids: set[str] = set()
    for row in rows:
        status = str(row.get("defense_status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        for claim in row.get("matched_claims", []):
            claim_id = str(claim.get("claim_id", "")).strip()
            if claim_id:
                claim_ids.add(claim_id)

    readiness_present = bool(readiness_report and readiness_report.exists())
    return {
        "schema_version": 1,
        "entry_count": len(rows),
        "defense_status_counts": status_counts,
        "readiness_ok": all(bool(row.get("readiness_ok")) for row in rows) if rows else False,
        "matched_claim_ids": sorted(claim_ids),
        "claim_ladder": claim_ladder or [],
        "concern_templates": concern_template_record(concerns, concerns_path),
        "inputs": {
            "evidence_matrix": file_record(evidence_matrix),
            "readiness_report": file_record(readiness_report) if readiness_present else None,
        },
        "outputs": {
            "rebuttal_pack_json": file_record(output_json),
            "rebuttal_pack_md": file_record(output_md),
        },
        "claim_discipline": [
            "Treat rebuttal entries as reviewer-facing readiness notes, not as new paper claims.",
            "Use an answer only when the matched Evidence Matrix rows are safe_to_claim and submission readiness is ok.",
        ],
    }


def claim_ladder_from_evidence_rows(evidence_rows: Any) -> list[dict[str, Any]]:
    claims = evidence_rows if isinstance(evidence_rows, list) else []
    return build_claim_ladder_status({"claims": claims})


def concern_template_record(concerns: list[dict[str, Any]], concerns_path: Path | None) -> dict[str, Any]:
    return {
        "source": str(concerns_path) if concerns_path else "DEFAULT_CONCERNS",
        "count": len(concerns),
        "sha256": json_sha256(concerns),
    }


def json_sha256(data: Any) -> str:
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_record(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": "", "present": False, "bytes": 0, "sha256": ""}
    present = path.exists() and path.stat().st_size > 0
    return {
        "path": str(path),
        "present": present,
        "bytes": path.stat().st_size if path.exists() else 0,
        "sha256": file_sha256(path) if present else "",
    }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def match_claims(evidence_rows: list[dict[str, Any]], concern: dict[str, Any]) -> list[dict[str, Any]]:
    sections = {str(item).lower() for item in concern.get("claim_sections", [])}
    keywords = [str(item).lower() for item in concern.get("keywords", [])]
    matches = []
    for row in evidence_rows:
        section = str(row.get("paper_section", "")).lower()
        claim_text = str(row.get("claim", "")).lower()
        section_match = not sections or section in sections
        keyword_match = not keywords or any(keyword in claim_text for keyword in keywords)
        if section_match and keyword_match:
            matches.append(row)
    return matches


def recommended_position(defense_status: str, concern: dict[str, Any]) -> str:
    if defense_status == "answer_ready":
        return "Use the matched evidence directly in the response."
    if defense_status == "needs_readiness":
        return (
            "Matched Evidence Matrix rows are safe_to_claim, but submission readiness is not ok; "
            "keep this as a draft response until readiness, synced assets, and raw gates pass."
        )
    if defense_status == "cannot_claim":
        return "Do not claim this result; move it to limitations or future work."
    if defense_status == "needs_evidence":
        return concern.get("fallback", "Keep as a planned or preliminary result until the gate passes.")
    return concern.get("fallback", "Add an evidence mapping before making this claim.")


def to_markdown(
    entries: list[dict[str, Any]],
    readiness: dict[str, Any] | None = None,
    claim_ladder: list[dict[str, Any]] | None = None,
) -> str:
    lines = [
        "# BlindSpot-RL Rebuttal Pack",
        "",
        f"- Submission readiness ok: `{bool((readiness or {}).get('ok', False))}`",
        "",
    ]
    if claim_ladder:
        lines.extend(
            [
                "## Claim Ladder Status",
                "",
                "| Level | Status | Required Claims | Blocking Claims |",
                "| --- | --- | --- | --- |",
            ]
        )
        for row in claim_ladder:
            lines.append(
                f"| {row.get('level', '')} | `{row.get('status', '')}` | "
                f"{', '.join(str(item) for item in row.get('required_claim_ids', []))} | "
                f"{'; '.join(str(item) for item in row.get('missing_or_non_safe_claims', [])) or 'none'} |"
            )
        lines.append("")
    for entry in entries:
        lines.extend(
            [
                f"## {entry['id']} - {entry['topic']}",
                "",
                f"**Reviewer question:** {entry['question']}",
                "",
                f"**Defense status:** `{entry['defense_status']}`",
                "",
                f"**Recommended position:** {entry['recommended_position']}",
                "",
                "**Matched claims:**",
                "",
            ]
        )
        if entry["matched_claims"]:
            for claim in entry["matched_claims"]:
                lines.append(
                    f"- `{claim.get('claim_id')}` [{claim.get('status')}]: "
                    f"{claim.get('claim')} Evidence: {claim.get('evidence')}"
                )
        else:
            lines.append("- No mapped claim yet.")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
