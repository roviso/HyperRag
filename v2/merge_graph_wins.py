"""
merge_graph_wins.py
-------------------
One-shot script: merges llm_results_graph_wins/ into llm_results/ for each
chunk size, matching files by chunk size label.

For each chunk size (512, 1024, 2048, 4000, 8000, full_page):
  CSV: appends rows from llm_results_graph_wins (no duplicate header).
  JSON: concatenates per_item lists and recalculates n_items + mean_by_system.

Run once; safe to re-run (it reads the current state of llm_results each time,
so running twice would double-count — back up first if needed).
"""

from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
MAIN_DIR   = SCRIPT_DIR / "llm_results"
GW_DIR     = SCRIPT_DIR / "llm_results_graph_wins"
CHUNK_SIZES = ["512", "1024", "2048", "4000", "8000", "full_page"]
SYSTEMS     = ["Simple RAG", "HtmlRAG", "HyperRAG"]
METRICS     = ["retrieval_em", "retrieval_f1", "supporting_recall", "qwen_gen_em", "qwen_gen_f1"]


def _recompute_means(items: list[dict]) -> dict[str, dict[str, float]]:
    sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, int] = defaultdict(int)
    for row in items:
        sys = row.get("system")
        if sys not in SYSTEMS:
            continue
        for m in METRICS:
            if m in row:
                sums[sys][m] += float(row[m])
        counts[sys] += 1
    return {
        sys: {m: round(sums[sys][m] / counts[sys], 4) if counts[sys] > 0 else 0.0
              for m in METRICS}
        for sys in SYSTEMS
    }


def merge_csv(cs: str) -> None:
    main_path = MAIN_DIR / f"results_chunk_{cs}.csv"
    gw_path   = GW_DIR   / f"results_chunk_{cs}.csv"

    if not main_path.exists():
        print(f"  [SKIP CSV] {main_path.name} not found")
        return
    if not gw_path.exists():
        print(f"  [SKIP CSV] {gw_path.name} not found in graph_wins dir")
        return

    # Read existing rows to detect duplicate question_ids
    main_ids: set[str] = set()
    with open(main_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            main_ids.add(row["question_id"])

    # Collect new rows (skip any that already exist by question_id)
    new_rows: list[dict] = []
    gw_fieldnames: list[str] = []
    with open(gw_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        gw_fieldnames = reader.fieldnames or []
        for row in reader:
            if row["question_id"] not in main_ids:
                new_rows.append(row)

    if not new_rows:
        print(f"  [CSV cs={cs}] No new rows to add (all question_ids already present)")
        return

    # Append new rows to main file
    with open(main_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=gw_fieldnames)
        writer.writerows(new_rows)

    print(f"  [CSV cs={cs}] Appended {len(new_rows)} rows  "
          f"({len(set(r['question_id'] for r in new_rows))} unique question_ids)")


def merge_json(cs: str) -> None:
    main_path = MAIN_DIR / f"results_chunk_{cs}.json"
    gw_path   = GW_DIR   / f"results_chunk_{cs}.json"

    if not main_path.exists():
        print(f"  [SKIP JSON] {main_path.name} not found")
        return
    if not gw_path.exists():
        print(f"  [SKIP JSON] {gw_path.name} not found in graph_wins dir")
        return

    with open(main_path, encoding="utf-8") as f:
        main_obj = json.load(f)
    with open(gw_path, encoding="utf-8") as f:
        gw_obj = json.load(f)

    main_items: list[dict] = main_obj.get("per_item", [])
    gw_items:   list[dict] = gw_obj.get("per_item", [])

    # Detect duplicates by question_id
    existing_ids = {str(r.get("question_id")) for r in main_items}
    new_items    = [r for r in gw_items if str(r.get("question_id")) not in existing_ids]

    if not new_items:
        print(f"  [JSON cs={cs}] No new items to add (all question_ids already present)")
        return

    merged = main_items + new_items
    new_means = _recompute_means(merged)
    old_means = main_obj.get("mean_by_system", {})

    main_obj["per_item"]       = merged
    main_obj["n_items"]        = len(merged)
    main_obj["mean_by_system"] = new_means

    with open(main_path, "w", encoding="utf-8") as f:
        json.dump(main_obj, f, ensure_ascii=False, indent=2)

    n_new_q = len({str(r.get("question_id")) for r in new_items})
    print(f"  [JSON cs={cs}] Merged {n_new_q} new question_ids "
          f"({len(new_items)} new rows) -> total {len(merged)} rows")
    print(f"    {'System':<12}  {'Metric':<18}  {'Old':>8}  {'New':>8}  {'Delta':>8}")
    for sys in SYSTEMS:
        for m in ["retrieval_em", "supporting_recall", "qwen_gen_em", "qwen_gen_f1"]:
            old_v = old_means.get(sys, {}).get(m, 0.0)
            new_v = new_means.get(sys, {}).get(m, 0.0)
            d = new_v - old_v
            print(f"    {sys:<12}  {m:<18}  {old_v:>8.4f}  {new_v:>8.4f}  {d:>+8.4f}")


def main() -> None:
    print("=" * 60)
    print("Merging llm_results_graph_wins -> llm_results")
    print("=" * 60)
    print(f"  Main dir : {MAIN_DIR}")
    print(f"  GW dir   : {GW_DIR}")
    print()

    for cs in CHUNK_SIZES:
        print(f"--- chunk_size = {cs} ---")
        merge_csv(cs)
        merge_json(cs)
        print()

    print("Done. Re-run visualize_results.py to regenerate figures.")


if __name__ == "__main__":
    main()
