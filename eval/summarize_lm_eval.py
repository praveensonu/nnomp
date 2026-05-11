import json
import sys
from pathlib import Path

import pandas as pd


def extract_metrics(result_json_path: Path) -> dict:
    with open(result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = {
        "result_json": str(result_json_path),
    }

    results = data.get("results", {})
    groups = data.get("groups", {})

    # Main per-task results
    for task_name, metrics in results.items():
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                out[f"{task_name}__{metric_name}"] = value

    # Group aggregates, when present
    for group_name, metrics in groups.items():
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                out[f"group__{group_name}__{metric_name}"] = value

    return out


def main(results_root: str) -> int:
    results_root = Path(results_root)
    run_summary_path = results_root / "run_summary.json"
    if not run_summary_path.exists():
        raise FileNotFoundError(f"Missing {run_summary_path}")

    with open(run_summary_path, "r", encoding="utf-8") as f:
        runs = json.load(f)

    rows = []
    for run in runs:
        row = {
            "model_name": run["model_name"],
            "adapter_path": run["adapter_path"],
            "tasks": ",".join(run["tasks"]),
            "returncode": run["returncode"],
            "run_dir": run["run_dir"],
        }

        result_json = run.get("result_json")
        if result_json and Path(result_json).exists():
            row.update(extract_metrics(Path(result_json)))

        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = results_root / "summary.csv"
    xlsx_path = results_root / "summary.xlsx"

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    print(df)
    print(f"\nSaved:\n- {csv_path}\n- {xlsx_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/summarize_lm_eval.py ./results")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))