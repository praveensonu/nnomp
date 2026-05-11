import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
import time
import yaml


def quote_model_arg_value(value: str) -> str:
    """
    Escape commas in model_args values, because lm-eval parses model_args
    as a comma-separated key=value string.
    """
    return value.replace(",", r"\,")


def build_model_args(base_model: str, adapter_path: str, dtype: str) -> str:
    parts = {
        "pretrained": base_model,
        "peft": adapter_path,
        "dtype": dtype,
    }
    return ",".join(f"{k}={quote_model_arg_value(str(v))}" for k, v in parts.items())


def find_results_json(run_dir: Path) -> Path | None:
    """
    lm-eval may write one or more json files under output_path.
    This function returns the first plausible results json.
    """
    candidates = sorted(run_dir.rglob("*.json"))
    for p in candidates:
        name = p.name.lower()
        if "result" in name or "results" in name:
            return p
    return candidates[0] if candidates else None


def main(config_path: str) -> int:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_model = cfg["base_model"]
    system_instruction = cfg.get("system_instruction")
    output_root = Path(cfg.get("output_root", "./results"))
    batch_size = str(cfg.get("batch_size", "auto:4"))
    device = cfg.get("device", "cuda:0")
    dtype = cfg.get("dtype", "bfloat16")

    output_root.mkdir(parents=True, exist_ok=True)

    summary = []

    for entry in cfg["models"]:
        model_name = entry["name"]
        adapter_path = entry["adapter_path"]
        tasks = entry["tasks"]

        task_str = ",".join(tasks)
        run_dir = output_root / model_name
        run_dir.mkdir(parents=True, exist_ok=True)

        model_args = build_model_args(
            base_model=base_model,
            adapter_path=adapter_path,
            dtype=dtype,
        )

        cmd = [
            "lm-eval",
            "run",
            "--model", "hf",
            "--model_args", model_args,
            "--tasks", task_str,
            "--device", device,
            "--batch_size", batch_size,
            "--apply_chat_template",
            "--output_path", str(run_dir),
            "--log_samples",
        ]

        if system_instruction:
            cmd.extend(["--system_instruction", system_instruction])

        print("=" * 80)
        print("Running:", " ".join(shlex.quote(x) for x in cmd))
        print("=" * 80)

        completed = subprocess.run(cmd, text=True)

        result_json = find_results_json(run_dir)

        summary.append({
            "model_name": model_name,
            "adapter_path": adapter_path,
            "tasks": tasks,
            "returncode": completed.returncode,
            "run_dir": str(run_dir),
            "result_json": str(result_json) if result_json else None,
        })
        time.sleep(10)  # brief pause between runs

    summary_path = output_root / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved run summary to: {summary_path}")
    failed = [x for x in summary if x["returncode"] != 0]
    if failed:
        print(f"{len(failed)} run(s) failed.")
        return 1

    print("All runs finished successfully.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_lm_eval_batch.py eval_configs/models.yaml")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))