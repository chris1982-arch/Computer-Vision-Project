"""
Module 3- Evaluation - Compare the simple detector-only baseline with the final filtered system.

Metrics:
- TPR = annotated real objects with an accepted detection / all annotated real objects
- FPR = annotated decoys with an accepted detection / all annotated decoys
- A detection matches an annotated object when the predicted box centre lies
  inside that object's reference box.
- Detections without "is_real" are treated as accepted, making the raw detector
  JSON the baseline.
Run from the project root:
    python src/evaluate.py --split test
    python src/evaluate.py --split eval

"""

import argparse
import json
import io
from contextlib import redirect_stdout
from pathlib import Path

from labels import load_labels, evaluate


ROOT = Path(__file__).resolve().parents[1]

LABEL_DIRS = {
    "test": [ROOT / "labels" / "chris_test"],
    "eval": [
        ROOT / "labels" / "eval_first64",
        ROOT / "labels" / "eval_last63",
    ],
}


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}\n"
            "Run the detector and decoy-filter pipeline first."
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list of image entries.")
    return data


def print_comparison(baseline, final):
    rows = [
        ("Baseline (detector only)", baseline),
        ("Final system (decoy filter)", final),
    ]

    print("\nEvaluation results")
    print("=" * 86)
    print(f"{'System':<30} {'Real accepted':>13} {'TPR':>9} "
          f"{'Decoys accepted':>16} {'FPR':>9}")
    print("-" * 86)

    for name, result in rows:
        print(
            f"{name:<30} "
            f"{result['real_accepted']:>6}/{result['n_real']:<6} "
            f"{result['TPR']:>8.3f} "
            f"{result['decoy_accepted']:>7}/{result['n_decoy']:<8} "
            f"{result['FPR']:>8.3f}"
        )

    print("-" * 86)
    print("Targets: TPR >= 0.750 and FPR <= 0.150")
    for name, result in rows:
        tpr_ok = result["TPR"] >= 0.75
        fpr_ok = result["FPR"] <= 0.15
        status = "PASS" if tpr_ok and fpr_ok else "FAIL"
        print(
            f"{name}: {status} "
            f"(TPR {'PASS' if tpr_ok else 'FAIL'}, "
            f"FPR {'PASS' if fpr_ok else 'FAIL'})"
        )

    print("\nCounts:")
    print(
        f"  Annotated real objects: {baseline['n_real']}; "
        f"annotated decoys: {baseline['n_decoy']}"
    )
    print(
        f"  Baseline missed real objects: "
        f"{baseline['n_real'] - baseline['real_accepted']}"
    )
    print(
        f"  Final missed/rejected real objects: "
        f"{final['n_real'] - final['real_accepted']}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare detector-only baseline against the filtered system."
    )
    parser.add_argument(
        "--split",
        choices=("test", "eval"),
        required=True,
        help="Use test for tuning; use eval for final reporting after freezing thresholds.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Optional override for the detector-only JSON.",
    )
    parser.add_argument(
        "--final",
        type=Path,
        default=None,
        help="Optional override for the final filtered JSON.",
    )
    args = parser.parse_args()

    baseline_path = args.baseline or ROOT / "outputs" / f"detections_{args.split}.json"
    final_path = args.final or ROOT / "outputs" / f"filtered_{args.split}.json"

    # Both systems are evaluated against the exact same annotations.
    labels = load_labels(LABEL_DIRS[args.split])

    baseline_images = load_json(baseline_path)
    final_images = load_json(final_path)

    # Prevent accidentally comparing different image sets.
    baseline_ids = {image.get("image_id") for image in baseline_images}
    final_ids = {image.get("image_id") for image in final_images}
    label_ids = set(labels)

    if baseline_ids != final_ids:
        raise ValueError(
            "Baseline and final JSON contain different image IDs. "
            f"Baseline-only: {len(baseline_ids - final_ids)}, "
            f"final-only: {len(final_ids - baseline_ids)}."
        )

    missing_labels = label_ids - baseline_ids
    if missing_labels:
        raise ValueError(
            f"{len(missing_labels)} labelled images are missing from the detections. "
            f"Example: {sorted(missing_labels)[0]}"
        )

    baseline_metrics = evaluate(baseline_images, labels)
    final_metrics = evaluate(final_images, labels)

    # Capture the report once so the same results are printed and saved.
    report = io.StringIO()
    with redirect_stdout(report):
        print(f"Split: {args.split}")
        print(f"Baseline JSON: {baseline_path.relative_to(ROOT) if baseline_path.is_relative_to(ROOT) else baseline_path}")
        print(f"Final JSON:    {final_path.relative_to(ROOT) if final_path.is_relative_to(ROOT) else final_path}")
        print_comparison(baseline_metrics, final_metrics)

    report_text = report.getvalue()
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"evaluation_{args.split}.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print(report_text, end="")
    print(f"\nResults saved to: {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
