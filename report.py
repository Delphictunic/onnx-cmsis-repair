"""
Formats and outputs analysis results.
Depends on classifier.Violation only. No other internal imports.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from analysis.classifier import Violation


def print_report(
    feasible: list[Violation],
    infeasible: list[Violation],
    applied: bool = False,
    padding_plan: dict[str, int] | None = None,
) -> None:
    # Print a table with columns:
    # Node | Op | Constraint | Current | Target | Status
    #
    # feasible rows:   "PATCHED" if applied else "WILL PATCH"
    # infeasible rows: f"LOCKED: {violation.reason}"
    #
    # Footer line:
    # f"{total} violations. {len(feasible)} patched. {len(infeasible)} locked."
    cols = ("Node", "Op", "Constraint", "Current", "Target", "Status")
    width = (20, 16, 18, 8, 8, 36)
    row_fmt = " | ".join(f"{{:{w}}}" for w in width)
    header = row_fmt.format(*cols)
    sep = "-" * len(header)
    print(header)
    print(sep)
    status_patched = "PATCHED" if applied else "WILL PATCH"
    for v in feasible:
        print(
            row_fmt.format(
                v.node_name[: width[0]],
                v.op_type[: width[1]],
                v.constraint_name[: width[2]],
                str(v.current_value),
                str(v.target_value),
                status_patched[: width[5]],
            )
        )
    for v in infeasible:
        status = f"LOCKED: {v.reason}"[: width[5]]
        print(
            row_fmt.format(
                v.node_name[: width[0]],
                v.op_type[: width[1]],
                v.constraint_name[: width[2]],
                str(v.current_value),
                str(v.target_value),
                status,
            )
        )
    total = len(feasible) + len(infeasible)
    print(sep)
    print(f"{total} violations. {len(feasible)} patched. {len(infeasible)} locked.")

    if padding_plan:
        print()
        print("Additional tensors patched via group expansion:")
        print("-" * 60)
        # collect dim_variables already covered by direct violations
        reported_vars = {v.dim_variable for v in feasible + infeasible}
        for var, target in sorted(padding_plan.items()):
            if var not in reported_vars:
                parts = var.rsplit("__dim", 1)
                tensor_name = parts[0]
                axis = parts[1] if len(parts) > 1 else "?"
                print(f"  {tensor_name:35s} axis={axis} \u2192 {target}")


def export_report(
    feasible: list[Violation],
    infeasible: list[Violation],
    path: str,
) -> None:
    # Serialize both lists to JSON using dataclasses.asdict
    # Write to path
    data = {
        "feasible": [asdict(v) for v in feasible],
        "infeasible": [asdict(v) for v in infeasible],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
