"""
Single self-contained script that runs the full cmsis_nn_optimizer pipeline
programmatically (no CLI args).
"""

from graph.loader import (
    get_graph_inputs,
    get_graph_outputs,
    get_tensor_shapes,
    load_model,
    save_model,
)
from graph.dim_variables import assign_dim_variables
from graph.propagator import propagate_constraints
from analysis.classifier import classify_violations
from analysis.feasibility import check_feasibility, get_padding_plan
from transforms.pad_channels import apply_padding_plan
from transforms.validate import validate
from report import export_report, print_report
import sys


def run_pipeline(
    model_path: str,
    output_path: str | None = None,
    report_json_path: str | None = None,
    report_only: bool = False,
) -> tuple[list, list]:
    # runs the full pipeline, prints report, optionally saves model and JSON
    # returns (feasible_violations, infeasible_violations)
    model = load_model(model_path)
    tensor_shapes = get_tensor_shapes(model)
    graph_inputs = get_graph_inputs(model)
    graph_outputs = get_graph_outputs(model)
    dim_vars = assign_dim_variables(tensor_shapes)
    uf, locked_vars = propagate_constraints(model, dim_vars)
    violations = classify_violations(
        model,
        dim_vars,
        uf,
        locked_vars,
        tensor_shapes,
        graph_inputs,
        graph_outputs,
    )
    feasible, infeasible = check_feasibility(violations)

    applied = False
    padding_plan = None
    if not report_only:
        from transforms.pad_channels import find_reshape_locked_vars
        padding_plan = get_padding_plan(feasible, uf)
        if padding_plan:
            reshape_locked_vars = find_reshape_locked_vars(model, padding_plan)
            if reshape_locked_vars:
                still_feasible = []
                for v in feasible:
                    if v.dim_variable in reshape_locked_vars:
                        v.classification = "LOCKED"
                        v.reason = "downstream Reshape has hardcoded shape"
                        infeasible.append(v)
                    else:
                        still_feasible.append(v)
                feasible = still_feasible
                padding_plan = get_padding_plan(feasible, uf)
            modified = apply_padding_plan(model, padding_plan, dim_vars, tensor_shapes)
            ok, msg = validate(model, modified)
            if not ok:
                print(f"Validation failed: {msg}")
            else:
                applied = True
                if output_path:
                    save_model(modified, output_path)

    print_report(feasible, infeasible, applied=applied, padding_plan=padding_plan)

    if report_json_path:
        export_report(feasible, infeasible, report_json_path)

    return (feasible, infeasible)


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "test_model.onnx"
    output_path = model_path.replace(".onnx", "_patched.onnx")
    report_path = model_path.replace(".onnx", "_report.json")
    run_pipeline(
        model_path=model_path,
        output_path=output_path,
        report_json_path=report_path,
    )
