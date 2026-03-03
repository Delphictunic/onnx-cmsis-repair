from __future__ import annotations
import numpy as np
import onnx
import onnxruntime

"""
Runs both original and modified models through onnxruntime and checks
outputs are numerically consistent.
Depends on: onnx, onnxruntime, numpy only.
"""

def generate_random_inputs(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    initializer_names = {init.name for init in model.graph.initializer}
    # For each graph input, read its shape from model.graph.input
    # Replace any dynamic dim (dim_value == 0) with 1
    # int8 tensors → np.random.randint(-128, 127, shape, dtype=np.int8)
    # float tensors → np.random.randn(*shape).astype(np.float32)
    result: dict[str, np.ndarray] = {}
    for inp in model.graph.input:
        if inp.name in initializer_names:
            continue
        if not inp.type.HasField("tensor_type"):
            continue
        tt = inp.type.tensor_type
        shape = []
        for dim in tt.shape.dim:
            v = dim.dim_value if dim.dim_value > 0 else 1
            shape.append(v)
        if not shape:
            shape = [1]
        elem_type = tt.elem_type
        if elem_type == onnx.TensorProto.INT8:
            result[inp.name] = np.random.randint(
                -128, 127, size=shape, dtype=np.int8
            )
        else:
            result[inp.name] = np.random.randn(*shape).astype(np.float32)
    return result


def run_model(
    model: onnx.ModelProto,
    inputs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    # onnxruntime.InferenceSession from model.SerializeToString()
    # session.run(None, inputs) → return dict of output_name: array
    opts = onnxruntime.SessionOptions()
    opts.log_severity_level = 3
    session = onnxruntime.InferenceSession(
        model.SerializeToString(),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    output_names = [o.name for o in model.graph.output]
    outputs = session.run(output_names, inputs)
    return dict(zip(output_names, outputs))


def validate(
    original_model: onnx.ModelProto,
    modified_model: onnx.ModelProto,
    tolerance: float = 1e-4,
) -> tuple[bool, str]:
    # Generate one shared random input
    # Run both models and check if output shapes match 
    # Check np.max(np.abs(original_out - modified_out)) < tolerance
    # Return (True, "OK") or (False, reason_string)
    inputs = generate_random_inputs(original_model)
    try:
        original_out = run_model(original_model, inputs)
        modified_out = run_model(modified_model, inputs)
    except Exception as e:
        return (False, f"run failed: {e}")
    if set(original_out) != set(modified_out):
        return (False, "output names differ")
    for name in original_out:
        a, b = original_out[name], modified_out[name]
        if a.ndim != b.ndim:
            return (False, f"output {name!r} ndim mismatch: {a.ndim} vs {b.ndim}")
        slices = tuple(slice(0, min(sa, sb)) for sa, sb in zip(a.shape, b.shape))
        diff = np.max(np.abs(a[slices].astype(np.float64) - b[slices].astype(np.float64)))
        if diff >= tolerance:
            return (False, f"output {name!r} max |diff| = {diff} >= {tolerance}")
    return (True, "OK")

