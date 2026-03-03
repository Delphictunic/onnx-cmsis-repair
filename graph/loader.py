import onnx
from onnx import shape_inference
"""
ONNX open neural network exchange. Allows portability between different frameworks.
Stores a computational graph of a neural network. We load the model perform checking for consistency and
also call shape_inference to extract tensor shapes. Model is represented as a protobuf object. Protobuf
developed by Google, provides efficient serialization and deserialization of data. Serialization is the process
of converting data structures stored in memory into a format that can be stored on disk or transmitted over a network.
We use value_info, input, output, and initializer to extract tensor shapes, names, and dtypes.
"""

def load_model(path: str) -> onnx.ModelProto:
    model = onnx.load(path)
    onnx.checker.check_model(model)
    model = shape_inference.infer_shapes(model)
    return model


def save_model(model: onnx.ModelProto, path: str) -> None:
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _dims_to_list(shape) -> list:
    """Extract [dim_value or None, ...] from a TensorShapeProto."""
    if shape is None:
        return []
    result = []
    for dim in shape.dim:
        if dim.WhichOneof("value") == "dim_value":
            result.append(dim.dim_value)
        else: 
            result.append(None)
    return result


def get_tensor_shapes(model: onnx.ModelProto) -> dict[str, list]:
    out: dict[str, list] = {}
    for vi in model.graph.value_info:
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            out[vi.name] = _dims_to_list(vi.type.tensor_type.shape)
    for inp in model.graph.input:
        if inp.type.HasField("tensor_type") and inp.type.tensor_type.HasField("shape"):
            out[inp.name] = _dims_to_list(inp.type.tensor_type.shape)
    for out_node in model.graph.output:
        if out_node.type.HasField("tensor_type") and out_node.type.tensor_type.HasField("shape"):
            out[out_node.name] = _dims_to_list(out_node.type.tensor_type.shape)
    for init in model.graph.initializer:
        out[init.name] = list(init.dims)
    return out


def get_graph_inputs(model: onnx.ModelProto) -> set[str]:
    initializer_names = {init.name for init in model.graph.initializer}
    return {i.name for i in model.graph.input if i.name not in initializer_names}

def get_graph_outputs(model: onnx.ModelProto) -> set[str]:
    return {o.name for o in model.graph.output}

def get_tensor_dtypes(model: onnx.ModelProto) -> dict[str, int]:
    initializer_names = {init.name for init in model.graph.initializer}
    return {
        i.name: i.type.tensor_type.elem_type
        for i in model.graph.input
        if i.name not in initializer_names
    }