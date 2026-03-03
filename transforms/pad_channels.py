
import numpy as np
import onnx
import copy
from __future__ import annotations
from typing import Any
from onnx import numpy_helper, shape_inference

"""
Applies the padding plan to the ONNX graph.

Converts ONNX tensors to numpy arrays and pads them
with zeros along the required axis to reach the target size.

Depends on: feasibility.get_padding_plan output, dim_vars, tensor_shapes.
Uses only onnx.numpy_helper and onnx.helper. No torch, no tensorflow.

Ensuring that cached shapes are updated to reflect the new padding and that 
shape inference is re-run to propagate the new shapes globally.
"""

def find_reshape_locked_vars(
    model: onnx.ModelProto,
    padding_plan: dict[str, int],
) -> set[str]:
    consumers: dict[str, list] = {}
    for node in model.graph.node:
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, []).append(node)

    initializer_names: set[str] = {init.name for init in model.graph.initializer}

    producer: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        if node.output:
            producer[node.output[0]] = node

    locked: set[str] = set()
    for var in padding_plan:
        tensor_name = var.rsplit("__dim", 1)[0]
        for consumer in consumers.get(tensor_name, []):
            if (
                consumer.op_type == "Reshape"
                and len(consumer.input) >= 2
                and consumer.input[1] in initializer_names
            ):
                locked.add(var)
                # Also lock the weight output-channel var for the Conv/Gemm that produces this tensor
                prod = producer.get(tensor_name)
                if (
                    prod is not None
                    and prod.op_type in ("Conv", "Gemm")
                    and len(prod.input) > 1
                    and prod.input[1]
                ):
                    weight_name = prod.input[1]
                    weight_dim0 = f"{weight_name}__dim0"
                    if weight_dim0 in padding_plan:
                        locked.add(weight_dim0)
                break
    return locked


def pad_tensor(
    array: np.ndarray,
    axis: int,
    target_size: int,
) -> np.ndarray:
    # np.pad with zeros along axis to reach target_size
    # raise ValueError if array.shape[axis] > target_size
    current = array.shape[axis]
    if current > target_size:
        raise ValueError(
            f"array.shape[{axis}] ({current}) > target_size ({target_size}); cannot shrink"
        )
    if current == target_size:
        return array
    pad_width: list[tuple[int, int]] = [(0, 0)] * array.ndim
    pad_width[axis] = (0, target_size - current)
    return np.pad(array, pad_width, mode="constant", constant_values=0)


def update_tensor_shape(
    model: onnx.ModelProto,
    tensor_name: str,
    new_shape: list[int],
) -> None:
    # Find tensor in model.graph.value_info or graph.input or graph.output
    # Update its type.tensor_type.shape dims in place
    for vi in model.graph.value_info:
        if vi.name == tensor_name:
            _set_shape(vi.type.tensor_type.shape, new_shape)
            return
    for inp in model.graph.input:
        if inp.name == tensor_name:
            _set_shape(inp.type.tensor_type.shape, new_shape)
            return
    for out in model.graph.output:
        if out.name == tensor_name:
            _set_shape(out.type.tensor_type.shape, new_shape)
            return


def _set_shape(shape_proto: Any, new_shape: list[int]) -> None:
    del shape_proto.dim[:]
    for v in new_shape:
        d = shape_proto.dim.add()
        d.dim_value = v


def apply_padding_plan(
    model: onnx.ModelProto,
    padding_plan: dict[str, int],
    dim_vars: dict[str, list[str]],
    tensor_shapes: dict[str, list],
) -> onnx.ModelProto:
    if not padding_plan:
        return model
    model = copy.deepcopy(model)

    # patch initializers (weights and biases)
    new_initializers: list[onnx.TensorProto] = []
    depthwise_weight_names: set[str] = set()
    for node in model.graph.node:
        if node.op_type == "Conv":
            for attr in node.attribute:
                if attr.name == "group" and attr.i > 1:
                    if len(node.input) > 1:
                        depthwise_weight_names.add(node.input[1])
    for init in model.graph.initializer:
        name = init.name
        if name not in dim_vars:
            new_initializers.append(init)
            continue
        var_list = dim_vars[name]
        array = numpy_helper.to_array(init)
        modified = False
        for axis in range(min(len(var_list), array.ndim)):
            if name in depthwise_weight_names and axis == 1:
                continue    # never pad the per-group kernel dim of a depthwise weight
            var = var_list[axis]
            if var not in padding_plan:
                continue
            target = padding_plan[var]
            if array.shape[axis] < target:
                array = pad_tensor(array, axis, target)
                modified = True
        if modified:
            new_initializers.append(numpy_helper.from_array(array, name=name))
        else:
            new_initializers.append(init)
    del model.graph.initializer[:]
    model.graph.initializer.extend(new_initializers)
    # pad bias tensors to match new output channel count
    bias_updates: dict[str, np.ndarray] = {}
    for node in model.graph.node:
        if node.op_type not in ("Conv", "Gemm") or len(node.input) < 3:
            continue
        bias_name = node.input[2]
        if not bias_name:
            continue
        weight_name = node.input[1] if len(node.input) > 1 else None
        if weight_name is None:
            continue
        new_out_ch = None
        for init in model.graph.initializer:
            if init.name == weight_name:
                new_out_ch = init.dims[0]
                break
        if new_out_ch is None:
            continue
        for init in model.graph.initializer:
            if init.name == bias_name:
                b = numpy_helper.to_array(init)
                if b.shape[0] < new_out_ch:
                    b = pad_tensor(b, 0, new_out_ch)
                    bias_updates[bias_name] = b
                break

    if bias_updates:
        new_inits = []
        for init in model.graph.initializer:
            if init.name in bias_updates:
                new_inits.append(numpy_helper.from_array(bias_updates[init.name], name=init.name))
            else:
                new_inits.append(init)
        del model.graph.initializer[:]
        model.graph.initializer.extend(new_inits)

    # Update group attribute on depthwise conv nodes to match padded channels
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        # check if it's depthwise (group > 1)
        for attr in node.attribute:
            if attr.name == "group" and attr.i > 1:
                # find the weight initializer for this node
                weight_name = node.input[1] if len(node.input) > 1 else None
                if weight_name is None:
                    break
                # find new out_channels from updated initializer
                for init in model.graph.initializer:
                    if init.name == weight_name:
                        new_out_ch = init.dims[0]
                        attr.i = new_out_ch
                        break
                break

    # clear all stale shape annotations so infer_shapes
    # to fix the metadata propagation bug
    del model.graph.value_info[:]
    for out in model.graph.output:
        out.type.tensor_type.shape.Clear()

    # re-run shape inference to propagate all new shapes globally
    model = shape_inference.infer_shapes(model)
    return model
