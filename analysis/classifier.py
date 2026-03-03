"""
Classifies each constraint violation as FREE, COUPLED, or LOCKED.
Depends on constraints defined in knowledge_base.constraints.py and groups found in propagator.UnionFind.
FREE violations are those which can be padded without affecting other dimensions.
COUPLED violations are those which can be padded, but affect other dimensions.
LOCKED violations are those which cannot be padded.
"""

from __future__ import annotations

from dataclasses import dataclass

import onnx
from graph.dim_variables import get_dim_variable
from graph.propagator import UnionFind
from knowledge_base.constraints import (
    get_alignment,
    get_patchable_constraints,
)


@dataclass
class Violation:
    node_name: str
    op_type: str
    constraint_name: str  # e.g. "input_channels"
    dim_variable: str  # the specific symbolic var that violates
    current_value: int
    target_value: int  # rounded up to nearest alignment multiple
    classification: str  # "FREE", "COUPLED", "LOCKED"
    coupled_nodes: list[str]
    reason: str


def round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _tensor_name_from_dim_var(dim_var: str) -> str:
    return dim_var.rsplit("__dim", 1)[0]


def _effective_op_type(node) -> str:
    """ONNX represents depthwise conv as Conv with group == in_channels."""
    if node.op_type == "Conv":
        for attr in node.attribute:
            if attr.name == "group" and attr.i > 1:
                return "DepthwiseConv"
    return node.op_type


def _node_id(node, index: int) -> str:
    return node.name if node.name else f"{node.op_type}_{index}"


def _nodes_using_tensors(model: onnx.ModelProto, tensor_names: set[str]) -> list[str]:
    result: list[str] = []
    for idx, node in enumerate(model.graph.node):
        for name in list(node.input) + list(node.output):
            if name and name in tensor_names:
                result.append(_node_id(node, idx))
                break
    return result


def classify_violations(
    model: onnx.ModelProto,
    dim_vars: dict[str, list[str]],
    uf: UnionFind,
    locked_vars: set[str],
    tensor_shapes: dict[str, list],
    graph_inputs: set[str],
    graph_outputs: set[str],
) -> list[Violation]:
    #   FREE   → group size == 1, not in locked_vars, not in graph i/o tensors
    #   COUPLED → group size > 1, no member is locked or graph i/o
    #   LOCKED  → anything else, set reason explaining why
    violations: list[Violation] = []
    groups_cache: dict[str, list[str]] | None = None

    def get_group(dim_var: str) -> list[str]:
        nonlocal groups_cache
        if groups_cache is None:
            groups_cache = uf.groups()
        root = uf.find(dim_var)
        return groups_cache.get(root, [dim_var])

    for idx, node in enumerate(model.graph.node):
        op_type = _effective_op_type(node)
        constraints = get_patchable_constraints(op_type)
        if not constraints:
            continue
        # For Conv, DepthwiseConv, and Gemm dim_index refers to weight tensor = input[1]
        inputs = [x for x in node.input if x]
        if len(inputs) < 2:
            continue
        weight_name = inputs[1]

        for constraint_name, c in constraints.items():
            dim_index = c.get("dim_index")
            if dim_index is None:
                continue
            alignment = get_alignment(op_type, constraint_name)
            if alignment is None:
                continue
            if weight_name not in tensor_shapes or weight_name not in dim_vars:
                continue
            shape = tensor_shapes[weight_name]
            if dim_index >= len(shape):
                continue
            current = shape[dim_index]
            if current is None or not isinstance(current, int):
                continue
            if current % alignment == 0:
                continue
            target_value = round_up(current, alignment)
            try:
                dim_variable = get_dim_variable(dim_vars, weight_name, dim_index)
            except KeyError:
                continue

            group_members = get_group(dim_variable)
            tensor_names_in_group = {_tensor_name_from_dim_var(v) for v in group_members}
            any_locked = any(v in locked_vars for v in group_members)
            any_graph_io = bool(
                tensor_names_in_group & graph_inputs or tensor_names_in_group & graph_outputs
            )

            if len(group_members) == 1 and dim_variable not in locked_vars:
                tname = _tensor_name_from_dim_var(dim_variable)
                if tname not in graph_inputs and tname not in graph_outputs:
                    classification = "FREE"
                    reason = "single dim var, not locked, not graph i/o"
                    coupled_nodes = []
                else:
                    classification = "LOCKED"
                    reason = "tensor is graph input or output"
                    coupled_nodes = _nodes_using_tensors(model, tensor_names_in_group)
            elif len(group_members) > 1 and not any_locked and not any_graph_io:
                classification = "COUPLED"
                reason = "dim var grouped with others; none locked or graph i/o"
                coupled_nodes = _nodes_using_tensors(model, tensor_names_in_group)
            else:
                classification = "LOCKED"
                if any_locked and any_graph_io:
                    reason = "dim var in locked set and tensor in graph i/o"
                elif any_locked:
                    reason = "dim var or group member in locked set"
                else:
                    reason = "tensor in graph input or output"
                coupled_nodes = _nodes_using_tensors(model, tensor_names_in_group)

            violations.append(
                Violation(
                    node_name=_node_id(node, idx),
                    op_type=op_type,
                    constraint_name=constraint_name,
                    dim_variable=dim_variable,
                    current_value=current,
                    target_value=target_value,
                    classification=classification,
                    coupled_nodes=coupled_nodes,
                    reason=reason,
                )
            )

    return violations
