from __future__ import annotations

import onnx
from graph.dim_variables import get_dim_variable
"""
Walks the ONNX graph and unions dimension variables that must be equal
by ONNX operation semantics (Conv, Add, Concat, Gemm, Reshape, Flatten).

For each op, hardcoded shape compatibility rules from the ONNX spec are
encoded as union calls, grouping dim variables that must move together
when a dimension is padded.

Reshape and Flatten output dims are added to the locked set, as their
shapes are determined by a hardcoded constant and cannot be changed by
static weight padding.

Returns a UnionFind (groups of coupled dim variables) and a locked set.
The locked set is used to identify dimensions that are not affected by padding operations.
"""

class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> None:
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1

    def groups(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for x in self.parent:
            r = self.find(x)
            if r not in result:
                result[r] = []
            result[r].append(x)
        return result


def _get_attr(node: object, name: str, default: int) -> int:
    for a in getattr(node, "attribute", []):
        if getattr(a, "name", None) == name:
            return getattr(a, "i", default)
    return default


def _try_union(
    uf: UnionFind,
    dim_vars: dict[str, list[str]],
    tensor_name: str,
    dim_index: int,
    other_name: str,
    other_dim: int,
) -> None:
    try:
        a = get_dim_variable(dim_vars, tensor_name, dim_index)
        b = get_dim_variable(dim_vars, other_name, other_dim)
        uf.union(a, b)
    except KeyError:
        pass


def _union_dims(
    uf: UnionFind,
    dim_vars: dict[str, list[str]],
    names: list[str],
    dim_index: int,
) -> None:
    vars_at_dim: list[str] = []
    for n in names:
        if not n:
            continue
        try:
            v = get_dim_variable(dim_vars, n, dim_index)
            vars_at_dim.append(v)
        except KeyError:
            pass
    for i in range(1, len(vars_at_dim)):
        uf.union(vars_at_dim[0], vars_at_dim[i])


def propagate_constraints(
    model: onnx.ModelProto,
    dim_vars: dict[str, list[str]],
) -> tuple[UnionFind, set[str]]:
    uf = UnionFind()
    locked: set[str] = set()

    for node in model.graph.node:
        op = node.op_type
        inputs = [x for x in node.input if x]
        outputs = [x for x in node.output if x]

        if op == "Conv":
            if len(inputs) >= 2 and len(outputs) >= 1:
                _try_union(uf, dim_vars, inputs[0], 1, inputs[1], 1)
                _try_union(uf, dim_vars, outputs[0], 1, inputs[1], 0)
                _try_union(uf, dim_vars, inputs[0], 0, outputs[0], 0)

        elif op == "Add":
            for dim_index in range(
                max(
                    (len(dim_vars[n]) for n in inputs + outputs if n in dim_vars),
                    default=0,
                )
            ):
                _union_dims(uf, dim_vars, inputs + outputs, dim_index)

        elif op == "Concat":
            axis = _get_attr(node, "axis", 0)
            n_dims = None
            for name in inputs + outputs:
                if name in dim_vars:
                    n_dims = len(dim_vars[name])
                    break
            if n_dims is not None:
                actual_axis = axis if axis >= 0 else axis + n_dims
                for dim_index in range(n_dims):
                    if dim_index == actual_axis:
                        continue
                    _union_dims(uf, dim_vars, inputs + outputs, dim_index)

        elif op == "Gemm":
            if len(inputs) >= 2 and len(outputs) >= 1:
                _try_union(uf, dim_vars, inputs[0], 1, inputs[1], 1)
                _try_union(uf, dim_vars, outputs[0], 1, inputs[1], 0)
                _try_union(uf, dim_vars, inputs[0], 0, outputs[0], 0)

        elif op == "Reshape":
            # only lock if the shape input is a stored constant initializer
            initializer_names = {init.name for init in model.graph.initializer}
            if len(inputs) >= 2 and inputs[1] in initializer_names:
                for out_name in outputs:
                    try:
                        for dim_index in range(len(dim_vars[out_name])):
                            v = get_dim_variable(dim_vars, out_name, dim_index)
                            locked.add(v)
                    except KeyError:
                        pass

        elif op == "Flatten":
            # Flatten computes its output shape dynamically from input
            # do not lock -- output channels propagate freely from the input tensor
            pass

    return (uf, locked)
