"""
Not using NumPy ndarray because this module operates at a structural/symbolic level rather than a numerical one.
Instead, dictionaries are usedsince this file is responsible for dimension-variable assignment and lookup.
"""
#**type hints are used to ensure the correctness of the code.

# for each tensor, for each dim index, assign f"{tensor_name}__dim{i}"
# return {tensor_name: [var_name, ...]}
def assign_dim_variables(
    tensor_shapes: dict[str, list],  
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for tensor_name, dims in tensor_shapes.items():
        result[tensor_name] = [f"{tensor_name}__dim{i}" for i in range(len(dims))]
    return result

# lookup function
# raise error with clear message if not found
def get_dim_variable(
    dim_vars: dict[str, list[str]],
    tensor_name: str,
    dim_index: int,
) -> str:
    if tensor_name not in dim_vars:
        raise KeyError(f"Tensor {tensor_name!r} not in dim_vars")
    names = dim_vars[tensor_name]
    if dim_index < 0 or dim_index >= len(names):
        raise KeyError(
            f"Dim index {dim_index} out of range for tensor {tensor_name!r} (has {len(names)} dims)"
        )
    return names[dim_index]
