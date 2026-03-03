from analysis.classifier import Violation
from functools import reduce
import math
"""
    Split violations into feasible and infeasible.

    For COUPLED violations in the same union-find group with conflicting target
    values, resolve by taking the LCM of all targets which is the smallest value
    that satisfies all alignment constraints simultaneously.

    FREE violations are always feasible.
    LOCKED violations are always infeasible.

    Return (feasible: list[Violation], infeasible: list[Violation])
"""

def check_feasibility(violations: list[Violation]) -> tuple[list[Violation], list[Violation]]:
    # Group COUPLED violations by same logical dimension (same coupled_nodes set)
    coupled_by_group: dict[frozenset[str], list[Violation]] = {}
    for v in violations:
        if v.classification == "COUPLED":
            key = frozenset(v.coupled_nodes)
            coupled_by_group.setdefault(key, []).append(v)
    for group_violations in coupled_by_group.values():
        targets = {v.target_value for v in group_violations}
        if len(targets) > 1:
            # resolve by taking lcm across the group to preserve alignment constraints.
            resolved_target = reduce(math.lcm, targets)
            for v in group_violations:
                v.target_value = resolved_target

    feasible = [v for v in violations if v.classification == "FREE" or v.classification == "COUPLED"]
    infeasible = [v for v in violations if v.classification == "LOCKED"]
    return (feasible, infeasible)


def get_padding_plan(feasible: list[Violation], uf) -> dict[str, int]:
    #Build {dim_variable: target_value} for every dim_variable that needs padding.
    #For each feasible violation, find all members of its union-find group and
    #assign them the same target_value. 
    intended: dict[str, int] = {}
    for v in feasible:
        intended[v.dim_variable] = v.target_value
    plan: dict[str, int] = {}
    groups = uf.groups()
    for root, members in groups.items():
        target = None
        for m in members:
            if m in intended:
                target = intended[m]
                break
        if target is None:
            continue
        for m in members:
            plan[m] = target

    for var, target in sorted(plan.items()):
        print(f"  padding_plan: {var} \u2192 {target}")

    return plan
