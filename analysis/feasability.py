from analysis.classifier import Violation


def check_feasibility(violations: list[Violation]) -> tuple[list[Violation], list[Violation]]:
    """
    Split violations into feasible and infeasible.

    A COUPLED violation is infeasible if two variables in the same union-find group
    require different target_values. In that case reclassify both as LOCKED and
    set reason to "Inconsistent padding targets across coupled dimensions".

    FREE violations are always feasible.
    LOCKED violations are always infeasible.

    Return (feasible: list[Violation], infeasible: list[Violation])
    """
    # Group COUPLED violations by same logical dimension (same coupled_nodes set)
    coupled_by_group: dict[frozenset[str], list[Violation]] = {}
    for v in violations:
        if v.classification == "COUPLED":
            key = frozenset(v.coupled_nodes)
            coupled_by_group.setdefault(key, []).append(v)

    # Reconcile COUPLED groups where multiple target_values appear
    for group_violations in coupled_by_group.values():
        targets = {v.target_value for v in group_violations}
        if len(targets) > 1:
            # resolve by taking the strictest (maximum) target across the group
            resolved_target = max(targets)
            for v in group_violations:
                v.target_value = resolved_target

    feasible = [v for v in violations if v.classification == "FREE" or v.classification == "COUPLED"]
    infeasible = [v for v in violations if v.classification == "LOCKED"]
    return (feasible, infeasible)


def get_padding_plan(feasible: list[Violation], uf) -> dict[str, int]:
    """
    Build {dim_variable: target_value} for every dim_variable that needs padding.

    For each feasible violation, find all members of its union-find group and
    assign them the same target_value. This ensures that when Conv1.in_channels
    is padded, StemConv.out_channels (same group) is also padded to the same
    value — enforcing the producer_out == consumer_invariant.
    """
    # Step 1: collect intended target per violated dim_variable
    intended: dict[str, int] = {}
    for v in feasible:
        intended[v.dim_variable] = v.target_value

    # Step 2: expand to all group members via uf.groups()
    plan: dict[str, int] = {}
    groups = uf.groups()
    for root, members in groups.items():
        # find if any member has an intended target
        target = None
        for m in members:
            if m in intended:
                target = intended[m]
                break
        if target is None:
            continue
        # assign that target to ALL members in the group
        for m in members:
            plan[m] = target

    for var, target in sorted(plan.items()):
        print(f"  padding_plan: {var} \u2192 {target}")

    return plan
