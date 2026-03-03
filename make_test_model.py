"""
Comprehensive edge case test model for cmsis_nn_optimizer.

Covers:
  1. FREE violation          — isolated node, nothing coupled downstream
  2. COUPLED chain           — producer.out == consumer.in across 3 layers
  3. COUPLED fan-out         — one producer feeds two consumers (Add merge)
  4. LOCKED graph input      — channel dim of X is immovable
  5. LOCKED graph output     — final output channels locked
  6. Depthwise conv          — group attribute must update after padding
  7. Depthwise into pointwise coupling — DW out coupled to PW in
  8. Already aligned dims    — should produce zero violations (no false positives)
  9. Inconsistent targets    — two violations in same group want different alignments
                               (resolved by taking lcm of targets)
 10. Reshape lock            — tensor flowing through Reshape is locked

Graph (all float32, opset 11, ir_version 7):

  X [1, 3, 16, 16]
  │
  ├─ BranchA ──────────────────────────────────────────────────────────────────
  │   StemA [8,3,1,1]       → stem_a [1,8,16,16]   # case 8: aligned, no violation
  │   DW_A  [8,1,3,3] g=8   → dw_a  [1,8,16,16]   # case 6+7: depthwise
  │   PW_A  [6,8,1,1]       → pw_a  [1,6,16,16]   # case 2: COUPLED chain start
  │   MidA  [6,6,1,1]       → mid_a [1,6,16,16]   # case 2: COUPLED chain middle
  │   EndA  [6,6,1,1]       → end_a [1,6,16,16]   # case 2: COUPLED chain end
  │
  ├─ BranchB ──────────────────────────────────────────────────────────────────
  │   StemB [6,3,1,1]       → stem_b [1,6,16,16]  # case 3: fan-out start
  │   FanA  [6,6,1,1]       → fan_a  [1,6,16,16]  # case 3: fan-out consumer A
  │   FanB  [6,6,1,1]       → fan_b  [1,6,16,16]  # case 3: fan-out consumer B
  │   Add(fan_a, fan_b)     → add_b  [1,6,16,16]  # case 3: forces equality
  │
  ├─ Merge: Add(end_a, add_b) → merged [1,6,16,16]
  │
  ├─ FreeConv [10,6,1,1]    → free_out [1,10,16,16]  # case 1: FREE (out_ch=10)
  │                                                    # case 9: in_ch=6 coupled to merged
  │
  ├─ Reshape(free_out → [1,10,256]) → reshaped [1,10,256]  # case 10: locks dims
  │
  ├─ Flatten → flat [1, 2560]
  │
  └─ Gemm [5, 2560] → output [1,5]   # case 5: output locked (graph output)
                                      # case 4: input locked (Flatten output)
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

np.random.seed(42)


def cv(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def w(name, shape):
    return numpy_helper.from_array(
        np.random.randn(*shape).astype(np.float32), name=name
    )


def b(name, size):
    return numpy_helper.from_array(np.zeros(size, dtype=np.float32), name=name)


def conv(inp, wname, bname, out, name, kernel=1, pad=0, group=1):
    return helper.make_node(
        "Conv",
        inputs=[inp, wname, bname],
        outputs=[out],
        name=name,
        kernel_shape=[kernel, kernel],
        pads=[pad] * 4,
        group=group,
    )


def make_model() -> onnx.ModelProto:
    X = cv("X", [1, 3, 16, 16])

    # ── BranchA ───────────────────────────────────────────────────────────────
    n_stem_a = conv("X",      "stem_a_w", "stem_a_b", "stem_a", "StemA")
    n_dw_a   = helper.make_node(
        "Conv",
        inputs=["stem_a", "dw_a_w", "dw_a_b"],
        outputs=["dw_a"],
        name="DW_A",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        group=8,
    )
    n_pw_a  = conv("dw_a",  "pw_a_w",  "pw_a_b",  "pw_a",  "PW_A")
    n_mid_a = conv("pw_a",  "mid_a_w", "mid_a_b", "mid_a", "MidA")
    n_end_a = conv("mid_a", "end_a_w", "end_a_b", "end_a", "EndA")

    # ── BranchB ───────────────────────────────────────────────────────────────
    n_stem_b = conv("X",      "stem_b_w", "stem_b_b", "stem_b", "StemB")
    n_fan_a  = conv("stem_b", "fan_a_w",  "fan_a_b",  "fan_a",  "FanA")
    n_fan_b  = conv("stem_b", "fan_b_w",  "fan_b_b",  "fan_b",  "FanB")
    n_add_b  = helper.make_node("Add", ["fan_a", "fan_b"], ["add_b"],  name="AddB")

    # ── Merge ─────────────────────────────────────────────────────────────────
    n_merge = helper.make_node("Add", ["end_a", "add_b"], ["merged"], name="Merge")

    # ── FreeConv ──────────────────────────────────────────────────────────────
    n_free = conv("merged", "free_w", "free_b", "free_out", "FreeConv")

    # ── Reshape ───────────────────────────────────────────────────────────────
    shape_tensor = numpy_helper.from_array(
        np.array([1, 10, 256], dtype=np.int64), name="reshape_shape"
    )
    n_reshape = helper.make_node(
        "Reshape",
        inputs=["free_out", "reshape_shape"],
        outputs=["reshaped"],
        name="Reshape1",
    )

    # ── Flatten ───────────────────────────────────────────────────────────────
    n_flatten = helper.make_node(
        "Flatten",
        inputs=["reshaped"],
        outputs=["flat"],
        name="Flatten1",
        axis=1,
    )

    # ── Gemm ──────────────────────────────────────────────────────────────────
    n_gemm = helper.make_node(
        "Gemm",
        inputs=["flat", "gemm_w", "gemm_b"],
        outputs=["output"],
        name="Gemm1",
        transB=1,
    )

    intermediates = [
        cv("stem_a",   [1,  8, 16, 16]),
        cv("dw_a",     [1,  8, 16, 16]),
        cv("pw_a",     [1,  6, 16, 16]),
        cv("mid_a",    [1,  6, 16, 16]),
        cv("end_a",    [1,  6, 16, 16]),
        cv("stem_b",   [1,  6, 16, 16]),
        cv("fan_a",    [1,  6, 16, 16]),
        cv("fan_b",    [1,  6, 16, 16]),
        cv("add_b",    [1,  6, 16, 16]),
        cv("merged",   [1,  6, 16, 16]),
        cv("free_out", [1, 10, 16, 16]),
        cv("reshaped", [1, 10, 256]),
        helper.make_tensor_value_info("flat", TensorProto.FLOAT, [1, 2560]),
    ]

    graph = helper.make_graph(
        nodes=[
            n_stem_a, n_dw_a, n_pw_a, n_mid_a, n_end_a,
            n_stem_b, n_fan_a, n_fan_b, n_add_b,
            n_merge, n_free, n_reshape, n_flatten, n_gemm,
        ],
        name="edge_case_graph",
        inputs=[X],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 5])],
        initializer=[
            w("stem_a_w", (8, 3, 1, 1)), b("stem_a_b", 8),
            w("dw_a_w",   (8, 1, 3, 3)), b("dw_a_b",   8),
            w("pw_a_w",   (6, 8, 1, 1)), b("pw_a_b",   6),
            w("mid_a_w",  (6, 6, 1, 1)), b("mid_a_b",  6),
            w("end_a_w",  (6, 6, 1, 1)), b("end_a_b",  6),
            w("stem_b_w", (6, 3, 1, 1)), b("stem_b_b", 6),
            w("fan_a_w",  (6, 6, 1, 1)), b("fan_a_b",  6),
            w("fan_b_w",  (6, 6, 1, 1)), b("fan_b_b",  6),
            w("free_w",   (10, 6, 1, 1)), b("free_b",  10),
            shape_tensor,
            w("gemm_w",   (5, 2560)),
            b("gemm_b",   5),
        ],
        value_info=intermediates,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    model = make_model()
    onnx.save(model, "test_model_edge.onnx")
    print("Saved test_model_edge.onnx")
    print()
    print("Edge cases covered:")
    print("  1. FREE          — FreeConv output_channels=10, nothing downstream")
    print("  2. COUPLED chain — PW_A → MidA → EndA (3-layer chain)")
    print("  3. COUPLED fanout— StemB → (FanA, FanB) → Add")
    print("  4. LOCKED input  — StemA/StemB input_channels=3 (graph input X)")
    print("  5. LOCKED output — Gemm output_channels=5 (graph output)")
    print("  6. Depthwise     — DW_A group=8, no violation (already aligned)")
    print("  7. DW→PW couple  — DW_A out feeds PW_A in")
    print("  8. No false pos  — StemA, DW_A channels already %4 aligned")
    print("  9. Max target    — merged group has %2 and %4 requirements, max wins")
    print(" 10. Reshape lock  — free_out → Reshape → Flatten → Gemm dims locked")
    print()
    print("Update pipeline.py model_path to 'test_model_edge.onnx' before running.")