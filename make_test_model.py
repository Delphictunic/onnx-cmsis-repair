"""
Generates test_model_mixed.onnx — a mixed architecture test model with
varied channel counts to avoid trivial all-8 padding targets.

Architecture:
  X [1, 3, 32, 32]
  │
  ├─ StemConv  [10, 3, 3, 3]    → stem  [1, 10, 16, 16]   # out=10 → target 12
  │
  ├─ BranchA
  │   DW_A     [10, 1, 3, 3] g=10 → dw_a [1, 10, 16, 16]  # depthwise, ch=10 → 12
  │   PW_A     [14, 10, 1, 1]   → pw_a  [1, 14, 16, 16]   # out=14 → target 16
  │   MidA     [14, 14, 1, 1]   → mid_a [1, 14, 16, 16]   # in=14 → target 16
  │
  ├─ BranchB
  │   PW_B     [10, 10, 1, 1]   → pw_b  [1, 10, 16, 16]   # out=10 → target 12
  │   ConvB    [14, 10, 1, 1]   → conv_b [1, 14, 16, 16]  # out=14 → target 16
  │
  ├─ Add(mid_a, conv_b)         → merged [1, 14, 16, 16]
  │
  ├─ ProjConv  [6, 14, 1, 1]    → proj  [1, 6, 8, 8]      # out=6 → target 8
  │
  ├─ GlobalAveragePool          → gap   [1, 6, 1, 1]
  │
  └─ Gemm      [5, 6]           → output [1, 5]            # in=6 → target 8

Violations:
  StemConv   output_channels=10  → 12  (COUPLED: feeds DW_A and PW_B)
  DW_A       channels=10         → 12  (COUPLED: coupled to stem output)
  PW_A       input_channels=10   → 12  (COUPLED: coupled to DW_A output)
  PW_A       output_channels=14  → 16  (COUPLED: coupled to MidA)
  MidA       input_channels=14   → 16  (COUPLED: coupled to PW_A output)
  MidA       output_channels=14  → 16  (COUPLED: coupled to Add)
  PW_B       input_channels=10   → 12  (COUPLED: coupled to stem output)
  PW_B       output_channels=10  → 12  (COUPLED: coupled to ConvB input)
  ConvB      input_channels=10   → 12  (COUPLED: coupled to PW_B output)
  ConvB      output_channels=14  → 16  (COUPLED: coupled to Add)
  ProjConv   input_channels=14   → 16  (COUPLED: coupled to Add output)
  ProjConv   output_channels=6   → 8   (COUPLED: coupled to Gemm input)
  Gemm       input_features=6    → 8   (COUPLED: coupled to ProjConv output)
  StemConv   input_channels=3    → 4   (LOCKED: graph input X)
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

np.random.seed(7)


def cv(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def w(name, shape):
    return numpy_helper.from_array(
        np.random.randn(*shape).astype(np.float32), name=name
    )


def b(name, size):
    return numpy_helper.from_array(np.zeros(size, dtype=np.float32), name=name)


def conv(inp, wname, bname, out, name, kernel=1, pad=0, stride=1, group=1):
    return helper.make_node(
        "Conv",
        inputs=[inp, wname, bname],
        outputs=[out],
        name=name,
        kernel_shape=[kernel, kernel],
        pads=[pad] * 4,
        strides=[stride, stride],
        group=group,
    )


def make_mixed_model() -> onnx.ModelProto:
    X = cv("X", [1, 3, 32, 32])

    # ── StemConv ──────────────────────────────────────────────────────────────
    n_stem = conv("X", "stem_w", "stem_b", "stem", "StemConv",
                  kernel=3, pad=1, stride=2)

    # ── BranchA: DW → PW → Mid ────────────────────────────────────────────────
    n_dw_a = helper.make_node(
        "Conv",
        inputs=["stem", "dw_a_w", "dw_a_b"],
        outputs=["dw_a"],
        name="DW_A",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        group=10,
    )
    n_pw_a  = conv("dw_a",  "pw_a_w",  "pw_a_b",  "pw_a",  "PW_A")
    n_mid_a = conv("pw_a",  "mid_a_w", "mid_a_b", "mid_a", "MidA")

    # ── BranchB: PW → Conv ────────────────────────────────────────────────────
    n_pw_b   = conv("stem",  "pw_b_w",  "pw_b_b",  "pw_b",  "PW_B")
    n_conv_b = conv("pw_b",  "conv_b_w","conv_b_b","conv_b", "ConvB")

    # ── Merge ─────────────────────────────────────────────────────────────────
    n_merge = helper.make_node("Add", ["mid_a", "conv_b"], ["merged"], name="Merge")

    # ── ProjConv ──────────────────────────────────────────────────────────────
    n_proj = conv("merged", "proj_w", "proj_b", "proj", "ProjConv",
                  kernel=1, pad=0, stride=2)

    # ── GlobalAveragePool ─────────────────────────────────────────────────────
    n_gap = helper.make_node(
        "GlobalAveragePool",
        inputs=["proj"],
        outputs=["gap"],
        name="GAP",
    )

    # ── Flatten before Gemm ───────────────────────────────────────────────────
    n_flat = helper.make_node(
        "Flatten",
        inputs=["gap"],
        outputs=["flat"],
        name="Flat1",
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

    graph = helper.make_graph(
        nodes=[
            n_stem,
            n_dw_a, n_pw_a, n_mid_a,
            n_pw_b, n_conv_b,
            n_merge,
            n_proj, n_gap, n_flat, n_gemm,
        ],
        name="mixed_graph",
        inputs=[X],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 5])],
        initializer=[
            w("stem_w",   (10,  3, 3, 3)), b("stem_b",   10),
            w("dw_a_w",   (10,  1, 3, 3)), b("dw_a_b",   10),
            w("pw_a_w",   (14, 10, 1, 1)), b("pw_a_b",   14),
            w("mid_a_w",  (14, 14, 1, 1)), b("mid_a_b",  14),
            w("pw_b_w",   (10, 10, 1, 1)), b("pw_b_b",   10),
            w("conv_b_w", (14, 10, 1, 1)), b("conv_b_b", 14),
            w("proj_w",   ( 6, 14, 1, 1)), b("proj_b",    6),
            w("gemm_w",   ( 5,  6)),
            b("gemm_b",   5),
        ],
        value_info=[
            cv("stem",   [1, 10, 16, 16]),
            cv("dw_a",   [1, 10, 16, 16]),
            cv("pw_a",   [1, 14, 16, 16]),
            cv("mid_a",  [1, 14, 16, 16]),
            cv("pw_b",   [1, 10, 16, 16]),
            cv("conv_b", [1, 14, 16, 16]),
            cv("merged", [1, 14, 16, 16]),
            cv("proj",   [1,  6,  8,  8]),
            cv("gap",    [1,  6,  1,  1]),
            helper.make_tensor_value_info("flat", TensorProto.FLOAT, [1, 6]),
        ],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    model = make_mixed_model()
    onnx.save(model, "test_model_mixed.onnx")
    print("Saved test_model_mixed.onnx")
    print()
    print("Expected targets (not all 8):")
    print("  StemConv / DW_A / PW_B  channels=10  → 12")
    print("  PW_A / MidA / ConvB     channels=14  → 16")
    print("  ProjConv / Gemm         channels=6   → 8")
    print("  StemConv input_channels=3            → LOCKED (graph input)")