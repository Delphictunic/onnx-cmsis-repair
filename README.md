# CMSIS-NN Graph Repair Tool
A deterministic, static ONNX graph repair tool that automatically zero-pads weight and bias tensors to satisfy CMSIS-NN fast-path alignment constraints, without requiring retraining. The transformation preserves original model behavior while reshaping internal dimensions to fully unlock optimized kernel routing on ARM Cortex-M microcontrollers.

CMSIS-NN is Arm’s embedded neural network library that provides highly optimized integer and DSP-accelerated kernels for efficient inference on resource-constrained microcontrollers.
---

## Motivation

CMSIS-NN selects optimized kernels at runtime via wrapper functions that inspect tensor dimensions before dispatching. A model exported with channels that do not satisfy alignment requirements silently falls through to scalar fallback kernels degrading inference throughput on Cortex-M devices with no warning at compile time.

Existing toolchains do not address this gap, the standard workflow is to retrain with aligned channel counts. 

This tool performs the alignment as a post-export graph transformation, preserving model semantics while unlocking fast-path dispatch. 

---

## Novelty

- **Constraint knowledge base built from CMSIS-NN wrapper source (open source github files)** — every alignment requirement is traced to an exact routing condition in the wrapper file. knowledge_base/constraints.py contains constraints from cmsis-nn documentation, this information is not assumed and strictly coherant to the original implementation.
- **Union-Find propagation of ONNX op semantics** — dimension variables are coupled by the shape compatibility rules of each operation (node) type, so padding one tensor automatically identifies all tensors that must move together (crucial in stability and unchanged behaviour).
- **Three-class violation taxonomy** — FREE, COUPLED, and LOCKED violations are distinguished before any patch is attempted, giving a clear audit trail of what was changed and why.
- **Reshape safety detection** — tensors that feed into Reshape nodes with hardcoded shape constants are locked rather than silently patched, preventing shape mismatches downstream.

---

## Core Structure

```
cmsis-ref/
├── pipeline.py               # programmatic entry point
├── cli.py                    # command-line entry point
├── report.py                 # report formatting
├── make_test_model.py        # test ONNX generation
├── knowledge_base/
│   └── constraints.py        # CMSIS-NN alignment constraints, source-cited
├── graph/
│   ├── loader.py             # ONNX load, shape extraction, graph I/O
│   ├── dim_variables.py      # symbolic dim variable assignment
│   └── propagator.py         # union-find constraint propagation
├── analysis/
│   ├── classifier.py         # FREE / COUPLED / LOCKED violation classification
│   └── feasibility.py        # padding plan construction
└── transforms/
    ├── pad_channels.py       # weight and bias padding, Reshape lock detection
    └── validate.py           # ONNX Runtime correctness check
```

Each layer depends only on the layer below it. `transforms` calls `analysis`, `analysis` calls `graph`, `knowledge_base` is read-only data with no dependencies.

---

## Pipeline

```
load model
    ↓
assign dim variables         # "stem_weight__dim0", "stem_out__dim1", ...
    ↓
propagate constraints        # union variables by ONNX op semantics
    ↓                        # lock Reshape/Flatten outputs
classify violations          # check alignment against knowledge base
    ↓                        # FREE / COUPLED / LOCKED
check feasibility            # resolve conflicting targets via LCM
    ↓
build padding plan           # expand targets to all group members
    ↓
detect Reshape locks         # demote unsafe vars before patching
    ↓
apply padding                # zero-pad weights, biases, update group attr
    ↓
validate                     # ONNX Runtime inference check
```

---

## Constraint Knowledge Base

Constraints are derived directly from CMSIS-NN wrapper routing conditions, not from benchmarking or heuristics. Example:

```python
# arm_convolve_wrapper_s8.c BRANCH B  →  arm_convolve_1_x_n_s8
#   elif (input_dims->h == 1
#         && dilation.w == 1
#         && filter_dims->h == 1
#         && (conv_params->stride.w * input_dims->c) % 4 == 0   ← KEY
#         && input_dims->c == filter_dims->c)
#
# For stride_w == 1  → input_channels % 4 == 0
# alignment=4 satisfies all stride values.
```

Every constraint entry records `dim_index` (which weight tensor axis), `alignment`, `patchable` flag, the kernel unlocked, and the exact C source condition.

---

## Union-Find Propagation

Dimension variables are symbolic names assigned to each axis of each tensor. The propagator walks every node and unions variables that ONNX op semantics requires to be equal:

- **Conv** — `input.channels == weight.in_channels`, `weight.out_channels == output.channels`
- **Add** — all inputs and output must match on every axis
- **Concat** — all tensors must match on every axis except the concat axis
- **Reshape / Flatten** — output dims are locked (hardcoded by shape constant)

Transitivity is handled automatically. Unioning `stem_weight__dim0` with `stem_out__dim1`, then `stem_out__dim1` with `conv1_weight__dim1`, puts all three in the same group without an explicit three-way union.

---

## Violation Classification

For each weight tensor dimension that violates an alignment constraint:

{Classification} {Condition} {Action}
FREE, group size == 1, not locked, not graph I/O [patch directly]
COUPLED, group size > 1, no member locked or graph I/O [patch all group members together]
LOCKED, any member locked, or touches graph I/O, or downstream Reshape [report reason, skip]

---

## Example Output

**MobileNet-style inverted residual block** (`test_model.onnx`):

```
Node       | Op            | Constraint      | Current | Target | Status
------------------------------------------------------------------------
PW1        | Conv          | output_channels | 6       | 8      | PATCHED
DW2        | DepthwiseConv | channels        | 6       | 8      | PATCHED
PW2        | Conv          | input_channels  | 6       | 8      | PATCHED
PW2        | Conv          | output_channels | 6       | 8      | PATCHED
Expand     | Conv          | input_channels  | 6       | 8      | PATCHED
Expand     | Conv          | output_channels | 10      | 12     | PATCHED
StemConv   | Conv          | input_channels  | 3       | 4      | LOCKED: tensor in graph input or output

7 violations. 6 patched. 1 locked.

Additional tensors patched via group expansion:
  add_out   axis=1 → 8
  dw2_out   axis=1 → 8
  dw2_w     axis=1 → 8
  exp_out   axis=1 → 12
  pw1_out   axis=1 → 8
  pw2_out   axis=1 → 8
```

`StemConv input_channels = 3` is correctly locked — RGB input cannot be padded without changing the model interface. All other violations are resolved by propagating the alignment target through the union-find groups.

**Edge case — Reshape boundary** (`test_model_edge.onnx`):

```
FreeConv   | Conv | output_channels | 10 | 12 | LOCKED: downstream Reshape has hardcoded shape
```

The tool detects that `FreeConv` output feeds into a Reshape with a hardcoded shape constant `{1, 10, 256}`. Padding output channels to 12 would make the Reshape input `{1, 12, 16, 16}` incompatible with that constant. The violation is demoted to LOCKED with an explicit reason rather than producing a broken model.

---

## Scope

The tool handles standard CNN building blocks: Conv, DepthwiseConv, pointwise Conv, residual Add, branch Concat, fully connected Gemm. It is designed for the class of models deployed on Cortex-M via CMSIS-NN — MobileNet variants, lightweight ResNets, custom embedded CNNs. Transformer and RNN architectures are outside scope.

ONNX graph intermediate representation allows higher level reasoning that allows consistent and behaviour preserving model transformation. 

While currently designed for CMSIS-NN, the framework is backend agnostic and can be extended to other industry standard kernel selection and neural network optimization libraries that rely on alignment or shape based fast path routing.

