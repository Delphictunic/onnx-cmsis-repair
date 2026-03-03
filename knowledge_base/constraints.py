"""
CMSIS-NN Kernel Routing Constraints
====================================
This file encodes the EXACT preconditions that CMSIS-NN wrapper functions
use to select optimized kernels over fallback implementations.

Every constraint entry is traced to a specific source file and the routing
condition that gates the fast path. Where multiple backends exist (DSP, MVE,
generic), the MOST RESTRICTIVE common condition is recorded so that padding
guarantees the fast path on ALL supported cores.

Source files analyzed:
  - arm_convolve_wrapper_s8.c        (V.2.5.1, 04 Nov 2024)
  - arm_convolve_wrapper_s4.c        (V.1.2.0, 27 May 2024)
  - arm_convolve_wrapper_s16.c       (V.3.0.0, 23 Apr 2024)
  - arm_depthwise_conv_wrapper_s8.c  (V.2.2.1, 17 Jan 2025)
  - arm_depthwise_conv_wrapper_s4.c  (V.1.0.0, 30 Oct 2023)
  - arm_depthwise_conv_wrapper_s16.c (V.1.1.0, 20 Jan 2023)
  - arm_convolve_1_x_n_s8.c         (stride*ch % 4 constraint detail)
  - arm_convolve_1_x_n_s4.c         (stride*ch % 4 constraint detail)
  - arm_depthwise_conv_s8_opt.c      (inner loop structure)
  - arm_depthwise_conv_3x3_s8.c     (inner loop structure)
  - arm_depthwise_conv_fast_s16.c   (inner loop structure)
  - arm_depthwise_conv_s4_opt.c     (inner loop structure)

How to read this file
---------------------
Each op type maps to a dict of named constraints. Each constraint has:

  dim_index   : which axis of the weight/filter tensor the constraint acts on
                  0 = output_channels  (N dim of weight [out_ch, in_ch, kH, kW])
                  1 = input_channels   (C dim of weight [out_ch, in_ch, kH, kW])
                NOTE: DepthwiseConv weight shape is [out_ch, 1, kH, kW].
                  Axis 0 IS the channel count. There is no meaningful axis 1
                  (it is always 1 per group). Only axis 0 is constrained.

  alignment   : value % alignment == 0 required for the fast path.
                None = no alignment requirement.

  patchable   : True  = can be fixed by zero-padding the weight tensor axis.
                False = depends on runtime params (stride, dilation, padding,
                        kernel size) that cannot be changed by static graph repair.

  kernel_path : which optimised kernel is unlocked when this condition holds.

  description : source citation with the EXACT condition from the C source.
"""

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _c(dim_index, alignment, patchable, kernel_path, description):
    return {
        "dim_index":   dim_index,
        "alignment":   alignment,
        "patchable":   patchable,
        "kernel_path": kernel_path,
        "description": description,
    }


# ---------------------------------------------------------------------------
# Constraint tables — one entry per op variant per constraint
# ---------------------------------------------------------------------------

_CONSTRAINTS = {

    # =========================================================================
    # Conv_s8  (arm_convolve_wrapper_s8)
    # =========================================================================
    #
    # WRAPPER ROUTING (arm_convolve_wrapper_s8.c):
    #
    # BRANCH A  →  arm_convolve_1x1_s8_fast  or  arm_convolve_1x1_s8
    #   if (padding.w == 0 && padding.h == 0
    #       && filter_dims->w == 1 && filter_dims->h == 1
    #       && dilation.w == 1 && dilation.h == 1
    #       && input_dims->c == filter_dims->c)          ← non-grouped only
    #     stride == 1  →  arm_convolve_1x1_s8_fast
    #     stride != 1  →  arm_convolve_1x1_s8
    #
    # BRANCH B  →  arm_convolve_1_x_n_s8
    #   elif (input_dims->h == 1
    #         && dilation.w == 1
    #         && filter_dims->h == 1
    #         && (conv_params->stride.w * input_dims->c) % 4 == 0   ← KEY
    #         && input_dims->c == filter_dims->c)
    #
    # BRANCH C  →  arm_convolve_s8  (generic fallback)
    #
    # The ONLY dimension-based alignment constraint that is statically patchable
    # is in BRANCH B: (stride_w * input_channels) % 4 == 0.
    # For stride_w == 1  → input_channels % 4 == 0
    # For stride_w == 2  → input_channels % 2 == 0
    # We conservatively record alignment=4 (satisfies all stride values).
    #
    # arm_convolve_s8.c also states: "Optimal use case for DSP/MVE is when
    # input and output channels are multiples of 4 or at least greater than 4."
    # This drives output_channels alignment=4 as a performance (not routing) constraint.

    "Conv_s8": {

        "input_channels": _c(
            dim_index   = 1,
            alignment   = 4,
            patchable   = True,
            kernel_path = "arm_convolve_1_x_n_s8",
            description = (
                "arm_convolve_wrapper_s8.c BRANCH B: "
                "(conv_params->stride.w * input_dims->c) % 4 == 0. "
                "For stride_w=1 this is input_channels % 4 == 0. "
                "arm_convolve_s8.c: 'optimal when channels are multiples of 4'."
            ),
        ),

        "output_channels": _c(
            dim_index   = 0,
            alignment   = 4,
            patchable   = True,
            kernel_path = "arm_convolve_s8",
            description = (
                "arm_convolve_s8.c: 'Optimal use case for DSP/MVE implementation "
                "is when input and output channels are multiples of 4 or at least "
                "greater than 4.' Output channels = RHS row count in mat-mult kernels."
            ),
        ),
    },

    # =========================================================================
    # Conv_s4  (arm_convolve_wrapper_s4)
    # =========================================================================
    #
    # WRAPPER ROUTING (arm_convolve_wrapper_s4.c):
    #
    # BRANCH A  →  arm_convolve_1x1_s4_fast  or  arm_convolve_1x1_s4
    #   if (padding.w == 0 && padding.h == 0
    #       && filter_dims->w == 1 && filter_dims->h == 1
    #       && dilation.w == 1 && dilation.h == 1)
    #
    # BRANCH B  →  arm_convolve_1_x_n_s4
    #   elif (input_dims->h == 1
    #         && dilation.w == 1
    #         && filter_dims->h == 1
    #         && (conv_params->stride.w * input_dims->c) % 4 == 0   ← KEY
    #         && input_dims->c == filter_dims->c)
    #
    # BRANCH C (MVE only)  →  arm_convolve_even_s4
    #   elif ((filter_dims->h * filter_dims->w * input_dims->c) & 0x1) == 0
    #   i.e. rhs_cols = kH*kW*input_channels must be EVEN.
    #   arm_convolve_even_s4 hard-returns ARG_ERROR if rhs_cols is odd.
    #
    # BRANCH D  →  arm_convolve_s4  (generic fallback)

    "Conv_s4": {

        "input_channels": _c(
            dim_index   = 1,
            alignment   = 4,
            patchable   = True,
            kernel_path = "arm_convolve_1_x_n_s4",
            description = (
                "arm_convolve_wrapper_s4.c BRANCH B: "
                "(conv_params->stride.w * input_dims->c) % 4 == 0. "
                "Identical gate to s8 variant."
            ),
        ),

        "output_channels": _c(
            dim_index   = 0,
            alignment   = 4,
            patchable   = True,
            kernel_path = "arm_convolve_s4",
            description = (
                "arm_convolve_s4.c: 'Optimal use case for DSP/MVE is when input "
                "and output channels are multiples of 4 or at least greater than 4.' "
                "Mirrors the s8 recommendation."
            ),
        ),

        "rhs_cols_even": _c(
            dim_index   = 1,
            alignment   = 2,
            patchable   = True,
            kernel_path = "arm_convolve_even_s4",
            description = (
                "arm_convolve_wrapper_s4.c BRANCH C (MVE only): "
                "((filter_dims->h * filter_dims->w * input_dims->c) & 0x1) == 0. "
                "arm_convolve_even_s4 returns ARG_ERROR if rhs_cols is odd. "
                "When kH*kW is odd, input_channels must be even to satisfy this. "
                "When kH*kW is even, constraint is always met regardless of channels."
            ),
        ),
    },

    # =========================================================================
    # Conv_s16  (arm_convolve_wrapper_s16)
    # =========================================================================
    #
    # WRAPPER ROUTING (arm_convolve_wrapper_s16.c):
    #
    # PASS-THROUGH — the wrapper unconditionally calls arm_convolve_s16.
    # There is NO routing branch. No hard alignment gate exists.
    #
    # arm_convolve_s16.c comment: "Optimal use case for DSP/MVE is when input
    # and output channels are multiples of 4 or at least greater than 4."
    # This is a performance hint only.

    "Conv_s16": {

        "input_channels": _c(
            dim_index   = 1,
            alignment   = 4,
            patchable   = True,
            kernel_path = "arm_convolve_s16",
            description = (
                "arm_convolve_wrapper_s16.c: pass-through, no routing branch. "
                "arm_convolve_s16.c: 'optimal when channels are multiples of 4'. "
                "No hard gate — alignment=4 is a performance recommendation."
            ),
        ),

        "output_channels": _c(
            dim_index   = 0,
            alignment   = 4,
            patchable   = True,
            kernel_path = "arm_convolve_s16",
            description = (
                "arm_convolve_s16.c: same 'multiples of 4' performance recommendation "
                "for output channels."
            ),
        ),
    },

    # =========================================================================
    # DepthwiseConv_s8  (arm_depthwise_conv_wrapper_s8)
    # =========================================================================
    #
    # WRAPPER ROUTING (arm_depthwise_conv_wrapper_s8.c V.2.2.1):
    #
    # [MVE only] SPECIAL PATH — convert to regular conv
    #   if (input_dims->c == 1
    #       && output_dims->c > CONVERT_DW_CONV_WITH_ONE_INPUT_CH_AND_OUTPUT_CH_ABOVE_THRESHOLD)
    #   → arm_depthwise_conv_to_conv_s8  (delegates to arm_convolve_wrapper_s8)
    #
    # OPTIMISED PATH
    #   if (ch_mult == 1
    #       && input_dims->n == 1
    #       && dilation.w == 1
    #       && dilation.h == 1)
    #     [non-MVE only] if (filter_dims->w == 3 && filter_dims->h == 3
    #                        && padding.h <= 1 && padding.w <= 1):
    #       → arm_depthwise_conv_3x3_s8
    #     else:
    #       → arm_depthwise_conv_s8_opt
    #
    # FALLBACK
    #   else → arm_depthwise_conv_s8  (generic, any dimensions)
    #
    # INNER LOOP ANALYSIS:
    #
    # arm_depthwise_conv_3x3_s8.c:
    #   for (; in_ch <= (input_ch - 4); in_ch += 4) { ... }   ← 4-wide loop
    #   // Leftover: for (; in_ch < input_ch; ++in_ch)         ← scalar tail
    #   → channels % 4 == 0 eliminates the scalar tail.
    #
    # arm_depthwise_conv_s8_opt.c (DSP path):
    #   row_count = output_ch / 4;   ← 4 channels per iteration
    #   row_count = output_ch & 0x3; ← scalar tail
    #   → channels % 4 == 0 eliminates scalar tail.
    #
    # arm_depthwise_conv_s8_opt.c (MVE path):
    #   active_ch = MIN(CH_IN_BLOCK_MVE, remaining_ch)
    #   Processes in blocks; any multiple of 4 is optimal.
    #
    # For DepthwiseConv with ch_mult==1 (the only patchable case):
    #   input_channels == output_channels
    #   ONNX weight shape: [out_channels, 1, kH, kW]
    #   BOTH input and output channel counts map to axis 0 of the weight tensor.

    "DepthwiseConv_s8": {

        "channels": _c(
            dim_index   = 0,
            alignment   = 4,
            patchable   = True,
            kernel_path = "arm_depthwise_conv_s8_opt / arm_depthwise_conv_3x3_s8",
            description = (
                "arm_depthwise_conv_wrapper_s8.c optimised branch: ch_mult==1, n==1, "
                "dilation==1. "
                "arm_depthwise_conv_3x3_s8.c: inner loop 'for (; in_ch <= input_ch-4; in_ch+=4)'. "
                "arm_depthwise_conv_s8_opt.c (DSP): row_count = output_ch / 4, tail at output_ch & 0x3. "
                "channels % 4 == 0 eliminates scalar tail in both kernels. "
                "Weight shape [out_ch, 1, kH, kW]; ch_mult==1 means in_ch==out_ch, both at axis 0."
            ),
        ),
    },

    # =========================================================================
    # DepthwiseConv_s4  (arm_depthwise_conv_wrapper_s4)
    # =========================================================================
    #
    # WRAPPER ROUTING (arm_depthwise_conv_wrapper_s4.c):
    #
    # OPTIMISED PATH
    #   if (ch_mult == 1
    #       && input_dims->n == 1
    #       && dilation.w == 1
    #       && dilation.h == 1)
    #   → arm_depthwise_conv_s4_opt
    #
    # FALLBACK
    #   else → arm_depthwise_conv_s4
    #
    # arm_depthwise_conv_s4_opt.c (DSP path):
    #   row_count = output_ch / 4;
    #   Tail handled separately.
    #   → channels % 4 == 0 for optimal throughput.

    "DepthwiseConv_s4": {

        "channels": _c(
            dim_index   = 0,
            alignment   = 4,
            patchable   = True,
            kernel_path = "arm_depthwise_conv_s4_opt",
            description = (
                "arm_depthwise_conv_wrapper_s4.c optimised branch: ch_mult==1, n==1, "
                "dilation==1. "
                "arm_depthwise_conv_s4_opt.c: row_count = output_ch / 4 (DSP loop). "
                "channels % 4 == 0 eliminates scalar tail. "
                "Weight shape [out_ch, 1, kH, kW]; channels at axis 0."
            ),
        ),
    },

    # =========================================================================
    # DepthwiseConv_s16  (arm_depthwise_conv_wrapper_s16)
    # =========================================================================
    #
    # WRAPPER ROUTING (arm_depthwise_conv_wrapper_s16.c):
    #
    #   if USE_FAST_DW_CONV_S16_FUNCTION(dw_conv_params, filter_dims, input_dims):
    #     → arm_depthwise_conv_fast_s16
    #   else:
    #     → arm_depthwise_conv_s16
    #
    # The macro USE_FAST_DW_CONV_S16_FUNCTION gates on ch_mult==1 and that
    # a DSP/MVE core is present and buffer is available.
    #
    # arm_depthwise_conv_fast_s16.c (DSP path):
    #   row_count = output_ch / 4;    ← 4 channels per iteration
    #   row_count = output_ch & 0x3;  ← scalar tail
    #   → channels % 4 == 0 for optimal throughput.
    #
    # arm_depthwise_conv_fast_s16.c (MVE path):
    #   Processes in 4-wide int16 vector loads/stores.

    "DepthwiseConv_s16": {

        "channels": _c(
            dim_index   = 0,
            alignment   = 4,
            patchable   = True,
            kernel_path = "arm_depthwise_conv_fast_s16",
            description = (
                "arm_depthwise_conv_wrapper_s16.c: fast path via USE_FAST_DW_CONV_S16_FUNCTION "
                "(ch_mult==1, DSP/MVE present). "
                "arm_depthwise_conv_fast_s16.c DSP loop: row_count = output_ch / 4, "
                "tail at output_ch & 0x3. MVE path: 4-wide int16 vectors. "
                "channels % 4 == 0 eliminates scalar tail. "
                "Weight shape [out_ch, 1, kH, kW]; channels at axis 0."
            ),
        ),
    },
}


# ---------------------------------------------------------------------------
# Backwards-compatible aliases — ONNX op type strings → s8 default variant
# ---------------------------------------------------------------------------

_CONSTRAINTS["Conv"]          = _CONSTRAINTS["Conv_s8"]
_CONSTRAINTS["DepthwiseConv"] = _CONSTRAINTS["DepthwiseConv_s8"]
_CONSTRAINTS["Gemm"]          = {}
_CONSTRAINTS["MaxPool"]       = {}
_CONSTRAINTS["AveragePool"]   = {}


# ---------------------------------------------------------------------------
# Public accessor API
# ---------------------------------------------------------------------------

def supported_ops():
    """Return list of op type keys with constraint entries."""
    return list(_CONSTRAINTS.keys())


def get_constraints(op_type):
    """Return all constraints for op_type. Empty dict if unknown."""
    return _CONSTRAINTS.get(op_type, {})


def get_patchable_constraints(op_type):
    """Return only constraints where patchable==True."""
    return {
        name: c
        for name, c in _CONSTRAINTS.get(op_type, {}).items()
        if c["patchable"]
    }


def get_non_patchable_constraints(op_type: str) -> dict:
    return {
        name: c
        for name, c in _CONSTRAINTS.get(op_type, {}).items()
        if not c.get("patchable", True)
    }


def get_alignment(op_type, constraint_name):
    """Return alignment integer for a constraint, or None if not found."""
    c = _CONSTRAINTS.get(op_type, {}).get(constraint_name)
    return c["alignment"] if c else None


def get_dim_index(op_type, constraint_name):
    """Return weight tensor axis index for a constraint, or None."""
    c = _CONSTRAINTS.get(op_type, {}).get(constraint_name)
    return c["dim_index"] if c else None


def get_max_value(op_type, constraint_name):
    """Upper bound for a constraint value. Currently None for all entries."""
    return None


def is_patchable(op_type, constraint_name):
    """True if this constraint can be fixed by zero-padding the weight axis."""
    c = _CONSTRAINTS.get(op_type, {}).get(constraint_name)
    return c["patchable"] if c else False


def describe(op_type, constraint_name):
    """Return the source citation string for a constraint."""
    c = _CONSTRAINTS.get(op_type, {}).get(constraint_name)
    return c["description"] if c else ""


def get_kernel_path(op_type, constraint_name):
    """Return the name of the optimised kernel gated by this constraint."""
    c = _CONSTRAINTS.get(op_type, {}).get(constraint_name)
    return c["kernel_path"] if c else ""


CONSTRAINTS = _CONSTRAINTS  # backwards-compatible alias


# ---------------------------------------------------------------------------
# Self-test / documentation dump
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for op in supported_ops():
        print(f"\n{'='*72}")
        print(f"  Op: {op}")
        print(f"{'='*72}")
        for name, c in get_constraints(op).items():
            print(f"\n  [{name}]")
            print(f"    dim_index  : {c['dim_index']}")
            print(f"    alignment  : {c['alignment']}")
            print(f"    patchable  : {c['patchable']}")
            print(f"    kernel     : {c['kernel_path']}")
            print(f"    source     : {c['description'][:100]}...")
