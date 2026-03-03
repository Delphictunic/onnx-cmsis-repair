"""
Command-line entry point. Parses arguments and delegates to pipeline.run_pipeline().
"""

import argparse
from pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair ONNX model weight dimensions for CMSIS-NN fast-path compatibility."
    )
    parser.add_argument("model", help="path to input .onnx")
    parser.add_argument("--output", default=None, help="path to write patched model")
    parser.add_argument("--report-only", action="store_true", help="classify violations only, no patching")
    parser.add_argument("--export-report", default=None, help="path to write JSON report")
    args = parser.parse_args()

    output = args.output or args.model.replace(".onnx", "_patched.onnx")

    run_pipeline(
        model_path=args.model,
        output_path=output,
        report_json_path=args.export_report,
        report_only=args.report_only,
    )


if __name__ == "__main__":
    main()