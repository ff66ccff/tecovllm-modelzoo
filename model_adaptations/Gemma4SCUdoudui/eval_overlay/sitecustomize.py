"""Explicit evaluation-client tokenizer normalization; no model overlay."""
import os

if os.environ.get("GEMMA4_EVAL_TOKENIZER") == "1":
    import ast
    from pathlib import Path
    import sys

    try:
        source = Path(__file__).resolve().parents[1] / "overlay/gemma4_shim.py"
        tree = ast.parse(source.read_text(), filename=str(source))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "patch_tokenizer_extra_special_tokens"]
        if len(functions) != 1:
            raise RuntimeError("expected one existing tokenizer normalization function")
        # Reuse the exact existing function in isolation. Importing the whole
        # model shim would also import its optional vLLM config-parser module.
        namespace = {}
        isolated = ast.Module(body=functions, type_ignores=[])
        exec(compile(isolated, str(source), "exec"), namespace)
        namespace["patch_tokenizer_extra_special_tokens"]()
    except Exception as exc:
        sys.stderr.write(f"FATAL: Gemma evaluation tokenizer initialization failed: {exc}\n")
        raise SystemExit(1) from exc
