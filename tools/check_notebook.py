"""Static check for a Colab notebook: are all names defined before they are used?

Why this exists: a patch to finetune_v4.ipynb built a modified cell but never wrote it
back, so `INDOOR_FP_CAP` was USED in the training loop and never DEFINED. Every code cell
still compiled -- `compile()` checks syntax, and an undefined name is a runtime NameError,
not a SyntaxError. The failure surfaced on Colab, after the bundle had been unzipped and
the model warm-started, i.e. at the most expensive possible moment.

Cells are treated as executing top to bottom in one namespace, which is how Run All works.
The check is deliberately over-permissive about what counts as "defined" (function
arguments and comprehension targets are added to the global set) because a false alarm
that trains people to ignore the tool is worse than a missed edge case. It is aimed at
exactly one bug: a name that no cell ever binds.

Usage:
    python -m tools.check_notebook stages/finetune_ball_model/finetune_v4.ipynb
"""
from __future__ import annotations

import argparse
import ast
import builtins
import json
from pathlib import Path

# Colab-injected and environment names that no cell assigns
PRESUPPLIED = {"get_ipython", "In", "Out", "exit", "quit", "display", "__file__"}


def _bound_names(tree: ast.AST) -> set[str]:
    """Every name this tree binds, at any scope. Over-inclusive on purpose."""
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            out.update(n.names)
        elif isinstance(n, ast.arguments):
            for a in (*n.posonlyargs, *n.args, *n.kwonlyargs):
                out.add(a.arg)
            for a in (n.vararg, n.kwarg):
                if a:
                    out.add(a.arg)
    return out


def _loaded_names(tree: ast.AST) -> list[tuple[str, int]]:
    return [(n.id, n.lineno) for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)]


def check(path: Path) -> int:
    nb = json.loads(path.read_text(encoding="utf-8"))
    known = set(dir(builtins)) | PRESUPPLIED
    problems: list[str] = []

    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = "\n".join(l for l in "".join(cell["source"]).splitlines()
                        if not l.strip().startswith(("!", "%")))
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            problems.append(f"cell {i}: SyntaxError line {e.lineno}: {e.msg}")
            continue
        # A cell's own bindings count for the whole cell: notebooks routinely call a
        # helper defined further down, and the loop below is not flow-sensitive.
        bound = _bound_names(tree)
        for name, lineno in _loaded_names(tree):
            if name not in known and name not in bound:
                problems.append(f"cell {i} line {lineno}: '{name}' is used but never defined")
        known |= bound

    if problems:
        print(f"{path.name}: {len(problems)} problem(s)")
        for p in problems:
            print(f"  {p}")
        return 1
    print(f"{path.name}: OK - every name is defined before use "
          f"({sum(c.get('cell_type') == 'code' for c in nb['cells'])} code cells)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebooks", nargs="+", type=Path)
    a = ap.parse_args(argv)
    return max(check(n) for n in a.notebooks)


if __name__ == "__main__":
    raise SystemExit(main())
