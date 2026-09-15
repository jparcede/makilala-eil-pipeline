"""Scope-aware dependency check for a notebook.

Only MODULE-LEVEL bindings count as definitions; function parameters and locals
do not leak into the global namespace. Only module-level loads are reported,
since those are what raise NameError at cell-execution time.
"""
import json, ast, builtins, sys

def module_level(tree):
    stores, loads = set(), set()
    def bind(t):
        for n in ast.walk(t):
            if isinstance(n, ast.Name): stores.add(n.id)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stores.add(node.name)
            for n in ast.walk(node):                      # globals a body reads
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                    loads.add(('deferred', n.id))
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names: stores.add((a.asname or a.name).split('.')[0]); continue
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                (stores if isinstance(n.ctx, (ast.Store, ast.Del)) else loads).add(
                    n.id if isinstance(n.ctx, ast.Load) else n.id)
            if isinstance(n, (ast.For, ast.comprehension)):
                bind(n.target)
            if isinstance(n, ast.withitem) and n.optional_vars is not None:
                bind(n.optional_vars)
            if isinstance(n, ast.ExceptHandler) and n.name: stores.add(n.name)
        for n in ast.walk(node):
            if isinstance(n, (ast.Lambda, ast.FunctionDef)):
                for a in n.args.args + n.args.kwonlyargs: stores.add(a.arg)
    loads = {l for l in loads if not isinstance(l, tuple)}
    return stores, loads

nb = json.load(open(sys.argv[1]))
cells = nb['cells']
defined = set()
problems = 0
for i, c in enumerate(cells):
    if c['cell_type'] != 'code': continue
    src = ''.join(c['source'])
    st, ld = module_level(ast.parse(src))
    missing = sorted(x for x in ld
                     if x not in defined and x not in st and not hasattr(builtins, x))
    if missing:
        print(f"  cell {i:>2}: NameError risk -> {missing}")
        problems += 1
    defined |= st
print(f"\n  cells with unresolved module-level names: {problems}")
