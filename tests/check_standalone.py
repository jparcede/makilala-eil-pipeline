"""Flag tracked scripts that reference names they never define: notebook cell
fragments that cannot run on their own."""
import ast, builtins, sys, os
BUILT=set(dir(builtins))
def check(path):
    try: tree=ast.parse(open(path,encoding='utf-8').read())
    except SyntaxError as e: return ["SYNTAX: "+str(e)]
    defined=set(); used={}
    class V(ast.NodeVisitor):
        def visit_Import(self,n):
            for a in n.names: defined.add((a.asname or a.name).split('.')[0])
        def visit_ImportFrom(self,n):
            for a in n.names: defined.add(a.asname or a.name)
        def visit_FunctionDef(self,n):
            defined.add(n.name)
            for a in n.args.args+n.args.kwonlyargs: defined.add(a.arg)
            if n.args.vararg: defined.add(n.args.vararg.arg)
            if n.args.kwarg: defined.add(n.args.kwarg.arg)
            self.generic_visit(n)
        visit_AsyncFunctionDef=visit_FunctionDef
        def visit_ClassDef(self,n): defined.add(n.name); self.generic_visit(n)
        def visit_Name(self,n):
            if isinstance(n.ctx,(ast.Store,)): defined.add(n.id)
            else: used.setdefault(n.id,n.lineno)
        def visit_comprehension(self,n):
            self.visit(n.target); self.generic_visit(n)
        def visit_ExceptHandler(self,n):
            if n.name: defined.add(n.name)
            self.generic_visit(n)
        def visit_With(self,n):
            for it in n.items:
                if it.optional_vars: self.visit(it.optional_vars)
            self.generic_visit(n)
        def visit_Global(self,n): defined.update(n.names)
    V().visit(tree)
    miss=sorted(k for k in used if k not in defined and k not in BUILT and k!='__file__')
    return miss
rows=[]
for f in sorted(os.popen("git ls-files '*.py'").read().split()):
    m=check(f)
    rows.append((f,m))
print(f"{'file':<52}{'status'}")
print("-"*96)
for f,m in rows:
    if m: print(f"  {f:<50}NEEDS A HOST NOTEBOOK -> {', '.join(m[:6])}")
for f,m in rows:
    if not m: print(f"  {f:<50}standalone")
