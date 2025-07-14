from numba import njit
import minitorch
import minitorch.fast_ops
import io
import contextlib


def log(f, title, diagnostics_func):
    f.write(f"\n## {title}\n\n")
    f.write("```\n")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        diagnostics_func()
    output = buf.getvalue()
    f.write(output if output else "No diagnostics available.")
    f.write("\n```\n")


with open("README.md", "a") as f:
    f.write("# Numba Parallel Diagnostics Report\n\n")

    # MAP
    print("MAP")
    tmap = minitorch.fast_ops.tensor_map(njit()(minitorch.operators.id))
    out, a = minitorch.zeros((10,)), minitorch.zeros((10,))
    tmap(*out.tuple(), *a.tuple())
    log(f, "MAP", lambda: tmap.parallel_diagnostics(level=3))

    # ZIP
    print("ZIP")
    out, a, b = minitorch.zeros((10,)), minitorch.zeros((10,)), minitorch.zeros((10,))
    tzip = minitorch.fast_ops.tensor_zip(njit()(minitorch.operators.eq))
    tzip(*out.tuple(), *a.tuple(), *b.tuple())
    log(f, "ZIP", lambda: tzip.parallel_diagnostics(level=3))

    # REDUCE
    print("REDUCE")
    out, a = minitorch.zeros((1,)), minitorch.zeros((10,))
    treduce = minitorch.fast_ops.tensor_reduce(njit()(minitorch.operators.add))
    treduce(*out.tuple(), *a.tuple(), 0)
    log(f, "REDUCE", lambda: treduce.parallel_diagnostics(level=3))

    # MATRIX MULTIPLY
    print("MATRIX MULTIPLY")
    out, a, b = (
        minitorch.zeros((1, 10, 10)),
        minitorch.zeros((1, 10, 20)),
        minitorch.zeros((1, 20, 10)),
    )
    tmm = minitorch.fast_ops.tensor_matrix_multiply
    tmm(*out.tuple(), *a.tuple(), *b.tuple())
    log(f, "MATRIX MULTIPLY", lambda: tmm.parallel_diagnostics(level=3))
