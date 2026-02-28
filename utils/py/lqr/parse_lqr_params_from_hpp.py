#!/usr/bin/env python3
import re
import sys
from pathlib import Path


def c_array_to_python(text: str) -> str:
    # convert C-style braces to Python brackets and ensure floats
    # remove line comments
    # keep nesting
    py = text
    # remove /* ... */ comments
    py = re.sub(r"/\*.*?\*/", "", py, flags=re.S)
    # replace braces with brackets
    py = py.replace("{", "[").replace("}", "]")
    # ensure trailing commas between numbers are preserved
    return py


def extract_constant(content: str, name: str) -> str | None:
    # regex to capture initialization of static const float name[][]... = { ... };
    # allow multiple bracketed dimensions
    pattern = rf"static\s+const\s+float\s+{re.escape(name)}\s*(?:\[[^\]]*\])+\s*=\s*(\{{.*?\}})\s*;"
    m = re.search(pattern, content, flags=re.S)
    if not m:
        return None
    return m.group(1)


def main():
    if len(sys.argv) < 2:
        print("Usage: generate_lqr_code.py <params.hpp>")
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File {path} not found")
        sys.exit(1)
    text = path.read_text()
    names = [
        'kMatureLqrPoly_whole',
        'kA_LPoly_whole',
        'kB_LPoly_whole',
        'kUdPoly_whole',
    ]
    print("# generated python arrays from params.hpp")
    print("import numpy as np")
    for name in names:
        block = extract_constant(text, name)
        if block is None:
            print(f"# WARNING: {name} not found")
            continue
        py_block = c_array_to_python(block)
        print(f"\n{name.upper()} = np.array({py_block})")

    # additionally generate code snippets for sim.py A/B matrix assignments
    def format_coeffs(arr):
        # produce multi-line literal akin to sim.py style
        # break into lines of up to 5 elements
        lines = []
        for i in range(0, len(arr), 5):
            chunk = arr[i : i + 5]
            lines.append(
                ", ".join(str(float(x)) for x in chunk)
            )
        inner = ",\n        ".join(lines)
        return "[" + inner + "]"

    # load arrays to evaluate
    import ast
    a_block = extract_constant(text, 'kA_LPoly_whole')
    b_block = extract_constant(text, 'kB_LPoly_whole')
    if a_block is not None and b_block is not None:
        a_list = ast.literal_eval(c_array_to_python(a_block))
        b_list = ast.literal_eval(c_array_to_python(b_block))
        # indices used in sim
        a_indices = [(1,4),(1,6),(3,4),(3,6),(5,4),(5,6),(7,4),(7,6),(9,4),(9,6),(9,8)]
        b_indices = [(1,0),(1,1),(1,2),(1,3),(3,0),(3,1),(3,2),(3,3),(5,0),(5,1),(5,2),(5,3),(7,0),(7,1),(7,2),(7,3),(9,0),(9,1),(9,2),(9,3)]
        print("\n# assignments for _compute_A_matrix")
        for i,j in a_indices:
            coeffs = a_list[i][j]
            print(
                f"A[{i}, {j}] = PolynomialFit.eval_poly33(\n"
                f"    np.array({format_coeffs(coeffs)}), l_l, l_r)"
            )
        print("\n# assignments for _compute_B_matrix")
        for i,j in b_indices:
            coeffs = b_list[i][j]
            print(
                f"B[{i}, {j}] = PolynomialFit.eval_poly33(\n"
                f"    np.array({format_coeffs(coeffs)}), l_l, l_r)"
            )

    # generate u_d feedforward assignments
    ud_block = extract_constant(text, 'kUdPoly_whole')
    if ud_block is not None:
        ud_list = ast.literal_eval(c_array_to_python(ud_block))
        print("\n# assignments for get_u_d")
        for i in range(len(ud_list)):
            coeffs = ud_list[i]
            print(
                f"u_d[{i}] = PolynomialFit.eval_poly33(\n"
                f"    np.array({format_coeffs(coeffs)}), l_l, l_r)"
            )

if __name__ == '__main__':
    main()
