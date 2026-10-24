"""Expand embedded CUDA strings without executing submission code."""
import ast
import base64
from pathlib import Path
import sys
import zlib

class Expand(ast.NodeTransformer):
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "_zsrc" and len(node.args) == 1 and isinstance(node.args[0], ast.Constant):
            return ast.copy_location(ast.Constant(zlib.decompress(base64.b64decode(node.args[0].value)).decode()), node)
        return self.generic_visit(node)

if __name__ == "__main__":
    source, destination = map(Path, sys.argv[1:])
    tree = Expand().visit(ast.parse(source.read_text()))
    destination.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n")
