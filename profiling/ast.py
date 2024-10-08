# The sys.settrace approach can't deal with x * y
# I didn't figure out on time if it can be solved.
# However we can build an AST tree and this should work but it's much more complicated
# and there are many operators that we need to deal with
# It's also what's used by actual profilers, so it's likely
# what I would use if I had worked on a prod grade profiler.

import ast
import time

code = """
for i in range(100):  # to get good aggregate
    preprocessed = yolo.predictor.preprocess(img)
    preds = yolo.model(preprocessed)
    postprocessed = yolo.predictor.postprocess(preds, preprocessed, img)
"""


code2 = """
x = torch.rand(500_000_000)
y = torch.rand(500_000_000)
x * y
"""

time_stats = {}


def timeit(func, func_name):
    start = time.time()
    result = func()
    end = time.time()
    elapsed = end - start

    if func_name not in time_stats:
        time_stats[func_name] = 0
    time_stats[func_name] += elapsed

    #print(f"Function '{func_name}' took {elapsed:.6f} seconds")
    return result


class TimingWrapperTransformer(ast.NodeTransformer):
    def __init__(self, wrapper_func_name='timeit'):
        self.wrapper_func_name = wrapper_func_name

    def visit_Call(self, node):
        # Try to resolve the full function name
        func_name = self._get_func_name(node.func)

        # Create a lambda function to capture the function call for timing
        new_node = ast.Call(
            func=ast.Name(id=self.wrapper_func_name, ctx=ast.Load()),  # Call timeit function
            args=[
                ast.Lambda(  # Create a lambda function to delay the actual call
                    args=ast.arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None,
                                       defaults=[]),
                    body=node
                ),
                ast.Constant(value=func_name)  # Pass the function name as an argument to timeit
            ],
            keywords=[]
        )
        # Copy line number and location information from the original node
        return ast.copy_location(new_node, node)

    def visit_BinOp(self, node):
        # This handles binary operations (e.g., a * b)
        if isinstance(node.op, ast.Mult):
            left_func_name = self._get_func_name(node.left)
            right_func_name = self._get_func_name(node.right)

            # Here we would try to add a timing decorator to a * b.
            new_node = ast.Call(
                func=ast.Name(id=self.wrapper_func_name, ctx=ast.Load()),
                args=[
                    ast.Lambda(
                        args=ast.arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[],
                                           kwarg=None, defaults=[]),
                        body=node
                    ),
                    # TODO missing reference to a/b
                ],
                keywords=[]
            )
            return ast.copy_location(new_node, node)
        return self.generic_visit(node)  # Visit other binary operations

    def _get_func_name(self, node):
        """
        Helper function to get the full function name as a string.
        This handles cases where the function is an attribute (e.g., module.func).
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_func_name(node.value)}.{node.attr}"
        else:
            return "<unknown>"

tree = ast.parse(code2)

transformer = TimingWrapperTransformer()
new_tree = transformer.visit(tree)
ast.fix_missing_locations(new_tree)

compiled_code = compile(new_tree, filename="<ast>", mode="exec")
exec(compiled_code)

# Print the accumulated time stats
print("Cumulative time spent on each function:")
for func_name, total_time in time_stats.items():
    print(f"{func_name}: {total_time:.6f} seconds")
