"""Verified arithmetic curriculum generation from a local Ollama teacher."""
import ast
import json
import random
import re
import urllib.error
import urllib.request
from urllib.parse import urlsplit


def evaluate_expression(expression):
    """Evaluate a small integer-arithmetic expression without eval()."""
    tree = ast.parse(expression, mode="eval")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            return left * right
        raise ValueError("unsupported arithmetic expression")

    return visit(tree)


def make_expressions(count, seed):
    rng, expressions = random.Random(seed), []
    seen = set()
    while len(expressions) < count:
        a, b, c = (rng.randint(2, 20) for _ in range(3))
        op1, op2 = rng.choice(("+", "-", "*")), rng.choice(("+", "-", "*"))
        expression = f"({a} {op1} {b}) {op2} {c}"
        if expression not in seen and abs(evaluate_expression(expression)) <= 10000:
            seen.add(expression)
            expressions.append(expression)
    return expressions


def canonical_solution(expression):
    """Create a short, exact trace for the supported two-operation curriculum."""
    expected = evaluate_expression(expression)
    root = ast.parse(expression, mode="eval").body
    if not (isinstance(root, ast.BinOp) and isinstance(root.left, ast.BinOp)
            and all(isinstance(node, ast.Constant) and type(node.value) is int
                    for node in (root.left.left, root.left.right, root.right))):
        raise ValueError("unsupported curriculum expression shape")
    symbols = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*"}
    inner_op, outer_op = type(root.left.op), type(root.op)
    if inner_op not in symbols or outer_op not in symbols:
        raise ValueError("unsupported curriculum operator")
    first_value = evaluate_expression(ast.unparse(root.left))
    steps = [
        f"{root.left.left.value} {symbols[inner_op]} {root.left.right.value} = {first_value}",
        f"{first_value} {symbols[outer_op]} {root.right.value} = {expected}",
    ]
    return {"expression": expression, "steps": steps, "answer": expected}


def validate_solution(expression, record):
    """Accept only the teacher's correct answer; construct the verified trace locally."""
    canonical = canonical_solution(expression)
    if not isinstance(record, dict) or record.get("expression") != expression:
        raise ValueError("teacher returned an unexpected expression")
    compact_steps = [re.sub(r"\s+", "", step).replace("×", "*").replace("x", "*").replace("−", "-")
                     for step in record.get("steps", []) if isinstance(step, str)]
    expected_steps = [re.sub(r"\s+", "", step) for step in canonical["steps"]]
    if compact_steps and compact_steps != expected_steps:
        raise ValueError("teacher supplied an invalid arithmetic trace")
    answer = record.get("answer")
    if type(answer) is not int:
        raise ValueError("teacher answer must be an integer")
    if answer != canonical["answer"]:
        raise ValueError("teacher final answer does not match the expression")
    return canonical


class OllamaTeacher:
    """Structured local-only teacher client. No prompts leave localhost."""

    def __init__(self, model="qwen2.5-coder:7b", url="http://127.0.0.1:11434", timeout=180):
        parsed = urlsplit(url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("teacher URL must be a local HTTP Ollama endpoint")
        self.model, self.url, self.timeout = model, url.rstrip("/"), timeout

    def _solve_batch(self, expressions):
        schema = {
            "type": "object",
            "properties": {
                "solutions": {
                    "type": "array",
                    "minItems": len(expressions),
                    "maxItems": len(expressions),
                    "items": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"},
                            "answer": {"type": "integer"},
                        },
                        "required": ["expression", "answer"],
                    },
                },
            },
            "required": ["solutions"],
        }
        prompt = (
            "Calculate each integer arithmetic expression exactly. Return one solution per expression in the same order. "
            "Return the original expression and its integer answer only. Do not return steps or prose. Expressions: "
            + json.dumps(expressions)
        )
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "format": schema,
            "stream": False,
            "options": {"temperature": 0, "num_predict": max(256, len(expressions) * 96)},
        }).encode("utf-8")
        request = urllib.request.Request(
            self.url + "/api/chat", data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read())
            content = json.loads(payload["message"]["content"])
            if not isinstance(content, dict) or not isinstance(content.get("solutions"), list):
                raise ValueError("teacher response must contain a solutions list")
            solutions = content["solutions"]
            by_expression = {}
            requested = set(expressions)
            for item in solutions:
                expression = item.get("expression") if isinstance(item, dict) else None
                if expression in requested:
                    if expression in by_expression:
                        raise ValueError("teacher returned duplicate solutions for an expression")
                    by_expression[expression] = item
            if set(by_expression) != requested:
                raise ValueError("teacher omitted one or more requested expressions")
            return [validate_solution(expression, by_expression[expression]) for expression in expressions]
        except (urllib.error.URLError, TimeoutError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"local Ollama teacher failed: {exc}") from exc

    def solve(self, expressions):
        try:
            return self._solve_batch(expressions)
        except ValueError:
            solutions = []
            for expression in expressions:
                try:
                    solutions.extend(self._solve_batch([expression]))
                except ValueError:
                    continue
            return solutions
