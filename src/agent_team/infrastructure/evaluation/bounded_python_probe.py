"""Interpret a small Python subset for hidden evaluation behavior probes.

Candidate source is never imported or executed. Only literals, containers,
local/self assignments, conditionals, returns, bool/set construction, JSON
serialization and list/set insertion have semantics here. Other syntax fails
closed. The source and evaluation budgets bound recursive expression walks.
"""

import ast
import json
from typing import cast

MAX_PROBE_NODES = 512
MAX_PROBE_STEPS = 1_024
MAX_PROBE_JSON_INDENT = 8


class BoundedPythonProbe:
    """Evaluate fixture class methods without granting Python capabilities."""

    def __init__(self, source: str, class_name: str) -> None:
        """Parse a bounded class and harmless module declarations."""
        tree = ast.parse(source)
        if sum(1 for _ in ast.walk(tree)) > MAX_PROBE_NODES:
            raise ValueError("Source exceeds the probe node budget.")
        # Compilation validates duplicate arguments and invalid AST contexts.
        # The resulting code object is discarded and is never executed.
        compile(tree, "<evaluation-probe>", "exec")
        self._steps = MAX_PROBE_STEPS
        self._imports: dict[str, str] = {}
        self._methods: dict[str, ast.FunctionDef] = {}
        self._instance = object()
        self.state: dict[str, object] = {}
        classes = [
            item for item in tree.body if isinstance(item, ast.ClassDef)
        ]
        if len(classes) != 1:
            raise ValueError("Exactly one class declaration is supported.")
        for statement in tree.body:
            if isinstance(statement, ast.ClassDef):
                if (
                    statement.name != class_name
                    or statement.bases
                    or statement.type_params
                ):
                    raise ValueError("Unexpected class declaration.")
                self._read_class(statement)
            else:
                self._read_import(statement)
        if class_name in self._imports:
            raise ValueError("Imports cannot shadow the required class.")
        self._validate_local_bindings()
        if (
            "__init__" in self._methods
            and self.call("__init__", ()) is not None
        ):
            raise ValueError("An initializer must return None.")

    def call(self, method_name: str, arguments: tuple[object, ...]) -> object:
        """Probe one instance method using plain bounded data values."""
        method = self._methods.get(method_name)
        if method is None:
            raise ValueError("Required method is missing.")
        parameters = method.args
        names = [arg.arg for arg in parameters.posonlyargs + parameters.args]
        if (
            parameters.vararg
            or parameters.kwarg
            or parameters.kwonlyargs
            or parameters.defaults
            or parameters.kw_defaults
        ):
            raise ValueError("Unsupported method signature.")
        values = (self._instance, *arguments)
        if len(names) != len(values):
            raise ValueError("Method signature does not match its contract.")
        scope = dict(zip(names, values, strict=True))
        return self._statements(method.body, scope)[1]

    def _read_class(self, declaration: ast.ClassDef) -> None:
        if declaration.decorator_list or declaration.keywords:
            raise ValueError("Class customization is unsupported.")
        for statement in declaration.body:
            if isinstance(statement, ast.FunctionDef):
                if statement.decorator_list or statement.name in self._methods:
                    raise ValueError("Method customization is unsupported.")
                if statement.name.startswith("__") and (
                    statement.name != "__init__"
                ):
                    raise ValueError(
                        "Object protocol overrides are unsupported."
                    )
                if statement.type_params:
                    raise ValueError("Generic declarations are unsupported.")
                if statement.args.defaults or statement.args.kw_defaults:
                    raise ValueError("Default arguments are unsupported.")
                self._methods[statement.name] = statement
            elif not _is_documentation(statement):
                raise ValueError("Only instance methods are supported.")

    def _read_import(self, statement: ast.stmt) -> None:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.name != "json":
                    raise ValueError("Only JSON serialization is supported.")
                self._register_import(alias.asname or alias.name, "json")
        elif isinstance(statement, ast.ImportFrom):
            if statement.module != "json" or statement.level:
                raise ValueError("Only JSON serialization is supported.")
            for alias in statement.names:
                if alias.name != "dumps":
                    raise ValueError("Only JSON serialization is supported.")
                self._register_import(alias.asname or alias.name, "json.dumps")
        elif not _is_documentation(statement):
            raise ValueError("Unsupported module statement.")

    def _register_import(self, name: str, target: str) -> None:
        if name in {"bool", "set"} or name in self._imports:
            raise ValueError("Import shadowing is unsupported.")
        self._imports[name] = target

    def _validate_local_bindings(self) -> None:
        reserved = {"bool", "set", *self._imports}
        supported = (
            ast.Return,
            ast.If,
            ast.Assign,
            ast.AnnAssign,
            ast.Expr,
            ast.Pass,
        )
        for method in self._methods.values():
            for node in ast.walk(method):
                if (
                    isinstance(node, ast.stmt)
                    and node is not method
                    and not isinstance(node, supported)
                ):
                    raise ValueError("Unsupported method statement.")
                if (
                    isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Store)
                    and node.id in reserved
                ) or (isinstance(node, ast.arg) and node.arg in reserved):
                    raise ValueError(
                        "Local shadowing of trusted names is unsupported."
                    )

    def _statements(
        self,
        statements: list[ast.stmt],
        scope: dict[str, object],
    ) -> tuple[bool, object]:
        for statement in statements:
            self._spend_step()
            if isinstance(statement, ast.Return):
                return True, self._expression(statement.value, scope)
            if isinstance(statement, ast.If):
                branch = (
                    statement.body
                    if self._expression(statement.test, scope)
                    else statement.orelse
                )
                returned, value = self._statements(branch, scope)
                if returned:
                    return True, value
            else:
                self._effect(statement, scope)
        return False, None

    def _effect(self, statement: ast.stmt, scope: dict[str, object]) -> None:
        if isinstance(statement, ast.Assign):
            value = self._expression(statement.value, scope)
            for target in statement.targets:
                self._assign(target, value, scope)
        elif isinstance(statement, ast.AnnAssign):
            if statement.value is not None:
                self._assign(
                    statement.target,
                    self._expression(statement.value, scope),
                    scope,
                )
            elif not isinstance(statement.target, ast.Name):
                raise ValueError("Only local annotations may omit a value.")
        elif isinstance(statement, ast.Expr):
            self._expression(statement.value, scope)
        elif not isinstance(statement, ast.Pass):
            raise ValueError("Unsupported method statement.")

    def _assign(
        self,
        target: ast.expr,
        value: object,
        scope: dict[str, object],
    ) -> None:
        if isinstance(target, ast.Name):
            scope[target.id] = value
        elif isinstance(target, ast.Attribute):
            owner = self._expression(target.value, scope)
            if (
                owner is not self._instance
                or target.attr.startswith("_")
                or target.attr in self._methods
            ):
                raise ValueError("Only public instance state is writable.")
            self.state[target.attr] = value
        else:
            raise ValueError("Unsupported assignment target.")

    def _expression(
        self,
        expression: ast.expr | None,
        scope: dict[str, object],
    ) -> object:
        self._spend_step()
        if expression is None:
            return None
        if isinstance(expression, ast.Constant):
            return cast("object", expression.value)
        if isinstance(expression, ast.Name):
            return scope[expression.id]
        if isinstance(expression, ast.Attribute):
            if self._expression(expression.value, scope) is not self._instance:
                raise ValueError("Only instance attributes are readable.")
            return self.state[expression.attr]
        if isinstance(expression, ast.Call):
            return self._call_expression(expression, scope)
        return self._compound_expression(expression, scope)

    def _compound_expression(
        self,
        expression: ast.expr,
        scope: dict[str, object],
    ) -> object:
        if isinstance(expression, (ast.List, ast.Set, ast.Tuple)):
            values = [
                self._expression(item, scope) for item in expression.elts
            ]
            if isinstance(expression, ast.Set):
                return set(values)
            return (
                tuple(values) if isinstance(expression, ast.Tuple) else values
            )
        if isinstance(expression, ast.Dict):
            if None in expression.keys:
                raise ValueError("Dictionary unpacking is unsupported.")
            return {
                self._expression(key, scope): self._expression(value, scope)
                for key, value in zip(
                    expression.keys,
                    expression.values,
                    strict=True,
                )
            }
        if isinstance(expression, ast.UnaryOp) and isinstance(
            expression.op,
            ast.Not,
        ):
            return not self._expression(expression.operand, scope)
        if isinstance(expression, ast.IfExp):
            branch = (
                expression.body
                if self._expression(expression.test, scope)
                else expression.orelse
            )
            return self._expression(branch, scope)
        raise ValueError("Unsupported expression.")

    def _call_expression(
        self,
        call: ast.Call,
        scope: dict[str, object],
    ) -> object:
        arguments = tuple(self._expression(arg, scope) for arg in call.args)
        keywords = {
            keyword.arg: self._expression(keyword.value, scope)
            for keyword in call.keywords
        }
        if isinstance(call.func, ast.Name):
            name = call.func.id
            if name in scope:
                raise ValueError("Calling candidate values is unsupported.")
            if name == "bool" and len(arguments) == 1 and not keywords:
                return bool(arguments[0])
            if name == "set" and not arguments and not keywords:
                return set[object]()
            name = self._imports.get(name, name)
        elif isinstance(call.func, ast.Attribute):
            name = self._attribute_call(call.func, arguments, keywords, scope)
            if name != "json.dumps":
                return None
        else:
            raise ValueError("Unsupported call target.")
        if name != "json.dumps" or len(arguments) != 1:
            raise ValueError("Only allow-listed calls are supported.")
        return _serialize_probe_json(arguments[0], keywords)

    def _attribute_call(
        self,
        function: ast.Attribute,
        arguments: tuple[object, ...],
        keywords: dict[str | None, object],
        scope: dict[str, object],
    ) -> str:
        owner = function.value
        if (
            isinstance(owner, ast.Name)
            and owner.id not in scope
            and self._imports.get(owner.id) == "json"
            and function.attr == "dumps"
        ):
            return "json.dumps"
        container = self._expression(owner, scope)
        if len(arguments) != 1 or keywords:
            raise ValueError("Unsupported container call.")
        if isinstance(container, set) and function.attr == "add":
            cast("set[object]", container).add(arguments[0])
        elif isinstance(container, list) and function.attr == "append":
            cast("list[object]", container).append(arguments[0])
        else:
            raise ValueError("Unsupported container operation.")
        return "inserted"

    def _spend_step(self) -> None:
        self._steps -= 1
        if self._steps < 0:
            raise ValueError("Behavior probe exceeded its step budget.")


def _is_documentation(statement: ast.stmt) -> bool:
    return isinstance(statement, ast.Pass) or (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _serialize_probe_json(
    value: object,
    keywords: dict[str | None, object],
) -> str:
    if not keywords.keys() <= {"sort_keys", "ensure_ascii", "indent"}:
        raise ValueError("Unsupported serialization option.")
    sort_keys = keywords.get("sort_keys", False)
    ensure_ascii = keywords.get("ensure_ascii", True)
    indent = keywords.get("indent")
    if not isinstance(sort_keys, bool) or not isinstance(ensure_ascii, bool):
        raise ValueError("Serialization flags must be boolean.")
    if indent is not None and (
        not isinstance(indent, int) or not 0 <= indent <= MAX_PROBE_JSON_INDENT
    ):
        raise ValueError("Serialization indentation must be bounded.")
    return json.dumps(
        value,
        sort_keys=sort_keys,
        ensure_ascii=ensure_ascii,
        indent=indent,
    )
