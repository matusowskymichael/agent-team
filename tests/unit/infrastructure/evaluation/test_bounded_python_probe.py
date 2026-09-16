"""Public behavior and denial tests for the closed Python probe grammar."""

import pytest

from agent_team.infrastructure.evaluation.bounded_python_probe import (
    MAX_PROBE_STEPS,
    BoundedPythonProbe,
)


class TestBoundedPythonProbe:
    """Preserve supported Python semantics and reject capability expansion."""

    @pytest.mark.parametrize(
        "source",
        (
            "class Different: pass",
            "class Probe(object): pass",
            "@decorator\nclass Probe: pass",
            "class Probe:\n    @decorator\n    def run(self): pass",
            "class Probe:\n    def run[T](self): pass",
            "class Probe:\n    value = 1",
            "import os\nclass Probe: pass",
            "from json import load\nclass Probe: pass",
            "print('never executed')\nclass Probe: pass",
            "class Probe:\n    def run(self):\n"
            "        return bool(True)\n        import json as bool",
            "class Probe:\n    def run(self):\n"
            "        return bool(True)\n        def bool(): pass",
            "import json as serializer\nclass Probe:\n"
            "    def run(self):\n"
            "        return serializer.dumps([])\n        serializer = None",
        ),
        ids=(
            "wrong-class",
            "inheritance",
            "class-decorator",
            "method-decorator",
            "generic-method",
            "class-attribute",
            "external-import",
            "nonallowlisted-json",
            "module-side-effect",
            "unreachable-local-import",
            "unreachable-local-function",
            "unreachable-import-shadow",
        ),
    )
    def test_declaration_capabilities_are_denied(self, source: str) -> None:
        """Never approximate declarations that may execute candidate code."""
        with pytest.raises(ValueError):
            BoundedPythonProbe(source, "Probe")

    @pytest.mark.parametrize(
        "body",
        (
            "value, other = (1, 2)",
            "return self.__class__.__name__",
            "return {**{}}",
            "return 1 + 2",
            "other = bool\n        return other(True)",
            "return (lambda: True)()",
            "return print('never executed')",
            "value = []\n        value.append()",
            "return json.dumps({}, default=True)",
            "return json.dumps({}, sort_keys='yes')",
            "return json.dumps({}, indent='massive')",
        ),
        ids=(
            "destructuring",
            "object-introspection",
            "dict-unpacking",
            "unsupported-operator",
            "candidate-callable",
            "lambda-call",
            "builtin-io",
            "container-arity",
            "serialization-callable",
            "serialization-flags",
            "serialization-budget",
        ),
    )
    def test_expression_capabilities_are_denied(self, body: str) -> None:
        """Fail closed unsupported calls, targets and serialization options."""
        source = (
            "import json\nclass Probe:\n    def run(self):\n        " + body
        )
        probe = BoundedPythonProbe(source, "Probe")
        with pytest.raises((ValueError, KeyError)):
            probe.call("run", ())

    @pytest.mark.parametrize(
        ("signature", "arguments"),
        (("self, *values", ()), ("self, value", ())),
    )
    def test_method_contract_requires_exact_arguments(
        self,
        signature: str,
        arguments: tuple[object, ...],
    ) -> None:
        """Reject variadic APIs and missing required parameters."""
        probe = BoundedPythonProbe(
            f"class Probe:\n    def run({signature}):\n        pass",
            "Probe",
        )
        with pytest.raises(ValueError):
            probe.call("run", arguments)

    @pytest.mark.parametrize(
        ("expression", "expected"),
        (
            ("{'value': True}", {"value": True}),
            ("{1, 2}", {1, 2}),
            ("(1, 2)", (1, 2)),
            ("False if True else True", False),
            ("False if False else True", True),
            ("bool(self)", True),
            ("not self", False),
            ("True if self else False", True),
        ),
    )
    def test_supported_values_keep_python_semantics(
        self,
        expression: str,
        expected: object,
    ) -> None:
        """Preserve containers and conditional expression behavior."""
        probe = BoundedPythonProbe(
            '"Module documentation."\nclass Probe:\n'
            f"    def run(self):\n        return {expression}",
            "Probe",
        )
        assert probe.call("run", ()) == expected

    def test_bare_return_and_pass_return_none(self) -> None:
        """Implement empty statements and bare returns as Python does."""
        probe = BoundedPythonProbe(
            "class Probe:\n    def run(self):\n        pass\n        return",
            "Probe",
        )
        assert probe.call("run", ()) is None

    def test_probe_budget_is_enforced(self) -> None:
        """Repeated public invocations cannot exceed the finite step budget."""
        probe = BoundedPythonProbe(
            "class Probe:\n    def run(self):\n        return True",
            "Probe",
        )
        with pytest.raises(ValueError, match="step budget"):
            for _ in range(MAX_PROBE_STEPS):
                probe.call("run", ())

    def test_local_annotation_keeps_existing_value(self) -> None:
        """An annotation without a value must not overwrite local state."""
        probe = BoundedPythonProbe(
            "class Probe:\n    def run(self):\n"
            "        value = True\n        value: bool\n        return value",
            "Probe",
        )
        assert probe.call("run", ()) is True
