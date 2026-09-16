"""Closed local TypeScript declaration grammar without code execution."""

import pytest

from agent_team.infrastructure.evaluation.bounded_tsx_probe import (
    CALLBACK_MARKER,
    probe_component,
)

CALLBACK_COMPONENT = (
    "export function AccountMenu({ onLogout }: Props) {\n"
    "  return <button onClick={onLogout}>Logout</button>;\n}\n"
)


class TestBoundedTsxProbe:
    """Resolve exact local shapes while rejecting executable type syntax."""

    @pytest.mark.parametrize(
        "declaration",
        (
            "import type { Props } from './props';\n",
            "interface Props extends External { onLogout: () => void; }\n",
            "interface Props<T> { onLogout: () => void; }\n",
            "type Props<T> = { onLogout: () => void; };\n",
            "type Props = { onLogout: () => void; } | External;\n",
            "type Props = { onLogout: () => void; } & External;\n",
            "type Props = External;\n",
            "type Props = import('./props').Props;\n",
            "type Props = typeof external;\n",
            "type Props { onLogout: () => void; }\n",
            "interface Props = { onLogout: () => void; }\n",
            "interface Props { onLogout: () => void; }\n"
            "interface Props { onLogout: () => void; }\n",
            "interface OtherProps { onLogout: () => void; }\n",
            "interface Props { onLogout: string; }\n",
            "interface Props { onLogout: (event: Event) => void; }\n",
            "interface Props { onLogout: () => string; }\n",
            "interface Props { onLogout?: () => void; }\n",
            "interface Props { onLogout: (() => void) | undefined; }\n",
            "interface Props { onLogout: Callback; }\n",
            "interface Props { onLogout(): void; }\n",
            "interface Props { onLogout: () => void; extra: string; }\n",
            "interface Props { onLogout: () => void; "
            "onLogout: () => void; }\n",
            "interface Props { [name: string]: () => void; }\n",
            "interface Props { onLogout: () => void = callback; }\n",
            "interface Props { onLogout: () => void; }\n"
            "globalThis.callback();\n",
            "globalThis.callback();\n"
            "interface Props { onLogout: () => void; }\n",
            "interface Props { onLogout: () => void; }\n"
            "import external from './external';\n",
            "interface Props { onLogout: () => void;\n",
        ),
        ids=(
            "imported-type",
            "inheritance",
            "generic-interface",
            "generic-alias",
            "union",
            "intersection",
            "external-alias",
            "import-type-query",
            "typeof-query",
            "missing-alias-assignment",
            "interface-assignment",
            "declaration-merging",
            "unresolved-exact-name",
            "wrong-property-type",
            "callback-argument",
            "callback-result",
            "optional-callback",
            "property-union",
            "external-property-type",
            "method-signature",
            "extra-property",
            "duplicate-property",
            "index-signature",
            "property-initializer",
            "execution-after-declaration",
            "execution-before-declaration",
            "import-after-declaration",
            "unbalanced-declaration",
        ),
    )
    def test_unsupported_local_declarations_fail_closed(
        self, declaration: str
    ) -> None:
        """Never resolve external types or approximate unsupported syntax."""
        with pytest.raises(ValueError):
            probe_component(
                declaration + CALLBACK_COMPONENT,
                "AccountMenu",
                {"onLogout": CALLBACK_MARKER},
            )

    @pytest.mark.parametrize(
        "annotation",
        ("OtherProps", "Props & Extra", "Props | undefined", "Props<string>"),
        ids=("different-name", "intersection", "union", "generic-reference"),
    )
    def test_named_reference_requires_exact_local_declaration(
        self, annotation: str
    ) -> None:
        """A valid declaration cannot authorize an unrelated annotation."""
        source = (
            "interface Props { onLogout: () => void; }\n"
            + CALLBACK_COMPONENT.replace(": Props)", f": {annotation})")
        )
        with pytest.raises(ValueError):
            probe_component(
                source, "AccountMenu", {"onLogout": CALLBACK_MARKER}
            )

    @pytest.mark.parametrize(
        "suffix",
        (
            "globalThis.callback();",
            "export function Other() { return <p>Other</p>; }",
            "const extra = globalThis.callback();",
        ),
        ids=("trailing-call", "extra-component", "trailing-initializer"),
    )
    def test_named_props_do_not_allow_extra_top_level_behavior(
        self, suffix: str
    ) -> None:
        """Local prop declarations preserve the sole component boundary."""
        source = (
            "interface Props { onLogout: () => void; }\n"
            + CALLBACK_COMPONENT
            + suffix
        )
        with pytest.raises(ValueError):
            probe_component(
                source, "AccountMenu", {"onLogout": CALLBACK_MARKER}
            )

    def test_exact_local_name_can_follow_another_valid_declaration(
        self,
    ) -> None:
        """Resolve a named shape without treating order as type identity."""
        source = (
            "interface OtherProps { onLogout: () => void; }\n"
            "export type Props = { onLogout: () => void };\n"
            + CALLBACK_COMPONENT
        )
        result = probe_component(
            source, "AccountMenu", {"onLogout": CALLBACK_MARKER}
        )
        assert result.tag == "button"
        assert result.get("onClick") == CALLBACK_MARKER
        assert result.text == "Logout"
