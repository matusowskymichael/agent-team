"""Parse a closed TSX component subset without executing JavaScript.

The supported grammar is an exported function with one props parameter and
one return of intrinsic JSX elements. Props may be destructured or qualified,
with inline message/callback types. JSX expressions may reference supplied
text or callback props, or wrap a callback in a zero-argument arrow. Unknown
syntax, extra statements, custom components and hidden elements fail closed.
"""

import re
from collections.abc import Mapping
from html import escape
from xml.etree import ElementTree

CALLBACK_MARKER = "evaluation-callback-reference"
INTRINSIC_ELEMENTS = frozenset({"div", "section", "span", "p", "button"})


def probe_component(
    source: str,
    component_name: str,
    props: Mapping[str, str],
) -> ElementTree.Element:
    """Render the supported component grammar using only supplied props."""
    prefix = re.match(
        rf"\s*export\s+(?:default\s+)?function\s+"
        rf"{re.escape(component_name)}\s*\(",
        source,
    )
    if prefix is None:
        raise ValueError("Required exported component is missing.")
    parameter_end = _balanced_end(source, prefix.end() - 1, "(", ")")
    bindings = _prop_bindings(source[prefix.end() : parameter_end], props)
    body = source[parameter_end + 1 :].strip().removesuffix(";").strip()
    if not body.startswith("{") or _balanced_end(body, 0, "{", "}") != (
        len(body) - 1
    ):
        raise ValueError("Unsupported component declaration.")
    returned = re.fullmatch(
        r"\s*return[ \t]+([^\r\n][\s\S]*?)\s*;?\s*",
        body[1:-1],
    )
    if returned is None:
        raise ValueError("Component must return its rendered element.")
    jsx = returned.group(1).strip().removesuffix(";").strip()
    if jsx.startswith("(") and _balanced_end(jsx, 0, "(", ")") == len(jsx) - 1:
        jsx = jsx[1:-1].strip()
    rendered = _render_expressions(jsx, bindings)
    # Explicitly reject boolean attributes instead of treating them as text.
    element = ElementTree.fromstring(rendered)  # noqa: S314
    for child in element.iter():
        if child.tag not in INTRINSIC_ELEMENTS:
            raise ValueError("Only intrinsic visible elements are supported.")
        if not child.attrib.keys() <= {"onClick", "type", "className", "role"}:
            raise ValueError("Unsupported element attribute.")
        if "className" in child.attrib:
            raise ValueError("External CSS visibility cannot be established.")
    return element


def _prop_bindings(
    parameters: str,
    props: Mapping[str, str],
) -> dict[str, str]:
    parameter = parameters.strip()
    if parameter.startswith("{"):
        end = _balanced_end(parameter, 0, "{", "}")
        names = tuple(name.strip() for name in parameter[1:end].split(","))
        if len(set(names)) != len(names) or any(
            name not in props for name in names
        ):
            raise ValueError("Unsupported destructured props.")
        _validate_annotation(parameter[end + 1 :], props)
        return {name: props[name] for name in names}
    match = re.match(r"([A-Za-z_$][\w$]*)", parameter)
    if match is None:
        raise ValueError("Unsupported props parameter.")
    _validate_annotation(parameter[match.end() :], props)
    return {f"{match.group(1)}.{name}": value for name, value in props.items()}


def _validate_annotation(annotation: str, props: Mapping[str, str]) -> None:
    if not annotation.strip():
        return
    fields: list[str] = []
    for name, value in props.items():
        field_type = (
            r"\(\s*\)\s*=>\s*void" if value == CALLBACK_MARKER else ("string")
        )
        fields.append(rf"{re.escape(name)}\s*:\s*{field_type}\s*;?")
    pattern = r"\s*:\s*\{\s*" + r"\s*".join(fields) + r"\s*\}\s*"
    if re.fullmatch(pattern, annotation) is None:
        raise ValueError("Unsupported props type annotation.")


def _render_expressions(jsx: str, bindings: Mapping[str, str]) -> str:
    if "<!" in jsx or "<?" in jsx:
        raise ValueError("XML declarations are not JSX syntax.")
    if re.search(r"\bonClick\s*=\s*([^\s{])", jsx):
        raise ValueError("Event handlers must come from prop expressions.")
    parts: list[str] = []
    position = 0
    while position < len(jsx):
        if jsx[position] != "{":
            parts.append(jsx[position])
            position += 1
            continue
        end = _balanced_end(jsx, position, "{", "}")
        expression = jsx[position + 1 : end].strip()
        value = bindings.get(expression)
        if value is None:
            value = _arrow_callback(expression, bindings)
        if value is None:
            raise ValueError("Unsupported JSX expression.")
        escaped = escape(value, quote=True)
        is_attribute = "".join(parts).rstrip().endswith("=")
        if value == CALLBACK_MARKER and not is_attribute:
            raise ValueError("A callback cannot render text.")
        parts.append(f'"{escaped}"' if is_attribute else escaped)
        position = end + 1
    return "".join(parts)


def _arrow_callback(
    expression: str, bindings: Mapping[str, str]
) -> str | None:
    match = re.fullmatch(r"\(\s*\)\s*=>\s*([\s\S]+)", expression)
    if match is None:
        return None
    body = match.group(1).strip()
    if body.startswith("{") and body.endswith("}"):
        body = body[1:-1].strip().removesuffix(";").strip()
    invocation = re.fullmatch(r"([\w$.]+)\s*\(\s*\)", body)
    if invocation is None:
        return None
    reference = bindings.get(invocation.group(1))
    return reference if reference == CALLBACK_MARKER else None


def _balanced_end(source: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    quote = ""
    escaped = False
    for position in range(start, len(source)):
        character = source[position]
        if quote:
            if character == quote and not escaped:
                quote = ""
            escaped = character == "\\" and not escaped
            continue
        if character in {"'", '"'}:
            quote = character
        elif character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return position
    raise ValueError("Unbalanced component syntax.")
