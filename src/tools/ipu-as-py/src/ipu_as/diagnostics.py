#!/usr/bin/env python3
"""Report assembly errors as positioned diagnostics, for editor squiggles.

A TextMate grammar assigns scopes to spans; it has no way to say "this is
wrong", and no notion of what the grammar expects next. Validity is not a
lexical property — whether a comma is legal depends on the parser's state, not
on the characters. So the only thing that can answer "is this valid" is the
parser itself, which is what this module runs.

Three stages are checked, in the order the assembler runs them, stopping at the
first failure (a template that will not render cannot be parsed, and a program
that will not parse cannot be encoded):

1. Jinja rendering   -> TemplateSyntaxError
2. Parsing           -> lark.exceptions.LarkError
3. Encoding          -> ValueError from CompoundInst

Positions are 0-based, matching LSP and the VS Code API.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import jinja2
import lark

import ipu_as.compound_inst as compound_inst
import ipu_as.label as ipu_label
from ipu_as.lark_tree import ASTBuilder, get_parser

#: Assembly-stage errors embed their position in the message text (see
#: AnnotatedToken.get_location_string). Line/column are 1-based there.
_LOCATION_RE = re.compile(r"Line (\d+), Column (\d+)")

#: Some encode errors name the offending token in quotes but carry no position
#: (e.g. "Opcode 'mac.ee' not found"); this recovers the name so it can be
#: located in the source.
_QUOTED_NAME_RE = re.compile(r"'([^']+)'")

#: Jinja is a preprocessing layer, so a template marker means the parser sees
#: different text than the editor shows. See _map_line.
_JINJA_MARKERS = ("{{", "{%", "{#")


@dataclass
class Diagnostic:
    line: int
    column: int
    end_line: int
    end_column: int
    message: str
    severity: str
    stage: str
    #: True when the position could not be pinned down exactly — either the
    #: error carried none, or Jinja rendering moved it. Callers should say so
    #: rather than imply byte accuracy.
    approximate: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _offset_to_linecol(text: str, offset: int) -> tuple[int, int]:
    """0-based (line, column) for a byte offset."""
    prefix = text[:offset]
    line = prefix.count("\n")
    column = offset - (prefix.rfind("\n") + 1)
    return line, column


def _map_line(raw: str, rendered: str, rendered_line: int) -> tuple[int, bool]:
    """Translate a line index in the rendered text back to the source file.

    When the file has no Jinja the two are identical and this is exact. When it
    does, rendering can insert, drop and rewrite lines, so there is no general
    mapping — but a line the template did not touch survives verbatim, which
    covers most real errors. Match it by content, and only when the match is
    unambiguous.

    Returns (line, exact).
    """
    if raw == rendered:
        return rendered_line, True

    rendered_lines = rendered.splitlines()
    if rendered_line >= len(rendered_lines):
        return 0, False

    target = rendered_lines[rendered_line]
    if not target.strip():
        return 0, False

    raw_lines = raw.splitlines()
    for candidate in (target, target.strip()):
        hits = [
            i
            for i, line in enumerate(raw_lines)
            if (line if candidate is target else line.strip()) == candidate
        ]
        if len(hits) == 1:
            return hits[0], True

    return 0, False


def _humanize(terminal_names) -> str:
    """Render a set of expected terminal names as something a human reads.

    The parser reports internal names like `_SEMI2`. Literal terminals carry
    their text, so show that; the rest fall back to a lowercased name.
    """
    by_name = {t.name: t for t in get_parser().terminals}
    shown = []
    for name in sorted(terminal_names):
        terminal = by_name.get(name)
        value = getattr(getattr(terminal, "pattern", None), "value", None)
        if terminal is not None and type(terminal.pattern).__name__ == "PatternStr":
            shown.append(f"'{value}'")
        else:
            shown.append(name.lstrip("_").lower())
    return ", ".join(shown)


def _template_diagnostic(error: jinja2.TemplateSyntaxError) -> Diagnostic:
    # Jinja reports against the source template, so this is already exact.
    line = max((error.lineno or 1) - 1, 0)
    return Diagnostic(
        line=line,
        column=0,
        end_line=line,
        end_column=0,
        message=f"Template error: {error.message}",
        severity="error",
        stage="template",
    )


def _parse_diagnostic(error: lark.exceptions.LarkError, raw: str, rendered: str):
    token = getattr(error, "token", None)
    if token is not None and getattr(token, "start_pos", None) is not None:
        start, end = token.start_pos, token.end_pos
        expected = getattr(error, "expected", set())
        found = f"unexpected {str(token)!r}"
    else:
        start = getattr(error, "pos_in_stream", 0) or 0
        end = start + 1
        expected = getattr(error, "allowed", set()) or set()
        found = f"unexpected {rendered[start:end]!r}"

    line, column = _offset_to_linecol(rendered, start)
    end_line, end_column = _offset_to_linecol(rendered, end)
    mapped, exact = _map_line(raw, rendered, line)

    message = found
    if expected:
        message += f"; expected one of: {_humanize(expected)}"

    return Diagnostic(
        line=mapped,
        column=column if exact else 0,
        end_line=mapped + (end_line - line) if exact else mapped,
        end_column=end_column if exact else 0,
        message=message,
        severity="error",
        stage="parse",
        approximate=not exact,
    )


def _tidy(message: str) -> str:
    """Flatten a multi-line assembler message and drop its trailing location.

    The assembler appends "At: Line 1, Column 1" for humans reading a terminal.
    A diagnostic already carries the position, and repeating it in the hover
    text is noise.
    """
    without_location = re.sub(r"\s*(?:At|In):?\s*Line \d+, Column \d+", "", message)
    return " ".join(without_location.split())


def _encode_diagnostic(error: ValueError, raw: str, rendered: str) -> Diagnostic:
    message = str(error)
    match = _LOCATION_RE.search(message)
    if match:
        line = max(int(match.group(1)) - 1, 0)
        column = max(int(match.group(2)) - 1, 0)
        mapped, exact = _map_line(raw, rendered, line)
    else:
        # Not every encode error carries a location — "Opcode 'mac.ee' not
        # found" does not. Fall back to locating the offending name, which the
        # message quotes, so the squiggle still lands on the right token.
        mapped, column, exact = 0, 0, False
        quoted = _QUOTED_NAME_RE.search(message)
        if quoted:
            offset = rendered.find(quoted.group(1))
            if offset >= 0:
                line, column = _offset_to_linecol(rendered, offset)
                mapped, exact = _map_line(raw, rendered, line)
                if not exact:
                    column = 0

    return Diagnostic(
        line=mapped,
        column=column if exact else 0,
        end_line=mapped,
        end_column=(column + 1) if exact else 0,
        message=_tidy(message),
        severity="error",
        stage="assemble",
        approximate=not exact,
    )


def check(text: str) -> list[Diagnostic]:
    """Return diagnostics for one assembly source. Empty means it assembles."""
    rendered = text
    if any(marker in text for marker in _JINJA_MARKERS):
        try:
            rendered = jinja2.Template(text).render()
        except jinja2.TemplateSyntaxError as error:
            return [_template_diagnostic(error)]
        except jinja2.TemplateError as error:  # undefined variable, etc.
            return [
                Diagnostic(0, 0, 0, 0, f"Template error: {error}", "error", "template")
            ]

    # The label registry is module-global; a stale entry from a previous check
    # would surface as a bogus "label defined twice".
    ipu_label.reset_labels()

    try:
        tree = get_parser().parse(rendered)
    except lark.exceptions.LarkError as error:
        return [_parse_diagnostic(error, text, rendered)]

    try:
        ast = ASTBuilder().transform(tree)
    except lark.exceptions.VisitError as error:
        # lark wraps anything a transformer raises in VisitError, which is
        # itself a LarkError — so without unwrapping, an encode error carrying
        # an exact location gets misreported as a parse error at offset 0.
        original = error.orig_exc
        if isinstance(original, ValueError):
            return [_encode_diagnostic(original, text, rendered)]
        raise

    try:
        for instruction in ast:
            compound_inst.CompoundInst(instruction).encode()
    except ValueError as error:
        return [_encode_diagnostic(error, text, rendered)]

    return []
