"""Diagnostics must land on the right token, not merely report that something is wrong.

An editor squiggle is only useful if it is in the right place, so these assert
positions, not just that an error was produced.
"""

import pytest

from ipu_as.diagnostics import check


def test_valid_program_has_no_diagnostics():
    assert check("start:\n  BKPT;;\n") == []


def test_comma_is_reported_at_the_comma():
    # The motivating case: commas are used throughout the prose docs but are a
    # lex error, and a TextMate grammar cannot flag them.
    (found,) = check("BEQ lr0, cr0, +1;;\n")
    assert found.stage == "parse"
    assert (found.line, found.column) == (0, 7)
    assert not found.approximate


def test_unknown_mnemonic_points_at_the_mnemonic():
    # This error carries no location of its own, so the position is recovered
    # by locating the name the message quotes.
    (found,) = check("BKPT;;\nmac.ee r0;;\n")
    assert found.stage == "assemble"
    assert found.line == 1
    assert "mac.ee" in found.message


def test_pseudo_instruction_arity_is_reported():
    # Raised inside the AST transformer, which lark wraps in VisitError -- a
    # LarkError. Without unwrapping it is misreported as a parse error at 0.
    (found,) = check("BGT lr0 lr1;;\n")
    assert found.stage == "assemble"
    assert found.line == 0
    assert "BGT" in found.message


def test_location_suffix_is_stripped_from_the_message():
    # The position is carried by the diagnostic; repeating it in the hover is noise.
    (found,) = check("BGT lr0 lr1;;\n")
    assert "Line 1, Column 1" not in found.message


def test_bad_operand_is_reported_at_the_operand():
    (found,) = check("ADD lr0 lr0 1.5;;\n")
    assert found.stage == "assemble"
    assert found.line == 0
    assert found.column > 0


def test_template_syntax_error_reports_the_template_line():
    (found,) = check("BKPT;;\n{% for x in %}\n")
    assert found.stage == "template"
    assert found.line == 1


def test_error_inside_a_jinja_file_maps_back_to_the_source_line():
    # The parser sees rendered text, the editor sees the template, so offsets
    # do not correspond. An untouched line still survives verbatim and is
    # located by content.
    source = "{% for i in range(2) %}\nBKPT;;\n{% endfor %}\nmac.ee r0;;\n"
    (found,) = check(source)
    assert source.splitlines()[found.line] == "mac.ee r0;;"


@pytest.mark.parametrize(
    "source",
    [
        "start:\n  BKPT;;\n",
        "BKPT;; mid: BKPT;;\n",
        "add:\n  BKPT;;\n",
        "  acc.add.first ;;\n",
    ],
)
def test_legal_constructs_are_not_flagged(source):
    assert check(source) == []
