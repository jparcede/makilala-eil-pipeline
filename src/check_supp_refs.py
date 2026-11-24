#!/usr/bin/env python3
"""Verify the hard-coded main-text cross-references in supplementary.tex.

supplementary.tex compiles as its own document, so it cannot \\ref{} labels that
live in the manuscript -- those render as "??". The five such references are
therefore written as literal numbers, which means they go stale silently
whenever a section is added or removed upstream. (Removing one Results
subsection on 15 Sep 2026 moved Section 5.6 to 5.5 and would have left the
Supplementary pointing at the wrong section.)

This script reads the numbers LaTeX actually assigned, from
Makilala_main_final.aux, and checks each literal against them.

    python3 check_supp_refs.py          # after building the manuscript

Exit status is 0 if every literal is correct, 1 otherwise. Add a row to
EXPECTED whenever a new cross-reference is written into supplementary.tex.
"""
import re
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
AUX = os.path.join(HERE, "Makilala_main_final.aux")
SUPP = os.path.join(HERE, "supplementary.tex")

# (label in the manuscript, how it is written in supplementary.tex)
# The literal is a regex; \N is the number position.
EXPECTED = [
    ("sec:results",            r"Section~(\S+) aggregate the FoS"),
    ("sec:analytical",         r"analytical properties of Section~(\S+) and"),
    ("app:proofs",             r"Appendix~(\S+) into directly readable form"),
    ("sec:cprime_sensitivity", r"Section~(\S+) establishes, and drawing"),
    ("sec:z_sensitivity",      r"Section~(\S+)\)\. The panel shows the mechanism"),
    ("sec:synthesis",          r"observations combined in Section~(\S+) of the main text"),
    ("sec:blocked_validation", r"Section~(\S+) of the manuscript reports the blocked"),
    # Added 24 Nov 2026 when the full CRS diagnosis moved out of the manuscript.
    ("sec:crs",                r"Section~(\S+) of the main text summarises the georeferencing"),
]

# References running the OTHER way: a number owned by supplementary.tex that the
# MANUSCRIPT writes as a literal. Added 22 Oct 2026 after the per-tile table was
# cited as "Table S1" -- which already belonged to the reference parameter set, so
# the manuscript pointed a reader at the wrong table. Same failure mode as above,
# opposite direction, and nothing was checking it.
REVERSE = [
    # Capture the "S" too: supplementary.aux stores the number as "S3", not "3".
    ("tab:S_blocked", r"tabulated in the supplementary material \(Table~(S\S+?)\)"),
    # The CRS diagnosis moved to the supplementary on 24 Nov 2026 and became its
    # FIRST section, shifting the other two down. The manuscript points at it by
    # literal number, so this is the entry that catches a future reordering.
    ("sec:S_crs",   r"supplementary material \(Section~(S\S+?)\)"),
]

SUPP_AUX = os.path.join(HERE, "supplementary.aux")
MAIN = os.path.join(HERE, "Makilala_main_final.tex")

if not os.path.exists(AUX):
    sys.exit(f"no {os.path.basename(AUX)} -- build the manuscript first")

aux = open(AUX, encoding="utf-8", errors="replace").read()
supp = open(SUPP, encoding="utf-8").read()
supp_aux = (open(SUPP_AUX, encoding="utf-8", errors="replace").read()
            if os.path.exists(SUPP_AUX) else "")
main = open(MAIN, encoding="utf-8").read()


def _number(blob, label):
    m = re.search(r"\\newlabel\{" + re.escape(label) + r"\}\{\{([^}]*)\}", blob)
    return m.group(1) if m else None


def number_of(label):
    return _number(aux, label)


bad = 0
print(f"{'label':<26}{'manuscript':>12}{'supplementary':>16}")
print("-" * 54)
for label, pattern in EXPECTED:
    truth = number_of(label)
    m = re.search(pattern, supp)
    written = m.group(1).rstrip(".,;)") if m else None

    if truth is None:
        status, bad = "LABEL GONE", bad + 1
    elif written is None:
        status, bad = "NOT FOUND", bad + 1
    elif written != truth:
        status, bad = "STALE", bad + 1
    else:
        status = "ok"

    print(f"{label:<26}{str(truth):>12}{str(written):>16}   {status}")

# --- the other direction: supplementary numbers written literally in the paper --
if REVERSE:
    print()
    print(f"{'label (in supp.)':<26}{'supplementary':>12}{'manuscript':>16}")
    print("-" * 54)
    for label, pattern in REVERSE:
        truth = _number(supp_aux, label)
        m = re.search(pattern, main)
        written = m.group(1).rstrip(".,;)") if m else None
        if not supp_aux:
            status = "no supplementary.aux -- build it"
        elif truth is None:
            status, bad = "LABEL GONE", bad + 1
        elif written is None:
            status, bad = "NOT FOUND", bad + 1
        elif written != truth:
            status, bad = "STALE", bad + 1
        else:
            status = "ok"
        print(f"{label:<26}{str(truth):>12}{str(written):>16}   {status}")

print("-" * 54)
if bad:
    print(f"{bad} reference(s) need attention")
    print("A STALE entry means the other document renumbered; edit the literal.")
    print("NOT FOUND means the sentence was reworded; update EXPECTED/REVERSE here.")
else:
    print(f"all {len(EXPECTED) + len(REVERSE)} literals agree across the two documents")
sys.exit(1 if bad else 0)
