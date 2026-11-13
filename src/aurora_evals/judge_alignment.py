"""Judge alignment: can we trust the LLM judge? Measure it against human labels.

An LLM-as-judge metric is itself a model making predictions, so it needs its own evaluation.
Collect a small set of human-labelled examples, run the judge on them, and report:

* accuracy      fraction of items where judge == human
* Cohen's kappa agreement corrected for chance (1.0 perfect, 0 = no better than chance)
* confusion     which way the judge errs. For compliance, a judge that *passes* bad answers
                (false pass) is far more dangerous than one that is too strict.

Rule of thumb used by many teams: don't put a judge in a release gate until kappa ≥ 0.6
on at least ~50 to 100 labelled items. This project ships 10 so it runs in a classroom.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa for two raters over the same items (no sklearn needed)."""
    if len(a) != len(b) or not a:
        raise ValueError("Need two equal-length, non-empty label lists")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[label] * cb[label] for label in set(a) | set(b)) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def alignment_report(human: list[str], judge: list[str], positive: str = "pass") -> dict:
    pairs = list(zip(human, judge))
    return {
        "n": len(pairs),
        "accuracy": round(sum(h == j for h, j in pairs) / len(pairs), 3),
        "cohen_kappa": round(cohen_kappa(human, judge), 3),
        # judge said pass, human said fail: the dangerous direction for a safety gate
        "false_pass": sum(h != positive and j == positive for h, j in pairs),
        # judge said fail, human said pass: annoying (blocks good releases) but safe
        "false_fail": sum(h == positive and j != positive for h, j in pairs),
    }


def confusion_table(human: list[str], judge: list[str]) -> pd.DataFrame:
    return pd.crosstab(pd.Series(human, name="human"), pd.Series(judge, name="judge"))
