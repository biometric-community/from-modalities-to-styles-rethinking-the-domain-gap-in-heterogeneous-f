"""REPORT.md, the committed output of the full run, against the published results."""

from __future__ import annotations

import re

import pytest


def metric(report: str, name: str) -> float:
    match = re.search(rf"\| {re.escape(name)} \| ([\d.]+)% \|", report)
    assert match, f"{name} row missing from REPORT.md"
    return float(match.group(1))


def test_cross_resolution_rank1_matches_scface_far(report, targets):
    expected = targets["scface_far"]["caim_rank1"]
    observed = metric(report, "CelebA LR→HR Rank-1")
    assert observed == pytest.approx(expected, abs=targets["tolerance_pp"])


def test_cross_resolution_rank1_reaches_the_unadapted_baseline(report, targets):
    """Table 4: even the pretrained model without CAIM reaches 74.42% at far range."""
    assert metric(report, "CelebA LR→HR Rank-1") >= targets["scface_far"]["baseline_rank1"]
