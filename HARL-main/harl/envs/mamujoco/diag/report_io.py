"""Shared debug-file / report / statistics helpers for the PCR diagnosis campaign.

Every campaign stage writes **two** artifacts:

  * machine-readable CSV/NPZ  — consumed by ``scripts/diag_report.py``;
  * a human-readable markdown debug file — the thing you actually read while
    ideating. ``DebugReport`` writes it and echoes to stdout at the same time, so
    a run's console log and its debug file never disagree.

Also home to the campaign's **CI rule** (spec §3.3): every reported comparison
carries a bootstrap 95% CI over episodes; a claim "X > Y" requires
non-overlapping CIs **or** a gap of >= 3x the pooled std. ``compare()``
implements exactly that and returns the verdict as a string, so no script has to
re-invent (or quietly weaken) the rule.

Pure stdlib + numpy. No torch, no simulator.
"""

import os
import sys
import time

import numpy as np

__all__ = ["DebugReport", "bootstrap_ci", "compare", "fmt_ci", "write_csv"]


# ==========================================================================
#  statistics
# ==========================================================================
def bootstrap_ci(x, n_boot=10000, alpha=0.05, seed=0, stat=np.mean):
    """Bootstrap CI of ``stat`` over the sample ``x`` (one entry per episode).

    Returns ``(point, lo, hi)``. NaNs are dropped. With < 2 finite entries the
    CI is (nan, nan) — never silently 0-width.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"),) * 3
    point = float(stat(x))
    if x.size < 2:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = stat(x[idx], axis=1)
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    return point, float(lo), float(hi)


def fmt_ci(x, n_boot=10000, seed=0, prec=1):
    """``mean [lo, hi]`` as a compact string (n=..)."""
    p, lo, hi = bootstrap_ci(x, n_boot=n_boot, seed=seed)
    n = int(np.sum(np.isfinite(np.asarray(x, dtype=np.float64))))
    if not np.isfinite(lo):
        return f"{p:.{prec}f} [n={n}]"
    return f"{p:.{prec}f} [{lo:.{prec}f}, {hi:.{prec}f}] (n={n})"


def compare(x, y, name_x="X", name_y="Y", n_boot=10000, seed=0):
    """The campaign CI rule (spec §3.3), applied to two episode samples.

    A claim "X > Y" is admissible iff the bootstrap 95% CIs do not overlap, OR
    the mean gap is >= 3x the pooled std. Returns a dict with the numbers and a
    ``verdict`` in {">", "<", "~"} plus a one-line human summary. "~" means
    "not separated at the campaign's evidence bar" — NOT "equal".
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    px, lox, hix = bootstrap_ci(x, n_boot=n_boot, seed=seed)
    py, loy, hiy = bootstrap_ci(y, n_boot=n_boot, seed=seed)
    gap = px - py
    if x.size > 1 and y.size > 1:
        pooled = float(np.sqrt(
            ((x.size - 1) * np.var(x, ddof=1) + (y.size - 1) * np.var(y, ddof=1))
            / max(1, (x.size + y.size - 2))
        ))
    else:
        pooled = float("nan")
    disjoint = bool(np.isfinite(lox) and np.isfinite(loy) and (lox > hiy or loy > hix))
    big = bool(np.isfinite(pooled) and pooled > 0 and abs(gap) >= 3.0 * pooled)
    separated = disjoint or big
    verdict = "~" if not separated else (">" if gap > 0 else "<")
    reason = ("CIs disjoint" if disjoint else
              ("gap >= 3x pooled std" if big else "not separated (CIs overlap "
                                                  "and gap < 3x pooled std)"))
    summary = (f"{name_x} = {fmt_ci(x, n_boot, seed)} {verdict} "
               f"{name_y} = {fmt_ci(y, n_boot, seed)}  "
               f"[gap {gap:+.1f}, pooled std {pooled:.1f}; {reason}]")
    return {"mean_x": px, "ci_x": (lox, hix), "n_x": int(x.size),
            "mean_y": py, "ci_y": (loy, hiy), "n_y": int(y.size),
            "gap": gap, "pooled_std": pooled, "separated": separated,
            "verdict": verdict, "reason": reason, "summary": summary}


# ==========================================================================
#  debug file
# ==========================================================================
class DebugReport:
    """A markdown debug file that also echoes to stdout.

    Usage::

        rep = DebugReport("diag_out/e2/e2.md", title="E2 — privileged cancellation")
        rep.h2("beta grid"); rep.line("..."); rep.table(hdr, rows)
        rep.verdict("V1 existence-control", passed)
        rep.close()
    """

    def __init__(self, path, title, subtitle=None, echo=True):
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._f = open(self.path, "w", encoding="utf-8")
        self.echo = echo
        self.verdicts = {}
        self._write(f"# {title}\n")
        if subtitle:
            self._write(f"*{subtitle}*\n")
        self._write(f"\n`generated {time.strftime('%Y-%m-%d %H:%M:%S')}`  "
                    f"`cmd: {' '.join(sys.argv)}`\n")
        if echo:
            print(f"[DIAG] debug report -> {self.path}", flush=True)

    # -- primitives --------------------------------------------------------
    def _write(self, s):
        self._f.write(s + "\n")
        self._f.flush()
        if self.echo:
            print(s, flush=True)

    def h1(self, s):
        self._write(f"\n# {s}\n")

    def h2(self, s):
        self._write(f"\n## {s}\n")

    def h3(self, s):
        self._write(f"\n### {s}\n")

    def line(self, s=""):
        self._write(s)

    def kv(self, key, val):
        self._write(f"* **{key}**: {val}")

    def note(self, s):
        self._write(f"\n> {s}\n")

    def table(self, header, rows, align=None):
        """Markdown table. ``rows`` is a list of lists (str()'d)."""
        header = [str(h) for h in header]
        self._write("")
        self._write("| " + " | ".join(header) + " |")
        self._write("|" + "|".join(["---" if align is None else align] * len(header)) + "|")
        for r in rows:
            self._write("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
        self._write("")

    def verdict(self, name, passed, detail=None):
        """Record and print a PASS/FAIL. Collected into ``self.verdicts``."""
        self.verdicts[name] = bool(passed)
        tag = "PASS" if passed else "FAIL"
        self._write(f"\n**[{tag}] {name}**" + (f" — {detail}" if detail else "") + "\n")

    def close(self):
        self._write("\n---\n")
        if self.verdicts:
            self._write("**verdicts**: " + ", ".join(
                f"{k}={'PASS' if v else 'FAIL'}" for k, v in self.verdicts.items()))
        self._f.close()
        if self.echo:
            print(f"[DIAG] debug report written: {self.path}", flush=True)


# ==========================================================================
#  csv
# ==========================================================================
def write_csv(path, header, rows):
    """Tiny dependency-free CSV writer (pandas is not assumed anywhere)."""
    import csv as _csv

    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    return path
