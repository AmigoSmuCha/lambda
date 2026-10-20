#!/usr/bin/env python3
# .local/bin/lambda
"""fplot - plot f(x), f'(x) and an antiderivative, with automatic intercepts and range.

Usage:
    fplot "x**3 - 2*x"
    fplot "sin(x)/x" --range -20 20
    fplot "sin(x)" --range 0 pi         # also prints the definite integral
    fplot "x^2 - 4" --save out.png
"""
import argparse
import os
import re
import signal
import sys

import numpy as np
import sympy as sp
from sympy.printing.precedence import PRECEDENCE
from sympy.printing.str import StrPrinter
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

x = sp.Symbol("x", real=True)
TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


# ---------------------------------------------------------------- input aliases
# Alternative name -> SymPy name. Edit freely to add your own.
ALIASES = {
    # tangent / cotangent
    "tg": "tan", "cotg": "cot", "ctg": "cot",
    # inverse functions
    "arcsin": "asin", "arcsinus": "asin",
    "arccos": "acos", "arcos": "acos", "arccosinus": "acos",
    "arctan": "atan", "arctg": "atan",
    "arccot": "acot", "arccotg": "acot", "arcctg": "acot",
    "arcsec": "asec", "arccsc": "acsc",
    # secant / cosecant
    "sec": "sec", "scs": "sec", "sekans": "sec", "senkans": "sec", "secans": "sec",
    "csc": "csc", "cosec": "csc", "cosenkans": "csc", "kosekans": "csc",
    "cosekans": "csc", "cosecans": "csc",
    # misc
    "ln": "log", "arsinh": "asinh", "arcosh": "acosh", "artanh": "atanh",
}
FUNCS = ["asinh", "acosh", "atanh", "sinh", "cosh", "tanh", "asin", "acos", "atan",
         "acot", "asec", "acsc", "sin", "cos", "tan", "cot", "sec", "csc",
         "log", "exp", "sqrt", "abs"]
INVERTIBLE = ["sin", "cos", "tan", "cot", "sec", "csc", "sinh", "cosh", "tanh"]
NOT_LETTER_BEFORE, NOT_LETTER_AFTER = r"(?<![A-Za-z])", r"(?![A-Za-z])"


def _matching_paren(s, start):
    depth = 0
    for i in range(start, len(s)):
        depth += {"(": 1, ")": -1}.get(s[i], 0)
        if depth == 0:
            return i
    sys.exit(f"Unbalanced parentheses in '{s}'")


def preprocess(text):
    s = text.replace("⁻¹", "^-1").replace("²", "^2").replace("³", "^3").replace("**", "^")
    s = s.replace("π", "pi")

    # 1. aliases (longest first, whole words only, so 'tg' inside 'arctg' is untouched)
    for alias in sorted(ALIASES, key=len, reverse=True):
        s = re.sub(NOT_LETTER_BEFORE + alias + NOT_LETTER_AFTER, ALIASES[alias], s,
                   flags=re.IGNORECASE)

    # 2. inverse notation: sin^-1, tan^(-1), tg^-1 -> asin, atan
    inv = "|".join(INVERTIBLE)
    s = re.sub(NOT_LETTER_BEFORE + rf"({inv})\s*\^\s*\(?\s*-\s*1\s*\)?",
               lambda m: "a" + m.group(1), s)

    # 3. powers of functions: sin2(x), sin^2(x), arctg^3(x) -> (sin(x))^2
    fn = "|".join(FUNCS)
    pat = re.compile(NOT_LETTER_BEFORE + rf"({fn})\s*(?:\^\s*\(?\s*(\d+)\s*\)?|(\d+))\s*\(")
    while (m := pat.search(s)):
        open_i = m.end() - 1
        close_i = _matching_paren(s, open_i)
        power = m.group(2) or m.group(3)
        s = (s[:m.start()] + f"({m.group(1)}{s[open_i:close_i + 1]})^{power}"
             + s[close_i + 1:])
    return s


def parse(text):
    cleaned = preprocess(text)
    try:
        expr = parse_expr(cleaned, local_dict={"x": x, "e": sp.E, "pi": sp.pi},
                          transformations=TRANSFORMS)
    except Exception as exc:
        sys.exit(f"Could not parse '{text}' (read as '{cleaned}'): {exc}")
    unknown = expr.free_symbols - {x}
    if unknown:
        names = ", ".join(sorted(str(u) for u in unknown))
        sys.exit(f"Unknown name(s) in '{text}': {names}\n"
                 f"(read as: {cleaned}). Use x as the variable, or add an alias to ALIASES.")
    return expr


def real_roots(expr, lo=None, hi=None, limit=20):
    """Real x-intercepts: symbolic first, numeric scan as a fallback."""
    roots = set()
    try:
        sol = sp.solveset(sp.Eq(expr, 0), x, domain=sp.S.Reals)
        if isinstance(sol, sp.FiniteSet):
            roots = {float(r) for r in sol if r.is_real}
        elif lo is None:
            # infinite/implicit solution set (e.g. sin): scan a default window
            lo, hi = -10.0, 10.0
    except Exception:
        lo, hi = (-10.0, 10.0) if lo is None else (lo, hi)

    if not roots and lo is not None:
        f = sp.lambdify(x, expr, "numpy")
        xs = np.linspace(lo, hi, 4000)
        with np.errstate(all="ignore"):
            ys = np.asarray(f(xs), dtype=float) * np.ones_like(xs)
        for i in range(len(xs) - 1):
            a, b = ys[i], ys[i + 1]
            # a sign change; two zeros in a row are underflow (e^(-x²) far out), not a root
            crosses = a * b < 0 or (a == 0 and b != 0)
            if np.isfinite(a) and np.isfinite(b) and crosses and abs(a - b) < 1e3:
                try:
                    roots.add(float(sp.nsolve(expr, x, (xs[i] + xs[i + 1]) / 2)))
                except Exception:
                    pass
    return sorted({round(r, 10) for r in roots})[:limit]


def y_intercept(expr):
    try:
        v = expr.subs(x, 0)
        v = sp.limit(expr, x, 0) if v.has(sp.nan, sp.zoo) else v
        return float(v) if v.is_real and v.is_finite else None
    except Exception:
        return None


def auto_range(points):
    if not points:
        return -10.0, 10.0
    lo, hi = min(points), max(points)
    span = max(hi - lo, 4.0)
    pad = 0.5 * span
    mid = (lo + hi) / 2
    return min(lo - pad, mid - span), max(hi + pad, mid + span)


def evaluate(expr, xs):
    with np.errstate(all="ignore"):
        try:
            f = sp.lambdify(x, expr, ["scipy", "numpy"])
            ys = np.asarray(f(xs), dtype=complex) * np.ones_like(xs)
        except Exception:  # special functions scipy lacks: evaluate point by point
            f = sp.lambdify(x, expr, "mpmath")
            def safe(v):
                try:
                    return complex(f(v))
                except Exception:
                    return complex(np.nan)
            ys = np.array([safe(v) for v in xs])
    ys = np.where(np.abs(ys.imag) < 1e-9, ys.real, np.nan).astype(float)
    # break lines at asymptotes
    jumps = np.abs(np.diff(ys)) > 50 * (np.nanstd(ys) or 1)
    ys[1:][jumps] = np.nan
    return ys


# ---------------------------------------------------------------- one-line output
SUP = str.maketrans("0123456789+-=()nixyabcdekmt", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱˣʸᵃᵇᶜᵈᵉᵏᵐᵗ")
SUP_OK = set("0123456789+-=()nixyabcdekmt")  # nested powers like e^(-x²) stay as ^(...)


def sup(text):
    """Superscript if every character has a superscript form, else None."""
    t = text.replace(" ", "").replace("*", "")
    return t.translate(SUP) if t and set(t) <= SUP_OK else None


class NicePrinter(StrPrinter):
    def _print_Pow(self, expr, rational=False):
        b, e = expr.base, expr.exp
        roots = {sp.Rational(1, 2): "√", sp.Rational(1, 3): "∛", sp.Rational(1, 4): "∜"}
        if e in roots:
            inner = self._print(b)
            return roots[e] + (inner if b.is_Atom else f"({inner})")
        if e.is_Integer and e < 0:
            if e == -1:
                return "1/" + self.parenthesize(b, PRECEDENCE["Pow"], strict=True)
            return "1/" + self._print(sp.Pow(b, -e, evaluate=False))
        if e.is_Integer and e > 0 or (e.is_Symbol or e.is_Add or e.is_Mul) and not e.has(sp.Pow):
            s = sup(self._print(e))
            if s:
                # atan(x)**3 -> atan³(x)
                if isinstance(b, sp.Function) and len(b.args) == 1 and e.is_Integer:
                    return f"{self._print_func_name(b)}{s}({self._print(b.args[0])})"
                if b == sp.E:
                    return "e" + s
                return self.parenthesize(b, PRECEDENCE["Pow"], strict=True) + s
        return super()._print_Pow(expr, rational)

    def _print_func_name(self, f):
        return {"log": "ln"}.get(type(f).__name__, type(f).__name__)

    def _print_log(self, expr):
        return f"ln({self._print(expr.args[0])})"

    def _print_exp(self, expr):
        s = sup(self._print(expr.args[0]))
        return "e" + s if s else f"e^({self._print(expr.args[0])})"

    def _print_Exp1(self, expr):
        return "e"

    def _print_Pi(self, expr):
        return "π"

    def _print_Infinity(self, expr):
        return "∞"

    def _print_NegativeInfinity(self, expr):
        return "-∞"


def nice(expr):
    s = NicePrinter().doprint(expr)
    s = s.replace("**", "^").replace("*", "·")
    s = re.sub(r"(?<![\w.])(\d+)·(?=[A-Za-zπe√(])", r"\1", s)  # 2·x -> 2x
    return s


def fmt(v):
    return f"{v:.6g}"


# ---------------------------------------------------------------- definite integral
INF_WORDS = {"inf": "oo", "infinity": "oo", "infty": "oo", "∞": "oo", "oo": "oo"}


def parse_bound(text):
    t = text.strip()
    sign, core = ("-", t[1:].strip()) if t.startswith("-") else ("", t.lstrip("+").strip())
    if core.lower() in INF_WORDS:
        return -sp.oo if sign else sp.oo
    b = parse(t)
    if b.free_symbols or not b.is_real:
        sys.exit(f"--range bound '{text}' must be a number (e.g. 2, pi, -pi/2, sqrt(2), inf)")
    return b


def _mp(v):
    import mpmath
    return mpmath.inf if v == sp.oo else -mpmath.inf if v == -sp.oo else float(v)


class _Timeout(Exception):
    pass


def _with_timeout(seconds, fn):
    def handler(*_):
        raise _Timeout
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def definite_integral(expr, a, b):
    """Return (exact or None, float or None). (None, None) means it diverges."""
    exact = None
    try:
        res = _with_timeout(8, lambda: sp.integrate(expr, (x, a, b)))
        if not res.has(sp.Integral):
            if res.has(sp.oo, -sp.oo, sp.zoo, sp.nan, sp.AccumBounds):  # AccumBounds = oscillates
                return None, None
            exact = sp.nsimplify(res) if res.is_Float else res
            value = float(sp.re(sp.N(exact)))
            if exact.has(sp.exp_polar, sp.I, sp.polar_lift):  # correct but unreadable form
                return None, value
            return exact, value
    except Exception:
        pass
    # numeric fallback (mpmath tanh-sinh handles endpoint singularities)
    import mpmath
    f = sp.lambdify(x, expr, "mpmath")
    try:
        val, err = mpmath.quad(lambda t: f(t), [_mp(a), _mp(b)], error=True)
        val, err = float(mpmath.re(val)), float(abs(err))
        if not np.isfinite(val) or err > 1e-4 * max(1.0, abs(val)):
            return None, None  # numeric integration did not converge
        return None, val
    except Exception:
        return None, None


def abs_area(expr, a, b):
    """Total area between f and the x-axis, splitting at sign changes."""
    import mpmath
    f = sp.lambdify(x, expr, "mpmath")
    wa = float(a) if a.is_finite else -50.0  # sign-change scan window for infinite ends
    wb = float(b) if b.is_finite else 50.0
    inner = [r for r in real_roots(expr, wa, wb) if (a < r) == True and (r < b) == True]
    if inner and not (a.is_finite and b.is_finite):
        return None  # sign changes on an infinite interval: can't count them all
    cuts = [_mp(a)] + inner + [_mp(b)]
    try:
        total = sum(abs(mpmath.quad(lambda t: f(t), [p, q])) for p, q in zip(cuts, cuts[1:]))
        total = float(mpmath.re(total))
        return total if np.isfinite(total) else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Plot f, f' and ∫f with intercepts.")
    ap.add_argument("function", help='e.g. "x^3 - 2x", "sin(x)/x", "exp(-x^2)"')
    ap.add_argument("-r", "--range", nargs=2, metavar=("A", "B"),
                    help="x range [A, B] (expressions allowed: 0 pi, -2pi 2pi, 1 inf); "
                         "also computes the definite integral over it")
    ap.add_argument("-s", "--save", metavar="FILE",
                    help="save to file instead of opening a window")
    ap.add_argument("-p", "--points", type=int, default=2000,
                    help="sample count (default 2000)")
    ap.add_argument("-w", "--wait", action="store_true",
                    help="keep the terminal busy until the window closes (no detaching)")
    argv = sys.argv[1:]
    for i, tok in enumerate(argv[:-2]):  # let "-r -pi pi" work (not seen as flags)
        if tok in ("-r", "--range"):
            argv[i + 1], argv[i + 2] = " " + argv[i + 1], " " + argv[i + 2]
    args = ap.parse_args(argv)

    f = parse(args.function)
    df = sp.simplify(sp.diff(f, x))
    F = sp.integrate(f, x)
    has_integral = not F.has(sp.Integral)

    bounds = None
    if args.range:
        bounds = [parse_bound(b) for b in args.range]
        if (bounds[0] >= bounds[1]) == True:  # noqa: E712 (sympy comparison)
            sys.exit("--range: A must be smaller than B")
    if bounds and all(b.is_finite for b in bounds):
        lo, hi = float(bounds[0]), float(bounds[1])
    else:
        lo, hi = None, None
    roots = real_roots(f, lo, hi)
    y0 = y_intercept(f)
    crit = real_roots(df, lo, hi)

    if lo is None:
        lo, hi = auto_range(roots + crit + ([0.0] if y0 is not None else []))
        if bounds:  # infinite bound: show the finite end plus a stretch toward infinity
            a, b = bounds
            if a.is_finite:
                span = max(hi - float(a), 10.0)
                lo, hi = float(a) - 0.1 * span, float(a) + span
            elif b.is_finite:
                span = max(float(b) - lo, 10.0)
                lo, hi = float(b) - span, float(b) + 0.1 * span
            roots = [r for r in roots if lo <= r <= hi] or real_roots(f, lo, hi)
            crit = [c for c in crit if lo <= c <= hi] or real_roots(df, lo, hi)

    # --- text output ---
    print("f(x)   =", nice(f))
    print("f'(x)  =", nice(df))
    print("∫f dx  =", (nice(F) + " + C") if has_integral else "no closed form (numeric shown)")
    print("\nx-intercepts:", ", ".join(fmt(r) for r in roots) or "none found")
    print("y-intercept: ", fmt(y0) if y0 is not None else "none (undefined at x=0)")
    print("critical pts:", ", ".join(fmt(c) for c in crit) or "none found")
    print(f"x range:      [{fmt(lo)}, {fmt(hi)}]")

    if bounds:
        a, b = bounds
        exact, approx = definite_integral(f, a, b)
        head = f"\n∫ from {nice(a)} to {nice(b)} f dx"
        if approx is None:
            print(head, "diverges (or does not converge)")
        elif exact is None:
            print(head, f"≈ {fmt(approx)}  (numeric)")
        elif exact.is_Integer:
            print(head, "=", nice(exact))
        else:
            print(head, "=", f"{nice(exact)} ≈ {fmt(approx)}")
        area = abs_area(f, a, b)
        if (area is not None and approx is not None
                and abs(area - abs(approx)) > 1e-7 * max(1.0, area)):
            print(f"area under |f|      = {fmt(area)}  (counts parts below the x-axis as positive)")

    # --- plot ---
    if not args.save and not args.wait:
        detach()
    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = np.linspace(lo, hi, args.points)
    ys = evaluate(f, xs)
    dys = evaluate(df, xs)
    if has_integral:
        Fys = evaluate(F, xs)
        Flabel = rf"$\int f\,dx = {sp.latex(F)}$"
    else:  # numeric cumulative integral from lo
        dx = xs[1] - xs[0]
        Fys = np.nancumsum(np.nan_to_num(ys)) * dx
        Flabel = r"$\int_{x_{min}}^{x} f\,dt$ (numeric)"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs, ys, lw=2.2, label=rf"$f(x) = {sp.latex(f)}$")
    ax.plot(xs, dys, "--", lw=1.6, label=rf"$f'(x) = {sp.latex(df)}$")
    ax.plot(xs, Fys, ":", lw=1.8, label=Flabel)
    if bounds:  # shade the definite integral: blue above the axis, red below
        ok = np.isfinite(ys) & (xs >= float(bounds[0])) & (xs <= float(bounds[1]))
        ax.fill_between(xs, ys, 0, where=ok & (ys >= 0), color="tab:blue", alpha=0.15)
        ax.fill_between(xs, ys, 0, where=ok & (ys < 0), color="tab:red", alpha=0.15)

    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    for r in roots:
        if lo <= r <= hi:
            ax.plot(r, 0, "o", color="crimson", zorder=5)
            ax.annotate(f"({fmt(r)}, 0)", (r, 0), textcoords="offset points",
                        xytext=(5, -14), fontsize=8, color="crimson")
    if y0 is not None and lo <= 0 <= hi:
        ax.plot(0, y0, "s", color="darkgreen", zorder=5)
        ax.annotate(f"(0, {fmt(y0)})", (0, y0), textcoords="offset points",
                    xytext=(6, 6), fontsize=8, color="darkgreen")

    # sensible y limits from f only, ignoring blow-ups
    focus = np.isfinite(ys)
    if bounds:  # scale y to the integrated part, not to spikes outside it
        inside = focus & (xs >= float(bounds[0])) & (xs <= float(bounds[1]))
        focus = inside if inside.any() else focus
    finite = ys[focus]
    if finite.size:
        q1, q99 = np.percentile(finite, [2, 98])
        if bounds:  # keep the value at a finite bound in view (e.g. f(1) for 1/x³ on [1, ∞))
            for bnd in bounds:
                if bnd.is_finite:
                    v = evaluate(f, np.array([float(bnd)]))[0]
                    if np.isfinite(v) and abs(v) < 10 * max(abs(q1), abs(q99), 1):
                        q1, q99 = min(q1, v, 0), max(q99, v, 0)
        span = q99 - q1
        pad = 0.15 * span if span > 0 else 0.5
        ax.set_ylim(q1 - pad, q99 + pad)

    ax.set_xlim(lo, hi)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_title("fplot")
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=130)
        print(f"\nsaved: {args.save}")
    else:
        plt.show()


def detach():
    """Fork: the parent returns the prompt, the child keeps the window open
    in its own session, so Ctrl+C or closing the terminal won't kill it."""
    sys.stdout.flush()
    if os.fork() > 0:
        os._exit(0)  # parent: back to the shell
    os.setsid()      # child: new session, no controlling terminal
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):  # silence GUI warnings that would spill into the shell
        os.dup2(devnull, fd)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")  # numpy noise from NaNs at singularities
    signal.signal(signal.SIGINT, lambda *_: sys.exit(130))  # Ctrl+C: quiet exit, no traceback
    main()
