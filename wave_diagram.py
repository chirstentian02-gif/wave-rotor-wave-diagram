# -*- coding: utf-8 -*-
"""
Wave diagrams for a 2D through-flow wave rotor turbine (ANSYS Fluent URANS)

Method follows the MASc thesis "Rujun Tian, SFU, 2025":
  - Geometry / port timing ............ Table 3.1, Table 5.4
  - Angle -> arc length (D*pi*a/360) .. Table 3.2
  - One cycle = 120 deg (3 cycles/rev)  Section 3.3.2
  - Data taken on the channel centre line at the periodic boundary, every
    ~2 deg (11 time steps), compiled over one cycle ......... Section 3.7
  - Last converged cycle used ............................... Section 3.6
  - Instantaneous power P = T*rpm/9550*1000, T = F*r ........ Section 3.7

Input: a folder of Fluent ASCII exports (one file per time step, name ends with
"-<time step>", e.g. data-3135 / test-3102) with columns
    nodenumber x-coordinate y-coordinate pressure x-velocity temperature
Optional: a Fluent report file (e.g. sing-f.out) with columns
    "Time Step" "<force>" "flow-time"

Usage examples:
    python wave_diagram.py "D:/opt/LPA_closing_opti2@design11"
    python wave_diagram.py "D:/opt/Inlet_opti1@baseline" "D:/opt/HPG_closing_opti2@design9" --vars pressure
    python wave_diagram.py --list-models
"""

import argparse
import glob
import math
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.interpolate import griddata

# ============================================================================
# 1. Rotor and solver parameters (Table 3.1 / Table 3.5)
# ============================================================================
ROTOR_DIAMETER = 0.060           # m
ROTOR_RADIUS = ROTOR_DIAMETER / 2
CYCLE_DEG = 120.0                # one cycle = 120 deg (3 cycles per revolution)
RPM = 32000                      # design speed
DT = 1e-6                        # time step size, s
ARC_PER_DEG = math.pi * ROTOR_DIAMETER / 360.0     # m of arc per degree (Table 3.2)
CYCLE_ARC = CYCLE_DEG * ARC_PER_DEG                 # 62.83 mm = H/3
DEG_PER_STEP = RPM * 360.0 / 60.0 * DT              # 0.192 deg per time step

# Absolute y (m) where port angle 0 lies in the Fluent model. The ports repeat
# every CYCLE_ARC. Checked against inflow/outflow at the channel ends (see
# --check-ports): for the thesis models it is y = 0.
PORT_ORIGIN_Y = 0.0

# ============================================================================
# 2. Port timing presets (deg)
#    L1 = HPG inlet, L2 = LPA inlet (left);  R1 = HPA outlet, R2 = LPG outlet (right)
# ============================================================================
BASELINE_PORTS = {
    "L1_open": 3.3, "L1_close": 31.0,      # HPG
    "L2_open": 73.0, "L2_close": 103.5,    # LPA
    "R1_open": 12.7, "R1_close": 36.0,     # HPA
    "R2_open": 52.0, "R2_close": 90.2,     # LPG
}


def _ports(**changes):
    p = dict(BASELINE_PORTS)
    p.update(changes)
    return p


# Designs 1-7 and Cases 1-2 only change inlet angle / channel profile, so they
# keep the baseline timing. Table 5.4 changes the inlet closing angles.
MODEL_PORTS = {
    "baseline": _ports(),
    **{f"design{i}": _ports() for i in range(1, 8)},
    "case1": _ports(),
    "case2": _ports(),
    "design8": _ports(L1_close=21.0),                     # HPG closes 10 deg earlier
    "design9": _ports(L1_close=41.0),                     # HPG closes 10 deg later
    "design10": _ports(L2_close=93.5),                    # LPA closes 10 deg earlier
    "design11": _ports(L2_close=113.5),                   # LPA closes 10 deg later
    "case3": _ports(L1_close=41.0, L2_close=113.5),
    "case4": _ports(L1_close=41.0, L2_close=113.5),
    "case5": _ports(L1_close=41.0, L2_close=103.5),
}

PORTS = [
    # key, label, side, colour
    ("L1", "HPG", "left", "#d62728"),
    ("L2", "LPA", "left", "#1f77b4"),
    ("R1", "HPA", "right", "#d62728"),
    ("R2", "LPG", "right", "#1f77b4"),
]

# ============================================================================
# 3. Plotted variables
# ============================================================================
VARIABLES = {
    # column: (label, unit, scale, colour map, (vmin, vmax) in displayed units)
    "pressure": ("Pressure", "kPa", 1e-3, "jet", (-50, 175)),
    "x-velocity": ("X velocity", "m/s", 1.0, "coolwarm", (-200, 400)),
    "temperature": ("Temperature", "K", 1.0, "plasma", (600, 1000)),
}

COLUMN_ALIASES = {"x-coordinate": "x", "y-coordinate": "y"}
FILE_RE = re.compile(r"-(\d+)$")


# ============================================================================
# 4. Loading
# ============================================================================
class Case:
    def __init__(self, label, data_dir, case_dir, model):
        self.label = label
        self.data_dir = data_dir
        self.case_dir = case_dir
        self.model = model
        self.ports = MODEL_PORTS[model]
        self.frames = []          # list of (time step, DataFrame)


def parse_case_arg(arg):
    """'PATH[@MODEL]' -> (path, model). PATH may be the case folder or its data folder."""
    path, model = arg, "baseline"
    if "@" in arg:
        path, model = arg.rsplit("@", 1)
    model = model.strip().lower().replace(" ", "")
    if model not in MODEL_PORTS:
        sys.exit(f"Unknown model '{model}'. Use --list-models.")
    path = os.path.abspath(path)
    if os.path.isdir(os.path.join(path, "data")):
        return os.path.join(path, "data"), path, model
    return path, os.path.dirname(path), model


def read_line_file(path):
    """Read one Fluent ASCII export (whitespace or comma separated)."""
    df = pd.read_csv(path, sep=r"[\s,]+", engine="python")
    df.columns = [COLUMN_ALIASES.get(c.strip(), c.strip()) for c in df.columns]
    missing = {"x", "y"} - set(df.columns)
    if missing:
        raise ValueError(f"missing columns {missing}")
    return df.dropna(axis=1, how="all")


def load_case(case):
    files = [p for p in glob.glob(os.path.join(case.data_dir, "*")) if FILE_RE.search(os.path.basename(p))]
    for p in sorted(files, key=lambda p: int(FILE_RE.search(os.path.basename(p)).group(1))):
        try:
            case.frames.append((int(FILE_RE.search(os.path.basename(p)).group(1)), read_line_file(p)))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {os.path.basename(p)}: {e}")
    if not case.frames:
        sys.exit(f"No data files '*-<step>' found in {case.data_dir}")
    print(f"[{case.label}] {len(case.frames)} files, steps {case.frames[0][0]}-{case.frames[-1][0]}, model={case.model}")


# ============================================================================
# 5. Angles
# ============================================================================
def channel_angle(df):
    """Rotor angle of the channel (deg in [0,120)) from the y of its inlet end.

    The channel ends sit at the same y (symmetric camber), and the ports open
    and close when the ends pass them, so the inlet end defines the timing.
    """
    end = df.nsmallest(2, "x")
    y_end = end["y"].mean()
    return ((y_end - PORT_ORIGIN_Y) / ARC_PER_DEG) % CYCLE_DEG, y_end


STEPS_PER_CYCLE = int(round(CYCLE_DEG / DEG_PER_STEP))   # 625


def select_cycle(case, which):
    """Return [(step, theta, y_end, df)]; 'last' keeps the final 120 deg of time steps.

    Every snapshot is placed at the stator angle of its channel, so snapshots
    from any converged cycle can be combined (the flow is periodic).
    """
    rows = [(step, *channel_angle(df), df) for step, df in case.frames]
    if which == "last":
        top = rows[-1][0]
        rows = [r for r in rows if r[0] > top - STEPS_PER_CYCLE]
    th = np.sort([r[1] for r in rows])
    gap = np.max(np.diff(np.concatenate([th, [th[0] + CYCLE_DEG]])))
    print(f"  using {len(rows)} files ({which} cycle), largest angle gap {gap:.1f} deg")
    return rows


class ChannelTracker:
    """Absolute y of the tracked channel's inlet end at any time step.

    The rotor mesh moves by ARC_PER_DEG*DEG_PER_STEP per step and wraps with
    its own periodic length (an integer number of channels, which need not be
    exactly 120 deg), so the period is measured from the data.
    """

    def __init__(self, frames):
        self.v = ARC_PER_DEG * DEG_PER_STEP
        pts = [(s, channel_angle(df)[1]) for s, df in frames]
        self.s0, self.y0 = pts[0]
        jumps = [y1 + self.v * (s2 - s1) - y2 for (s1, y1), (s2, y2) in zip(pts, pts[1:])
                 if y2 < y1 - CYCLE_ARC / 2]
        self.period = float(np.median(jumps)) if jumps else None
        ys = [y for _, y in pts]
        self.ylo = (max(ys) - self.period + min(ys)) / 2 if self.period else None
        err = max(abs(self.y(s) - y) for s, y in pts)
        info = f"rotor wrap {self.period * 1000:.2f} mm" if self.period else "no rotor wrap in data"
        print(f"  channel tracking: {info}, max position error {err * 1000:.3f} mm")

    def y(self, step):
        y = self.y0 + self.v * (np.asarray(step, dtype=float) - self.s0)
        if self.period:
            y = self.ylo + np.mod(y - self.ylo, self.period)
        return y

    def angle(self, step):
        return np.mod((self.y(step) - PORT_ORIGIN_Y) / ARC_PER_DEG, CYCLE_DEG)


def build_points(rows, variable, frame):
    """Point cloud (x mm along channel, angle deg, value) for one variable."""
    scale = VARIABLES[variable][2]
    x0 = min(df["x"].min() for _, _, _, df in rows)
    xs, angs, vals = [], [], []
    for _, th, y_end, df in rows:
        ang = np.full(len(df), th % CYCLE_DEG)
        if frame == "stator":
            # Keep the channel camber: each node at its own absolute position
            ang = ang + (df["y"].to_numpy() - y_end) / ARC_PER_DEG
        xs.append((df["x"].to_numpy() - x0) * 1000.0)
        angs.append(ang)
        vals.append(df[variable].to_numpy() * scale)
    return np.concatenate(xs), np.concatenate(angs), np.concatenate(vals)


def interpolate(x, ang, val, nx=300, na=481):
    """Periodic (120 deg) linear interpolation on a regular grid."""
    xx = np.concatenate([x, x, x])
    aa = np.concatenate([ang - CYCLE_DEG, ang, ang + CYCLE_DEG])
    vv = np.concatenate([val, val, val])
    gx, ga = np.meshgrid(np.linspace(x.min(), x.max(), nx), np.linspace(0, CYCLE_DEG, na))
    gv = griddata((xx, aa), vv, (gx, ga), method="linear", rescale=True)
    return gx, ga, gv


# ============================================================================
# 6. Port check: measured open windows vs. configured timing
# ============================================================================
def _windows(theta, flag):
    """Contiguous angle windows where flag is True (theta sorted, periodic)."""
    out, start = [], None
    for t, f in zip(theta, flag):
        if f and start is None:
            start = t
        if not f and start is not None:
            out.append((start, t))
            start = None
    if start is not None:
        out.append((start, theta[-1]))
    return out


def check_ports(case, rows, threshold=30.0):
    if "x-velocity" not in rows[0][3].columns:
        return
    th, uin, uout = [], [], []
    for _, t, _, df in rows:
        th.append(t % CYCLE_DEG)
        uin.append(df.nsmallest(4, "x")["x-velocity"].mean())
        uout.append(df.nlargest(4, "x")["x-velocity"].mean())
    order = np.argsort(th)
    th, uin, uout = np.array(th)[order], np.array(uin)[order], np.array(uout)[order]
    p = case.ports
    fmt = lambda ws: ", ".join(f"{a:.0f}-{b:.0f}" for a, b in ws) or "none"
    print(f"  port check (|u|>{threshold:g} m/s at channel ends, resolution ~2 deg):")
    print(f"    inlet  flow  measured: {fmt(_windows(th, np.abs(uin) > threshold))}"
          f" | set HPG {p['L1_open']}-{p['L1_close']}, LPA {p['L2_open']}-{p['L2_close']}")
    print(f"    outlet flow  measured: {fmt(_windows(th, np.abs(uout) > threshold))}"
          f" | set HPA {p['R1_open']}-{p['R1_close']}, LPG {p['R2_open']}-{p['R2_close']}")


# ============================================================================
# 7. Power from force report
# ============================================================================
def load_power(case, force_file):
    path = force_file if os.path.isabs(force_file) else os.path.join(case.case_dir, force_file)
    if not os.path.isfile(path):
        return None
    data = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            try:
                data.append([float(v) for v in parts[:3]])
            except ValueError:
                continue                      # header lines
    a = np.array([d for d in data if len(d) == 3])
    step, force = a[:, 0], a[:, 1]
    torque = force * ROTOR_RADIUS
    power = torque * RPM / 9550.0 * 1000.0    # W  (= T * omega)

    # Stator angle of the channel at each step, tracked from the data files of
    # the same run (assumes the force wall and the exported line belong to the
    # same channel).
    angle = case.tracker.angle(step)

    last = step > step.max() - STEPS_PER_CYCLE
    print(f"  power from {os.path.basename(path)}: mean of all steps = {power[step > 0].mean():.2f} W, "
          f"last cycle = {power[last].mean():.2f} W")
    o = np.argsort(angle[last])
    return angle[last][o], power[last][o], os.path.basename(path)


# ============================================================================
# 8. Plotting
# ============================================================================
def draw_ports(ax, ports, xmin, xmax, events):
    halo = [pe.withStroke(linewidth=3, foreground="black")]
    for key, label, side, colour in PORTS:
        a0, a1 = ports[f"{key}_open"], ports[f"{key}_close"]
        x = xmin if side == "left" else xmax
        ax.plot([x, x], [a0, a1], color=colour, lw=6, solid_capstyle="butt", clip_on=False, zorder=6)
        dx = 0.03 * (xmax - xmin) * (1 if side == "left" else -1)
        ax.text(x + dx, (a0 + a1) / 2, label, ha="left" if side == "left" else "right", va="center",
                color="white", fontsize=11, fontweight="bold", path_effects=halo, zorder=7)
        if events:
            for a in (a0, a1):
                ax.axhline(a, color="0.2", lw=0.6, ls=(0, (4, 4)), alpha=0.6, zorder=5)


def plot_variable(cases_rows, variable, args, powers):
    label, unit, _, cmap, (vmin, vmax) = VARIABLES[variable]
    n = len(cases_rows)
    has_power = any(p is not None for p in powers)
    # columns: n diagrams | colour bar | (spacer | power)
    widths = [1] * n + [0.07] + ([0.35, 1.0] if has_power else [])
    fig = plt.figure(figsize=(3.6 * n + (4.2 if has_power else 0) + 1.2, 7.5))
    gs = fig.add_gridspec(1, len(widths), width_ratios=widths, wspace=0.12)
    axes = [fig.add_subplot(gs[0, 0])]
    axes += [fig.add_subplot(gs[0, i], sharey=axes[0]) for i in range(1, n)]
    cax = fig.add_subplot(gs[0, n])
    if has_power:
        axes.append(fig.add_subplot(gs[0, n + 2], sharey=axes[0]))
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    levels = np.linspace(vmin, vmax, 61)

    cf = None
    for ax, (case, rows) in zip(axes, cases_rows):
        x, ang, val = build_points(rows, variable, args.frame)
        gx, ga, gv = interpolate(x, ang, val)
        cf = ax.contourf(gx, ga, np.clip(gv, vmin, vmax), levels=levels, cmap=cmap, extend="both")
        ax.set_xlim(gx.min(), gx.max())
        draw_ports(ax, case.ports, gx.min(), gx.max(), args.events)
        ax.set_title(case.label, fontsize=12)
        ax.set_xlabel("Channel length (mm)")
    axes[0].set_ylabel("Angle (°)")
    axes[0].set_ylim(0, CYCLE_DEG)
    axes[0].set_yticks(np.arange(0, CYCLE_DEG + 1, 20))

    if has_power:
        axp = axes[-1]
        same_timing = all(c.ports == cases_rows[0][0].ports for c, _ in cases_rows)
        for (case, _), p in zip(cases_rows, powers):
            if p is None:
                continue
            line, = axp.plot(p[1], p[0], lw=1.2, label=case.label)
            if not same_timing:      # inlet closing angles of this case, in its colour
                for key in ("L1", "L2"):
                    axp.axhline(case.ports[f"{key}_close"], color=line.get_color(), lw=0.9, ls="--")
        axp.axvline(0, color="0.3", lw=0.8)
        # shade the inlet (HPG / LPA) windows
        ref = cases_rows[0][0].ports
        for key, lab, side, colour in PORTS:
            if side == "left":
                top = ref[f"{key}_close"] if same_timing else min(c.ports[f"{key}_close"] for c, _ in cases_rows)
                axp.axhspan(ref[f"{key}_open"], top, color=colour, alpha=0.08)
                axp.text(0.02, (ref[f"{key}_open"] + top) / 2, lab, transform=axp.get_yaxis_transform(),
                         color=colour, fontsize=9, fontweight="bold", va="center")
        axp.set_xlabel(f"Power, {powers[0][2] if powers[0] else 'force'} (W)")
        axp.set_title("Instantaneous power", fontsize=12)
        axp.grid(True, ls=":", alpha=0.5)
        if len(cases_rows) > 1:
            axp.legend(fontsize=8, loc="upper right")

    cbar = fig.colorbar(cf, cax=cax)
    cbar.locator = matplotlib.ticker.MaxNLocator(nbins=9)
    cbar.update_ticks()
    cbar.set_label(f"{label} ({unit})")
    fig.suptitle(f"{label} wave diagram", fontsize=14)

    os.makedirs(args.out, exist_ok=True)
    name = "_".join(re.sub(r"\W+", "", c.label) for c, _ in cases_rows)
    path = os.path.join(args.out, f"{variable}_{name}.png")
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    print(f"saved {path}")
    if args.show:
        plt.show()
    plt.close(fig)


# ============================================================================
# 9. Main
# ============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="Wave rotor wave diagrams from Fluent line exports.")
    ap.add_argument("cases", nargs="*", help="CASE_DIR[@MODEL], e.g. LPA_closing_opti2@design11")
    ap.add_argument("--labels", nargs="*", help="panel titles (default: folder names)")
    ap.add_argument("--vars", nargs="*", default=list(VARIABLES), choices=list(VARIABLES))
    ap.add_argument("--frame", choices=["stator", "rotor"], default="stator",
                    help="stator: keep channel camber (as in the thesis figures); rotor: straight channel")
    ap.add_argument("--cycle", choices=["last", "all"], default="last", help="data cycle to plot")
    ap.add_argument("--force-file", default="sing-f.out", help="force report in the case folder ('' to skip)")
    ap.add_argument("--events", action="store_true", help="draw dashed lines at every port open/close angle")
    ap.add_argument("--check-ports", action="store_true",
                    help="print rough measured inflow/outflow windows at the channel ends")
    ap.add_argument("--out", default=None, help="output folder (default: <first case>/wave_diagrams)")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--show", action="store_true", help="also open the figures")
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args(argv)

    if args.list_models or not args.cases:
        for m, p in MODEL_PORTS.items():
            print(f"{m:9s} HPG {p['L1_open']}-{p['L1_close']}  LPA {p['L2_open']}-{p['L2_close']}  "
                  f"HPA {p['R1_open']}-{p['R1_close']}  LPG {p['R2_open']}-{p['R2_close']}")
        return

    if not args.show:
        matplotlib.use("Agg")

    cases_rows, powers = [], []
    for i, arg in enumerate(args.cases):
        data_dir, case_dir, model = parse_case_arg(arg)
        label = args.labels[i] if args.labels and i < len(args.labels) else os.path.basename(case_dir)
        case = Case(label, data_dir, case_dir, model)
        load_case(case)
        rows = select_cycle(case, args.cycle)
        if args.check_ports:
            check_ports(case, rows)
        case.tracker = ChannelTracker(case.frames)
        powers.append(load_power(case, args.force_file) if args.force_file else None)
        cases_rows.append((case, rows))

    args.out = args.out or os.path.join(cases_rows[0][0].case_dir, "wave_diagrams")
    for var in args.vars:
        if all(var in rows[0][3].columns for _, rows in cases_rows):
            plot_variable(cases_rows, var, args, powers)
        else:
            print(f"skip {var}: column not in data")


if __name__ == "__main__":
    main()
