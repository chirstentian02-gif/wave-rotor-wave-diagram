# Wave Rotor Wave Diagram

Python tool that builds **wave diagrams** (channel length × rotor angle) for a 2D through-flow wave rotor turbine simulated in ANSYS Fluent (transient URANS). It follows the method of the MASc thesis *Rujun Tian, Simon Fraser University, 2025*, and can place the **instantaneous power curve** beside the diagram on the same angle axis.

## What it does

| Step | Method | Thesis |
|---|---|---|
| Read snapshots | One Fluent ASCII export per time step (`data-3135`, `test-3102`, …), taken on the channel centre line at the periodic boundary every ~2° (11 steps) | §3.5, §3.7 |
| Angle of each snapshot | Absolute y of the channel inlet end → arc length → angle, `angle = y / (π·D/360)`, modulo 120° (3 cycles per revolution) | Table 3.2, §3.3.2 |
| Cycle selection | Last converged cycle (final 625 time steps = 120°) by default | §3.6 |
| Diagram | Periodic (120°) linear interpolation; `stator` frame keeps the channel camber like the thesis figures, `rotor` frame shows a straight channel | Fig. 4.2, 5.11, 5.16 |
| Ports | HPG / LPA (inlets, left) and HPA / LPG (outlets, right) drawn at their open–close angles; presets for every model | Table 3.1, 5.4 |
| Power | `T = F·r`, `P = T·rpm/9550 × 1000` from a Fluent force report (e.g. `sing-f.out`), placed at the true stator angle of the channel | §3.7, Fig. 5.17 |

**Channel tracking.** The rotor mesh wraps with its own periodic length (an integer number of channels, about 123 mm here, which is not exactly 2 × 120°). The script measures this length from the data and tracks the channel's position at every time step, so the power curve lines up with the ports. It prints the tracking error against the data files; for the thesis cases this error is 0.001 mm.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# one case: folder that contains "data/" and "sing-f.out", with its port-timing model
python wave_diagram.py "D:/optimization/LPA_closing_opti2@design11"

# side-by-side comparison with a shared colour bar and power panel
python wave_diagram.py "D:/opt/Inlet_opti1@case1" "D:/opt/HPG_closing_opti1@design8" "D:/opt/HPG_closing_opti2@design9" --labels "Case 1" "Design 8" "Design 9" --vars pressure

# list port-timing models (baseline, design1-11, case1-5)
python wave_diagram.py --list-models
```

| Option | Default | Meaning |
|---|---|---|
| `PATH@MODEL` | model `baseline` | case folder (or its `data` folder) and port-timing preset |
| `--vars` | all | `pressure` (kPa), `x-velocity` (m/s), `temperature` (K) |
| `--frame` | `stator` | `stator` keeps the channel camber; `rotor` straight channel |
| `--cycle` | `last` | `last` = final 120° of time steps, `all` = every file |
| `--force-file` | `sing-f.out` | force report in the case folder; `""` to skip the power panel |
| `--events` | off | dashed lines at every port open/close angle |
| `--check-ports` | off | print rough inflow/outflow windows measured at the channel ends |
| `--labels` | folder names | panel titles |
| `--out` | `<case>/wave_diagrams` | output folder for PNG files |
| `--dpi`, `--show` | 200, off | resolution; open the figures interactively |

Rotor and solver constants (D = 60 mm, 32,000 rpm, Δt = 1 µs), colour ranges and port presets are at the top of `wave_diagram.py`.

## Notes

- The script prints the power averaged over **all** time steps and over the **last cycle**. The single-channel power values in the thesis tables (e.g. 12.99, 6.72, 17.80, 13.43 W for Case 1 and Designs 8, 9 and 11) equal the mean over all time steps.
- The power curve assumes the force wall (`rotor_wall_single`) and the exported line belong to the same channel.
- `PORT_ORIGIN_Y = 0` (port angle 0 at absolute y = 0) was checked against the flow at the channel ends. If your geometry differs, compare with `--check-ports` and adjust it.
