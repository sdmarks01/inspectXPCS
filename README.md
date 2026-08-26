# inspect_xpcs

A fast, interactive detector-frame and metadata viewer for XPCS data at APS 8-ID.

Frames are loaded cropped to the qmap ROI into RAM and displayed in a PyQtGraph
`ImageView`. A second window plots the two-time correlation function (TTCF) up
to the slider position. Qmaps are auto-detected from `results.hdf` files.

## TTCF engines

The TTCF is computed either with a plain NumPy `matmul` or with `boost_corr`'s
`TwotimeCorrelator` (torch, GPU-capable). Pick with `--engine`, or the Engine
combo in the TTCF window; pick the torch device with `--device` or the Device
combo. Both engines give the same matrix, so the choice is purely about speed —
changing either combo recomputes the current ROI immediately.

If `boost_corr` / `torch` are not installed, the Engine combo still lists
`boost_corr`, greyed out, with the reason in its tooltip.

The TTCF ROI persists between runs. It is stored in detector coordinates, so it
lands on the same patch of detector even though the qmap crop moves from scan to
scan. The viewer opens on the last frame and correlates that ROI immediately, so
the full map is on screen at startup.

## Requirements

```
pip install h5py hdf5plugin numpy pyqtgraph pyqt5 matplotlib
```

Optional, for the GPU-capable TTCF engine:

```
pip install boost_corr torch
```

## Usage

```
inspect_xpcs.py DATA [--qmap QMAP] [--results RESULTS] [--every N] [--pad PIX]
                     [--log] [--cmap CMAP] [--metadata KEY ...] [--dt SEC]
                     [--engine {auto,numpy,boost}] [--device DEV]
```

`DATA` is an HDF5 data file, given either as a path or as a bare stem, in which
case it is searched for under `DATA_DIR`.

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--every N` | use every Nth frame (default 1) |
| `--pad PIX` | padding around the qmap bounding box (default 10) |
| `--log` | start in log intensity scale |
| `--cmap` | colormap name (default `inferno`) |
| `--metadata KEY ...` | NDAttribute keys to plot alongside the frames, e.g. `--metadata biologic_current biologic_voltage` |
| `--dt SEC` | frame period in seconds; overrides the value read from the HDF |
| `--engine` | `numpy`, `boost`, or `auto` (boost if importable) |
| `--device` | torch device for the boost engine: `auto`, `cpu`, `cuda`, `cuda:0`, `mps` |

Example:

```
inspect_xpcs.py my_scan_00123 --log --metadata biologic_current biologic_voltage
```

## Configuration

The default search paths for data and analysis are set at the top of
`inspect_xpcs.py` and point at an APS 8-ID-E experiment directory:

```python
DATA_DIR = Path('/gdata/dm/8ID/8IDE/2026-2/marks202606/data')
BOTH_DIR = Path('/gdata/dm/8ID/8IDE/2026-2/marks202606/analysis/Both')
```

Edit these for a different experiment, or pass explicit paths on the command
line.

The shebang runs the script inside a conda environment named
`mwi_xpcs_analysis`. Change it, or invoke via `python inspect_xpcs.py`, if your
environment is named differently.

## To do

- Other sources of metadata (txt files vs. NDAttributes)
- Add Qxy, Qz values to cursor positions
- "Discover" masks instead of using qmaps
- Integrate bad-pixel mask
